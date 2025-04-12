# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Haiku modules for the Diffuser model."""

from collections.abc import Sequence
from typing import Literal

from alphafold3.common import base_config
from alphafold3.jax.attention import attention_th 
from alphafold3.jax.attention import attention
from alphafold3.jax.gated_linear_unit import gated_linear_unit
from alphafold3.model import model_config
from alphafold3.model.components import haiku_modules as hm
from alphafold3.model.components import torch_modules as tm
from alphafold3.model.components import mapping
from alphafold3.model.components import mapping_th
# from alphafold3.model.diffusion import diffusion_transformer
from alphafold3.model.diffusion import diffusion_transformer_th as diffusion_transformer
from alphafold3.model.diffusion.attention_modules import torch_attn, F_attn
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math
import intel_extension_for_pytorch as ipex

def get_shard_size(
    num_residues: int, shard_spec: Sequence[tuple[int | None, int | None]]
) -> int | None:
  shard_size = shard_spec[0][-1]
  for num_residues_upper_bound, num_residues_shard_size in shard_spec:
    shard_size = num_residues_shard_size
    if (
        num_residues_upper_bound is None
        or num_residues <= num_residues_upper_bound
    ):
      break
  return shard_size

class TransitionBlock(nn.Module):
    """Transition block for transformer."""

    class Config(base_config.BaseConfig):
        num_intermediate_factor: int = 4
        use_glu_kernel: bool = True


    def __init__(
        self, config: Config, global_config: model_config.GlobalConfig, num_channels:int, *, name
    ):
        super().__init__()
        self.config = config
        self.num_channels = num_channels
        self.use_ipex_kernel = global_config.enable_ipex_kernel

        self.num_intermediate = int(num_channels * self.config.num_intermediate_factor)
        self.input_layer_norm = tm.LayerNorm(num_channels)
        self.transition1 = nn.Linear(num_channels, self.num_intermediate * 2, bias=False)
        self.transition2 = nn.Linear(self.num_intermediate, num_channels, bias=False)

        if self.use_ipex_kernel:
            self.linear_s = nn.Linear(self.num_channels, self.num_intermediate, bias=False)
            self.linear_m = nn.Linear(self.num_channels, self.num_intermediate, bias=False)
            self.ipex_transition = ipex.llm.modules.Linear2SiluMul(self.linear_s, self.linear_m)


    def load_ipex_params(self):
        linear_s, linear_m = self.transition1.weight.chunk(2, dim=0)
        self.linear_s.weight, self.linear_m.weight = nn.Parameter(linear_s), nn.Parameter(linear_m)

    def forward(self, act: torch.Tensor, broadcast_dim: int = 0) -> torch.Tensor:
        act = self.input_layer_norm(act)

        if self.use_ipex_kernel:
            c = self.ipex_transition(act)
        else:
            act = self.transition1(act)
            a, b = torch.chunk(act, 2, dim=-1)
            c = F.silu(a) * b

        return self.transition2(c)


class MSAAttention(nn.Module):
    """MSA Attention."""

    class Config(base_config.BaseConfig):
        num_head: int = 8

    def __init__(
        self, 
        config: Config, 
        global_config: model_config.GlobalConfig, 
        msa_channels: int, 
        pair_channels: int, 
        *, 
        name
    ):
        super(MSAAttention, self).__init__()
        self.config = config
        self.global_config = global_config

        self.value_dim = msa_channels // self.config.num_head

        self.pair_norm = tm.LayerNorm(pair_channels)
        self.pair_logits = nn.Linear(pair_channels, config.num_head, bias=False)
        
        self.act_norm = tm.LayerNorm(msa_channels)
        self.v_projection = nn.Parameter(torch.randn(msa_channels, self.config.num_head, self.value_dim), requires_grad=False)
        self.gating_query = nn.Linear(msa_channels, msa_channels, bias=False)
        self.output_projection = nn.Linear(msa_channels, msa_channels, bias=False)

    def forward(self, act: torch.Tensor, mask: torch.Tensor, pair_act: torch.Tensor) -> torch.Tensor:
        act = self.act_norm(act)
        
        pair_act = self.pair_norm(pair_act)
        logits = self.pair_logits(pair_act)

        logits = logits.permute(2, 0, 1)
        logits += 1e9 * (mask.max(dim=0)[0].bfloat16() - 1.0)
        weights = F.softmax(logits, dim=-1)
        v = torch.einsum('...i,ijk->...jk', act, self.v_projection)
        v_avg = torch.einsum('hqk, bkhc -> bqhc', weights, v)
        v_avg = v_avg.reshape(v_avg.shape[:-2] + (-1,))
        gate_values = self.gating_query(act)
        v_avg *= torch.sigmoid(gate_values)

        return self.output_projection(v_avg)


class GridSelfAttention(nn.Module):
    """Self attention that is either per-sequence or per-residue."""

    class Config(base_config.BaseConfig):
        num_head: int = 4

    def __init__(
        self,
        config: Config,
        global_config: model_config.GlobalConfig,
        num_channels: int, 
        transpose: bool,
        *,
        name: str,
    ):
        super(GridSelfAttention, self).__init__()
        self.config = config
        self.global_config = global_config
        self.transpose = transpose
        self.num_channels = num_channels
        self.num_head = config.num_head

        self.qkv_dim = max(num_channels // self.config.num_head, 16)
        self.act_norm = tm.LayerNorm(num_channels)
        self.pair_bias_projection = nn.Linear(num_channels, self.num_head, bias=False)
        self.q_projection = nn.Parameter(torch.randn(num_channels, self.num_head, self.qkv_dim), requires_grad=False)
        self.k_projection = nn.Parameter(torch.randn(num_channels, self.num_head, self.qkv_dim), requires_grad=False)
        self.v_projection = nn.Parameter(torch.randn(num_channels, self.num_head, self.qkv_dim), requires_grad=False)
        self.gating_query = nn.Linear(num_channels, num_channels, bias=False)
        self.output_projection = nn.Linear(num_channels, num_channels, bias=False)

    def _attention(self, act: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor, rm_bias=False) -> torch.Tensor:
        num_channels = act.shape[-1]
        assert num_channels % self.config.num_head == 0
        # Triton requires a minimum dimension of 16 for doing matmul.
        # qkv_dim = max(num_channels // self.config.num_head, 16)

        q = torch.einsum('...i, ijk -> ...jk', act, self.q_projection)
        k = torch.einsum('...i, ijk -> ...jk', act, self.k_projection)
        v = torch.einsum('...i, ijk -> ...jk', act, self.v_projection)


        bias = bias.unsqueeze(0)

        if hasattr(self.global_config,'attention_implementation') and self.global_config.attention_implementation == 'torch':
            weighted_avg = torch_attn(q, k, v, bias, mask)
        elif hasattr(self.global_config,'attention_implementation') and self.global_config.attention_implementation == 'F':
            weighted_avg = F_attn(q, k, v, bias, mask)
        else:
            weighted_avg = torch_attn(q, k, v, bias, mask)

        # shaper = lambda x: x.transpose(-3, -2) # ([bs], num_head, num_tokens, value_dim_per_head)
        # q, k, v = shaper(q), shaper(k), shaper(v)

        # if mask is not None:
        #     if bias is not None:
        #         bias = bias + (1e9 * (mask.to(bias.dtype) - 1.0)) # ([bs], 1, 1, num_tokens)
        #     else:
        #         bias = mask.bool()
        # weights = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        # return shaper(weights)
        # return weighted_avg

        weighted_avg = torch.reshape(weighted_avg, weighted_avg.shape[:-2] + (-1,))
        gate_values = torch.matmul(act,self.gating_query.weight)
        # return weighted_avg
        weighted_avg *= torch.sigmoid(gate_values)
        output = self.output_projection(weighted_avg)
        return output

    def attn_forward(self, q_data, m_data, bias, nonbatched_bias=torch.Tensor(),rm_bias=False):
        return self._attention(q_data, bias, nonbatched_bias,True)
        
    def forward(self, act: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
          act: [num_seq, num_res, channels] activations tensor
          pair_mask: [num_seq, num_res] mask of non-padded regions in the tensor.
            Only used in inducing points attention currently.

        Returns:
          Result of the self-attention operation.
        """
        assert len(act.shape) == 3
        assert len(pair_mask.shape) == 2

        pair_mask = pair_mask.permute(1, 0)
        act = self.act_norm(act)

        nonbatched_bias = self.pair_bias_projection(act)
        nonbatched_bias = nonbatched_bias.permute(2, 0, 1)
        pair_mask = pair_mask[:, None, None, :]

        if self.transpose:
            act = act.permute(1, 0, 2)

        act = self.attn_forward(act, None, bias=pair_mask.bool(), nonbatched_bias=nonbatched_bias)

        if self.transpose:
            act = act.permute(1, 0, 2)

        return act


class TriangleMultiplication(nn.Module):
    """Triangle Multiplication."""

    class Config(base_config.BaseConfig):
        equation: Literal['ikc,jkc->ijc', 'kjc,kic->ijc']
        use_glu_kernel: bool = True

    def __init__(
        self, config: Config, global_config: model_config.GlobalConfig, num_channels: int, *, name
    ):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.num_intermediate_channel = num_channels
        self.save_and_quit = False

        self.c_equation = {
            'ikc,jkc->ijc': 'cik,cjk->cij',
            'kjc,kic->ijc': 'ckj,cki->cij',
        }[self.config.equation]

        self.left_norm_input = tm.LayerNorm(num_channels)
        self.projection = nn.Linear(num_channels, num_channels * 2, bias=False)
        self.gate = nn.Linear(num_channels, num_channels * 2, bias=False)
        self.center_norm = tm.LayerNorm(num_channels)
        self.output_projection = nn.Linear(num_channels, num_channels, bias=False)
        self.gating_linear = nn.Linear(num_channels, num_channels, bias=False)

    def forward(self, act: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask[None, ...]
        act = self.left_norm_input(act)
        input_act = act
        if self.save_and_quit:
            torch.save(input_act,'align/tri_input_act')

        projection = self.projection(act)
        projection = projection.permute(2, 0, 1)
        projection *= mask
        gate = self.gate(act)
        gate = gate.permute(2, 0, 1)
        projection *= torch.sigmoid(gate)


        # 与直接chunk并不等价
        projection = projection.reshape(self.num_intermediate_channel, 2, *projection.shape[1:])
        a, b = torch.chunk(projection, 2, dim=1)
        a, b = torch.squeeze(a, 1), torch.squeeze(b, 1)
        # return a
        act = torch.einsum(self.c_equation, a, b)
        act = act.permute(1, 2, 0)
        act = self.center_norm(act)
        act = self.output_projection(act)

        # if self.save_and_quit:
        #     torch.save(act,'align/tri_output_projection')

        gate_out = self.gating_linear(input_act)
        act = act * torch.sigmoid(gate_out)
        return act


class OuterProductMean(nn.Module):
    """Computed mean outer product."""

    class Config(base_config.BaseConfig):
        chunk_size: int = 128
        num_outer_channel: int = 32

    def __init__(
        self,
        config: Config,
        global_config: model_config.GlobalConfig,
        input_channels,
        num_output_channel,
        *,
        name,
    ):
        super(OuterProductMean, self).__init__()
        self.global_config = global_config
        self.config = config
        self.num_output_channel = num_output_channel

        self.layer_norm_input = tm.LayerNorm(input_channels)
        self.left_projection = nn.Linear(input_channels, config.num_outer_channel, bias=False)
        self.right_projection = nn.Linear(input_channels, config.num_outer_channel, bias=False)
        self.output_w = nn.Parameter(torch.zeros(config.num_outer_channel, config.num_outer_channel, num_output_channel))
        self.output_b = nn.Parameter(torch.zeros(num_output_channel))

    def forward(self, act: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        act = self.layer_norm_input(act)

        left_act = mask * self.left_projection(act)
        right_act = mask * self.right_projection(act)

        def compute_chunk(left_act):
            # Make sure that the 'b' dimension is the most minor batch like dimension
            # so it will be treated as the real batch by XLA (both during the forward
            # and the backward pass)
            left_act = left_act.permute(0, 2, 1)
            act = torch.einsum('acb,ade->dceb', left_act, right_act)
            act = torch.einsum('dceb,cef->dbf', act, self.output_w) + self.output_b
            return act.permute(1, 0, 2)

        act = compute_chunk(left_act)

        epsilon = 1e-3
        norm = torch.einsum('abc,adc->bdc', mask.bfloat16(), mask.bfloat16())
        return act / (epsilon + norm)


class PairFormerIteration(nn.Module):
    """Single Iteration of Pair Former."""

    class Config(base_config.BaseConfig):
        """Config for PairFormerIteration."""

        num_layer: int
        pair_attention: GridSelfAttention.Config = base_config.autocreate()
        pair_transition: TransitionBlock.Config = base_config.autocreate()
        single_attention: diffusion_transformer.SelfAttentionConfig | None = None
        single_transition: TransitionBlock.Config | None = None
        triangle_multiplication_incoming: TriangleMultiplication.Config = (
            base_config.autocreate(equation='kjc,kic->ijc')
        )
        triangle_multiplication_outgoing: TriangleMultiplication.Config = (
            base_config.autocreate(equation='ikc,jkc->ijc')
        )
        shard_transition_blocks: bool = True

    def __init__(
        self,
        config: Config,
        global_config: model_config.GlobalConfig,
        num_channels: int,
        seq_channel: int | None = None,
        with_single=False,
        *,
        name,
    ):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.with_single = with_single
        self.triangle_multiplication_outgoing = TriangleMultiplication(config.triangle_multiplication_outgoing, global_config, num_channels,name='triangle_multiplication_outgoing')
        self.triangle_multiplication_incoming = TriangleMultiplication(config.triangle_multiplication_incoming, global_config, num_channels, name='triangle_multiplication_incoming')
        self.pair_attention1 = GridSelfAttention(config.pair_attention, global_config, num_channels, transpose=False, name='pair_attention1')
        self.pair_attention2 = GridSelfAttention(config.pair_attention, global_config, num_channels, transpose=True, name='pair_attention2')
        self.pair_transition = TransitionBlock(config.pair_transition, global_config, num_channels,  name='pair_transition')
        # print(f'In PairFormerIteration, in {name}: self.with_single',self.with_single)
        if name =='template_embedding_iteration': # confidence head
            config.single_attention.key_dim = 384
            config.single_attention.value_dim = 384
        if self.with_single:
            self.single_pair_logits_projection = nn.Linear(num_channels, self.config.single_attention.num_head, bias=False)
            self.single_pair_logits_norm = tm.LayerNorm(num_channels)
            self.single_attention = diffusion_transformer.SelfAttention(
                    num_channels=seq_channel,
                    config=config.single_attention,
                    global_config=global_config,
                    name='single_attention_'
                )
            self.single_transition = TransitionBlock(config.single_transition, global_config, seq_channel, name='single_transition')
        self.save_and_quit = False
    # @torch.compile
    def forward(
        self,
        act: torch.Tensor,
        pair_mask: torch.Tensor,
        single_act: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_residues = act.shape[0]
        # [num_res, num_token, seq_chan]
        # [num_res, num_token]
        act += self.triangle_multiplication_outgoing(act, pair_mask)
        # if self.save_and_quit:
        #     torch.save(act,'align/triangle_multiplication_outgoing')
        act += self.triangle_multiplication_incoming(act, pair_mask)
        # if self.save_and_quit:
        #     torch.save(act,'align/triangle_multiplication_incoming')
        act += self.pair_attention1(act, pair_mask)
        # if self.save_and_quit:
        #     torch.save(act,'align/pair_attention1')
        act += self.pair_attention2(act, pair_mask)
        # if self.save_and_quit:
        #     torch.save(act,'align/pair_attention2')
        act += self.pair_transition(act)
        # if self.save_and_quit:
        #     torch.save(act,'align/pair_transition')
        #     print('In pairFormerIteration, self.with_single =',self.with_single)
        #     print('quit at pairformer'); quit()

        if self.with_single:
            pair_logits = self.single_pair_logits_projection(self.single_pair_logits_norm(act))
            pair_logits = pair_logits.permute(2, 0, 1)

            single_act += self.single_attention(
                        single_act,
                        seq_mask,
                        pair_logits=pair_logits,)  # (num_heads, num_tokens, num_tokens)                   )
            single_act += self.single_transition(single_act, broadcast_dim=None)

            return act, single_act
        else:
            return act


class EvoformerIteration(nn.Module):
    """Single Iteration of Evoformer Main Stack."""

    class Config(base_config.BaseConfig):
        """Configuration for EvoformerIteration."""

        num_layer: int = 4
        msa_attention: MSAAttention.Config = base_config.autocreate()
        outer_product_mean: OuterProductMean.Config = base_config.autocreate()
        msa_transition: TransitionBlock.Config = base_config.autocreate()
        pair_attention: GridSelfAttention.Config = base_config.autocreate()
        pair_transition: TransitionBlock.Config = base_config.autocreate()
        triangle_multiplication_incoming: TriangleMultiplication.Config = (
            base_config.autocreate(equation='kjc,kic->ijc')
        )
        triangle_multiplication_outgoing: TriangleMultiplication.Config = (
            base_config.autocreate(equation='ikc,jkc->ijc')
        )
        shard_transition_blocks: bool = True


    def __init__(
        self,
        config: Config,
        global_config: model_config.GlobalConfig,
        msa_channels: int, 
        pair_channels: int, 
        name='evoformer_iteration',
    ):
        super(EvoformerIteration, self).__init__()
        self.config = config
        self.global_config = global_config

        self.outer_product_mean = OuterProductMean(config.outer_product_mean, global_config, input_channels=msa_channels, num_output_channel=pair_channels, name='outer_product_mean')
        self.msa_attention1 = MSAAttention(config.msa_attention, global_config,  msa_channels=msa_channels, pair_channels=pair_channels, name='msa_attention1')
        self.msa_transition = TransitionBlock(config.msa_transition, global_config, num_channels=msa_channels, name='msa_attention')
        self.triangle_multiplication_outgoing = TriangleMultiplication(config.triangle_multiplication_outgoing, global_config, num_channels=pair_channels, name='triangle_multiplication_outgoing')
        self.triangle_multiplication_incoming = TriangleMultiplication(config.triangle_multiplication_incoming, global_config, num_channels=pair_channels, name='triangle_multiplication_incoming')
        self.pair_attention1 = GridSelfAttention(config.pair_attention, global_config, num_channels=pair_channels, transpose=False, name='pair_attention1')
        self.pair_attention2 = GridSelfAttention(config.pair_attention, global_config, num_channels=pair_channels, transpose=True, name='pair_attention2')
        self.pair_transition = TransitionBlock(config.pair_transition, global_config, num_channels=pair_channels, name='pair_transition')

    def forward(self, activations: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        msa_act, pair_act = activations['msa'], activations['pair']
        msa_mask, pair_mask = masks['msa'], masks['pair']

        num_residues = pair_act.shape[0]

        
        pair_act += self.outer_product_mean(msa_act, msa_mask)
        # torch.save(pair_act,'align/msa_pair_act')
        msa_act += self.msa_attention1(msa_act, msa_mask, pair_act)
        # torch.save(pair_act,'align/msa_attention1')
        msa_act += self.msa_transition(msa_act)
        # torch.save(pair_act,'align/msa_transition')
        pair_act += self.triangle_multiplication_outgoing(pair_act, pair_mask)
        pair_act += self.triangle_multiplication_incoming(pair_act, pair_mask)
        pair_act += self.pair_attention1(pair_act, pair_mask)
        pair_act += self.pair_attention2(pair_act, pair_mask)
        pair_act += self.pair_transition(pair_act)

        return {'msa': msa_act, 'pair': pair_act}