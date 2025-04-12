# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Diffusion transformer model."""

from alphafold3.common import base_config
from alphafold3.jax.gated_linear_unit import gated_linear_unit_th as gated_linear_unit
from alphafold3.model import model_config
from alphafold3.model.atom_layout import atom_layout_th as atom_layout
# from alphafold3.model.components import haiku_modules as hm
import haiku as hk
import jax
from jax import numpy as jnp
import torch
import torch.nn as nn
import torch.nn.functional as F
from alphafold3.model.components import torch_modules as tm
import intel_extension_for_pytorch as ipex
from alphafold3.model.diffusion.attention_modules import torch_attn, F_attn

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, num_features: int,  name: str = '', single_featrues:int=None,):
        super(AdaptiveLayerNorm, self).__init__()
        if single_featrues is None: single_featrues = num_features
        self.name = name
        # single attention 不需要初始化这个
        # print('init AdaptiveLayerNorm name is',name)
        # single_attention_ cross_att_block_0_cross_attentionq pair_transition_0ffw_ single_transition_1ffw_ transformer_block_3_self_attention
        use_singlge_cond = False

        if name!='single_attention_'  \
            and name!='single_transition_1ffw_' and name!= 'single_transition_0ffw_' \
            and name!='pair_transition_0ffw_' and name!='pair_transition_1ffw_':
            use_singlge_cond = True
        # print('AdaptiveLayerNorm, name:',name, 'use_singlge_cond',use_singlge_cond)

        if use_singlge_cond:
            self.layer_norm = tm.LayerNorm(
                num_features,
                name=f'{name}layer_norm',
                use_fast_variance=False,
                create_scale=False,
                create_offset=False,
            )
            self.single_cond_scale = nn.Linear(single_featrues, num_features, bias=True)
            self.single_cond_bias = nn.Linear(single_featrues, num_features, bias=False)
            self.single_cond_layer_norm = tm.LayerNorm(
                single_featrues, 
                use_fast_variance=False,
                create_offset=False,
            )
        else:
            self.layer_norm = tm.LayerNorm(num_features, name=f'{name}layer_norm', use_fast_variance=False)


    def forward(self, x: torch.Tensor, single_cond: torch.Tensor = None) -> torch.Tensor:
        if single_cond is None:
            x = self.layer_norm(x)
        else:
            x = self.layer_norm(x)
            single_cond = self.single_cond_layer_norm(single_cond)
            single_scale = self.single_cond_scale(single_cond)
            single_bias = self.single_cond_bias(single_cond)
            x = torch.sigmoid(single_scale) * x + single_bias
        return x

### 注意注意，这里的transition根据single_cond有不同的初始化方法！！还没实现
# TODO
class AdaptiveZeroInit(nn.Module):
    def __init__(self, num_channels, num_intermediates, single_channels, global_config, name):
        super(AdaptiveZeroInit, self).__init__()
        self.num_channels = num_channels
        self.global_config = global_config
        self.name = name
        use_singlge_cond = False
        if name!='single_attention_'  \
            and name!='single_transition_1ffw_' and name!= 'single_transition_0ffw_' \
            and name!='pair_transition_0ffw_' and name!='pair_transition_1ffw_':
            use_singlge_cond = True

        # print('AdaptiveZeroInit, name:',name, 'use_singlge_cond',use_singlge_cond)

        self.transition2 = nn.Linear(
            num_intermediates,
            num_channels,
            bias=False,
            # name=f'{name}transition2',
        )
        if use_singlge_cond:
            self.adaptive_zero_cond = nn.Linear(
                single_channels,
                num_channels,
                bias=True
                # name=f'{name}adaptive_zero_cond',
            )

    def forward(self, x, single_cond=None):
        output = self.transition2(x)
        if single_cond is not None:
            cond = self.adaptive_zero_cond(single_cond)
            output = torch.sigmoid(cond) * output

        return output
    
class TransitionBlock(nn.Module):
    def __init__(
        self,
        num_intermediate_factor: int,
        global_config,
        num_channels : int = 128,
        single_channels: int = None,
        use_glu_kernel: bool = True,
        name: str = '',
    ):
        super(TransitionBlock, self).__init__()
        self.num_channels = num_channels
        self.num_intermediates = num_intermediate_factor * num_channels
        self.use_ipex_kernel = global_config.enable_ipex_kernel
        self.name = name

        if single_channels is None: single_channels = num_channels

        # Adaptive LayerNorm
        self.adaptive_layernorm = AdaptiveLayerNorm(num_channels, single_featrues=single_channels, name=f'{name}ffw_')

        # Linear layer for transition
        self.ffw_transition1 = nn.Linear(num_channels, self.num_intermediates * 2, bias=False)

        # Adaptive Zero Init
        self.adaptive_zero_init = AdaptiveZeroInit(num_channels, self.num_intermediates, single_channels, global_config, name=f'{name}ffw_')

        if self.use_ipex_kernel:
            self.linear_s = nn.Linear(self.num_channels, self.num_intermediates, bias=False)
            self.linear_m = nn.Linear(self.num_channels, self.num_intermediates, bias=False)
            self.ipex_transition = ipex.llm.modules.Linear2SiluMul(self.linear_s, self.linear_m)


    def load_ipex_params(self):
        linear_s, linear_m = self.ffw_transition1.weight.chunk(2, dim=0)
        self.linear_s.weight, self.linear_m.weight = nn.Parameter(linear_s), nn.Parameter(linear_m)


    def forward(self, x: torch.Tensor, single_cond: torch.Tensor = None) -> torch.Tensor:
        x = self.adaptive_layernorm(x, single_cond)

        # if self.use_glu_kernel:
        #     # weights,_ = tm.torch_linear_get_params(
        #     #     x,
        #     #     num_output=self.num_intermediates * 2,
        #     #     initializer='relu',
        #     #     name=f'{self.name}ffw_transition1',
        #     # )
        #     weights = self.ffw_transition1.weight
        #     print(weights.shape)
        #     weights = weights.view(weights.shape[0], 2, self.num_intermediates)
        #     # x = self.ffw_transition1(x)
        #     c = gated_linear_unit.gated_linear_unit(x, weights)
        # else:

        if self.use_ipex_kernel:
            c = self.ipex_transition(x)
        else:
            x = self.ffw_transition1(x)
            a, b = torch.chunk(x, 2, dim=-1)
            c = F.silu(a) * b


        output = self.adaptive_zero_init(c, single_cond)
        return output


class SelfAttentionConfig(base_config.BaseConfig):
  num_head: int = 16
  key_dim: int | None = None
  value_dim: int | None = None

class SelfAttention(nn.Module):
    def __init__(
        self,
        config: SelfAttentionConfig,
        global_config,
        num_channels: int,
        single_channels: int = None,
        name: str = '',
    ):
        super(SelfAttention, self).__init__()
        self.name = name
        self.num_channels = num_channels
        self.config = config
        self.global_config = global_config

        # Sensible defaults for key_dim and value_dim
        self.key_dim = config.key_dim if config.key_dim is not None else num_channels
        self.value_dim = config.value_dim if config.value_dim is not None else num_channels
        self.single_channels = single_channels if single_channels is not None else num_channels
        self.num_head = config.num_head

        # Ensure key_dim and value_dim are divisible by num_head
        assert self.key_dim % self.num_head == 0, f'{self.key_dim=} % {self.num_head=} != 0'
        assert self.value_dim % self.num_head == 0, f'{self.value_dim=} % {self.num_head=} != 0'

        self.key_dim_per_head = self.key_dim // self.num_head
        self.value_dim_per_head = self.value_dim // self.num_head

        # Projections for Q, K, V
        self.q_projection = nn.Parameter(torch.Tensor(num_channels, self.num_head, self.key_dim_per_head))
        self.q_projection_bias = nn.Parameter(torch.Tensor(1, self.num_head, self.key_dim_per_head))
        self.k_projection = nn.Parameter(torch.Tensor(num_channels, self.num_head, self.key_dim_per_head))
        self.v_projection = nn.Parameter(torch.Tensor(num_channels, self.num_head, self.value_dim_per_head))

        # Gating mechanism
        self.gating_query = nn.Linear(num_channels, self.num_head * self.value_dim_per_head, bias=False)

        # Adaptive LayerNorm and Adaptive Zero Init
        self.adaptive_layernorm = AdaptiveLayerNorm(num_channels, single_featrues=self.single_channels, name=f'{name}')
        self.adaptive_zero_init = AdaptiveZeroInit(num_channels, self.num_head * self.value_dim_per_head, self.single_channels, global_config, name=f'{name}')
        
    def forward(
        self,
        x: torch.Tensor,  # ([bs], num_tokens, num_channels)
        mask: torch.Tensor,  # ([bs], num_tokens,)
        pair_logits: torch.Tensor | None = None,  # ([bs], num_heads, num_tokens, num_tokens)
        single_cond: torch.Tensor | None = None,  # ([bs], num_tokens, num_channels)
    ) -> torch.Tensor:
        # Ensure mask has the correct shape
        assert len(mask.shape) == len(x.shape) - 1, f'{mask.shape}, {x.shape}'

        # Create attention bias from mask
        bias = (1e9 * (mask.float() - 1.0))[..., None, None, :]  # ([bs], 1, 1, num_tokens)

        # Apply adaptive LayerNorm
        x = self.adaptive_layernorm(x, single_cond)

        # Project Q, K, V
        q = torch.einsum('...i,ijk->...jk', x, self.q_projection)
        q += self.q_projection_bias
        k = torch.einsum('...i,ijk->...jk', x, self.k_projection)
        v = torch.einsum('...i,ijk->...jk', x, self.v_projection)

        if pair_logits is not None:
            bias = bias + pair_logits  # ([bs], num_head, num_tokens, num_tokens)

        if hasattr(self.global_config,'attention_implementation') and self.global_config.attention_implementation == 'torch':
            weighted_avg = torch_attn(q, k, v, bias)
        elif hasattr(self.global_config,'attention_implementation') and self.global_config.attention_implementation == 'F':
            weighted_avg = F_attn(q, k, v, bias)
        else:
            weighted_avg = torch_attn(q, k, v, bias)

        weighted_avg = weighted_avg.reshape(-1, self.num_head * self.value_dim_per_head)  # ([bs], num_tokens, value_dim)

        # Apply gating mechanism
        gate_logits = self.gating_query(x)  # ([bs], num_tokens, num_head * value_dim_per_head)
        weighted_avg *= torch.sigmoid(gate_logits)

        # Apply adaptive zero init
        output = self.adaptive_zero_init(weighted_avg, single_cond)
        return output
  

class Transformer(nn.Module):
    """Simple transformer stack in PyTorch."""

    class Config(base_config.BaseConfig):
        attention: SelfAttentionConfig = base_config.autocreate()
        num_blocks: int = 24
        block_remat: bool = False
        super_block_size: int = 4
        num_intermediate_factor: int = 2

    def __init__(
        self,
        config: Config, # 使用 PyTorch 版本的 TransformerConfig
        global_config: model_config.GlobalConfig, # 使用 PyTorch 版本的 model_config.GlobalConfig
        act_channels: int,
        pair_channels: int,
        single_channels: int,
        name: str = 'transformer',
    ):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.num_super_block = config.num_blocks // config.super_block_size
        self.pair_input_layer_norm = tm.LayerNorm(
            normalized_shape=pair_channels, 
            eps=1e-5, 
            create_offset=False, 
            name='pair_input_layer_norm'
        ) # hm.LayerNorm 假设为 PyTorch LayerNorm
        
        # super block的循环
        self.pair_logits_projection = nn.ParameterList([
            nn.Parameter(torch.randn(pair_channels, config.super_block_size, config.attention.num_head), requires_grad=False)
            for _ in range(self.num_super_block)
        ])
        # super block的循环 + transformer block的循环
        self.super_transformer_blocks = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(config, global_config, 
                                num_channels=act_channels,
                                single_channels=single_channels,
                                name=f'transformer_block_{i}')
                for i in range(config.super_block_size) # 4
            ])
            for _ in range(self.num_super_block) # 6
        ])

    def forward(
        self,
        act: torch.Tensor,
        mask: torch.Tensor,
        single_cond: torch.Tensor,
        pair_cond: torch.Tensor | None,
    ) -> torch.Tensor:
        """Forward pass of Transformer in PyTorch."""
        # Precompute pair logits for performance
        if pair_cond is None:
            pair_act = None
        else:
            pair_act = self.pair_input_layer_norm(pair_cond) # 使用 self.pair_input_layer_norm
        
        transformer_input = act # 初始化 transformer input
        # 循环处理 super_blocks 4 次
        for pair_logits_projection, super_transformer_block in zip(self.pair_logits_projection,self.super_transformer_blocks):
            if pair_act is None:
                pair_logits = None
            else:
                pair_logits_proj = torch.einsum('...i, ijk -> ...jk', pair_act, pair_logits_projection)
                pair_logits = pair_logits_proj.permute(2, 3, 0, 1) # 使用 transpose 调整维度顺序            
            for i,transformer_block in enumerate(super_transformer_block):
                transformer_input = transformer_block(transformer_input, mask, pair_logits[i], single_cond)


        return transformer_input


class TransformerBlock(nn.Module):
    """Transformer block in PyTorch."""
    def __init__(self, config, global_config, num_channels, single_channels=None, name='transformer'):
        super().__init__()
        self.config = config
        self.global_config = global_config
        if single_channels is None: single_channels = num_channels
        self.self_attention = SelfAttention( # 使用 PyTorch 版本的 SelfAttention
            num_channels=num_channels,
            single_channels=single_channels,
            config=config.attention,
            global_config=global_config,
            name=f'{name}_self_attention'
        )
        self.transition_block = TransitionBlock( # 使用 PyTorch 版本的 TransitionBlock
            config.num_intermediate_factor,
            global_config,
            num_channels=num_channels,
            single_channels=single_channels,
            name=f'{name}_transition_block'
        )

    def forward(self, act, mask, pair_logits, single_cond):
        """Forward pass of TransformerBlock in PyTorch."""
        act += self.self_attention( # 使用 self.self_attention
            act,
            mask,
            pair_logits,
            single_cond,
        )
        act += self.transition_block( # 使用 self.transition_block
            act,
            single_cond,
        )
        return act


class CrossAttentionConfig(base_config.BaseConfig):
  num_head: int = 4
  key_dim: int = 128
  value_dim: int = 128

class CrossAttention(nn.Module):
    def __init__(self, config: CrossAttentionConfig, global_config, name: str = ''):
        super(CrossAttention, self).__init__()
        self.config = config
        self.global_config = global_config
        self.name = name

        # Initialize linear layers
        # self.q_projection = nn.Linear(config.key_dim, config.key_dim, bias=True)
        # self.k_projection = nn.Linear(config.key_dim, config.key_dim, bias=False)
        # self.v_projection = nn.Linear(config.value_dim, config.value_dim , bias=False)
        self.num_head = self.config.num_head
        key_dim = self.config.key_dim // self.config.num_head
        value_dim = self.config.value_dim // self.config.num_head

        self.q_projection = nn.Parameter(torch.randn(config.key_dim, self.num_head, key_dim))
        self.q_projection_bias = nn.Parameter(torch.randn(1, self.num_head, key_dim))
        self.k_projection = nn.Parameter(torch.randn(config.key_dim, self.num_head, key_dim))
        self.v_projection = nn.Parameter(torch.randn(config.value_dim, self.num_head, value_dim))
        self.gating_query = nn.Linear(config.value_dim, config.value_dim, bias=False)

        self.adaptive_zero_init = AdaptiveZeroInit(config.value_dim, config.value_dim, config.value_dim, global_config, name)
        self.adaptive_layernorm_q = AdaptiveLayerNorm(config.value_dim, f'{self.name}q')
        self.adaptive_layernorm_k = AdaptiveLayerNorm(config.key_dim, f'{self.name}k')
# ## diffuser/~/diffusion_head/diffusion_atom_transformer_encoder/__layer_stack_with_per_layer/diffusion_atom_transformer_encoderk_projection
# weights (3, 128, 4, 32)

    def forward(
        self,
        x_q: torch.Tensor,  # (..., Q, C)
        x_k: torch.Tensor,  # (..., K, C)
        mask_q: torch.Tensor,  # (..., Q)
        mask_k: torch.Tensor,  # (..., K)
        pair_logits: torch.Tensor | None = None,  # (..., Q, K)
        single_cond_q: torch.Tensor | None = None,  # (..., Q, C)
        single_cond_k: torch.Tensor | None = None,  # (..., K, C)
    ) -> torch.Tensor:
        """Multihead self-attention."""
        assert len(mask_q.shape) == len(x_q.shape) - 1, f'{mask_q.shape}, {x_q.shape}'
        assert len(mask_k.shape) == len(x_k.shape) - 1, f'{mask_k.shape}, {x_k.shape}'
        mask_k = mask_k.float()
        mask_q = mask_q.float()
        # bias: ... x heads (1) x query x key
        bias = (
            1e9
            * (mask_q - 1.0)[..., None, :, None]
            * (mask_k - 1.0)[..., None, None, :]
        )

        x_q = self.adaptive_layernorm_q(x_q, single_cond_q)
        x_k = self.adaptive_layernorm_k(x_k, single_cond_k)

        assert self.config.key_dim % self.config.num_head == 0
        assert self.config.value_dim % self.config.num_head == 0
        key_dim = self.config.key_dim // self.config.num_head
        value_dim = self.config.value_dim // self.config.num_head

        # Project queries, keys, and values
        # q = self.q_projection(x_q)  # (..., Q, num_head * key_dim)
        # k = self.k_projection(x_k)  # (..., K, num_head * key_dim)
        # v = self.v_projection(x_k)  # (..., K, num_head * value_dim)

        # # Reshape for multi-head attention
        # q = q.view(*q.shape[:-1], self.config.num_head, key_dim)  # (..., Q, num_head, key_dim)
        # k = k.view(*k.shape[:-1], self.config.num_head, key_dim)  # (..., K, num_head, key_dim)
        # v = v.view(*v.shape[:-1], self.config.num_head, value_dim)  # (..., K, num_head, value_dim)

        q = torch.einsum('...i,ijk->...jk', x_q, self.q_projection)
        q += self.q_projection_bias
        k = torch.einsum('...i,ijk->...jk', x_k, self.k_projection)
        v = torch.einsum('...i,ijk->...jk', x_k, self.v_projection)

        # Compute attention logits
        logits = torch.einsum('...qhc,...khc->...hqk', q * (key_dim ** (-0.5)), k) + bias
        ## we add
        # logits = logits.unsqueeze(0)
        if pair_logits is not None:
            logits = logits + pair_logits
            # logits += pair_logits

        # Compute attention weights
        weights = F.softmax(logits, dim=-1).to(x_q.dtype)  # (..., num_head, Q, K)

        # Compute weighted average of values
        weighted_avg = torch.einsum('...hqk,...khc->...qhc', weights, v)  # (..., Q, num_head, value_dim)
        weighted_avg = weighted_avg.reshape(*weighted_avg.shape[:-2], -1)  # (..., Q, num_head * value_dim)

        # Gating mechanism
        gate_logits = self.gating_query(x_q)  # (..., Q, num_head * value_dim)
        weighted_avg *= torch.sigmoid(gate_logits)

        # Adaptive zero initialization
        output = self.adaptive_zero_init(weighted_avg, single_cond_q)
        return output


class CrossAttTransformer(nn.Module):
    """Transformer with cross attention between two sets of subsets in PyTorch."""

    class Config(base_config.BaseConfig):
        num_intermediate_factor: int
        num_blocks: int
        attention: CrossAttentionConfig = base_config.autocreate()

    def __init__(
        self,
        config: Config, # 使用 PyTorch 版本的 CrossAttTransformerConfig
        global_config: model_config.GlobalConfig, # 使用 PyTorch 版本的 model_config.GlobalConfig
        pair_channels: int,
        single_channels: int, 
        name: str = 'transformer',
    ):
        super().__init__()
        self.config = config
        self.global_config = global_config

        self.pair_input_layer_norm = tm.LayerNorm(normalized_shape=pair_channels, 
                                                  use_fast_variance=False,
                                                  create_offset=False, 
                                                  name='pair_input_layer_norm') # hm.LayerNorm 假设为 PyTorch LayerNorm
        self.pair_logits_projection = nn.Parameter(torch.randn(pair_channels, config.num_blocks, config.attention.num_head))
        self.cross_att_blocks = nn.ModuleList() # 使用 nn.ModuleList 管理 cross attention blocks
        for i in range(config.num_blocks): # 根据 num_blocks 创建 cross attention blocks
            self.cross_att_blocks.append(
                CrossAttTransformerBlock(
                    config, 
                    global_config, 
                    pair_channels=pair_channels,
                    single_channels=single_channels, 
                    name=f'cross_att_block_{i}')) # 使用 CrossAttTransformerBlock 模块


    def forward(
        self,
        queries_act: torch.Tensor,  # (num_subsets, num_queries, ch)
        queries_mask: torch.Tensor,  # (num_subsets, num_queries)
        queries_to_keys: atom_layout.GatherInfo,  # (num_subsets, num_keys) # 假设 atom_layout.GatherInfo 已转换为 PyTorch
        keys_mask: torch.Tensor,  # (num_subsets, num_keys)
        queries_single_cond: torch.Tensor,  # (num_subsets, num_queries, ch)
        keys_single_cond: torch.Tensor,  # (num_subsets, num_keys, ch)
        pair_cond: torch.Tensor,  # (num_subsets, num_queries, num_keys, ch)
    ) -> torch.Tensor:
        """Forward pass of CrossAttTransformer in PyTorch."""
        
        # Precompute pair logits for performance
        pair_act = self.pair_input_layer_norm(pair_cond) # 使用 self.pair_input_layer_norm

        # (num_subsets, num_queries, num_keys, num_blocks, num_heads)
        # pair_logits_proj = self.pair_logits_projection(pair_act) # 获取对应 block 的 pair_logits_projection 层
        pair_logits_proj = torch.einsum('...i,ijk->...jk',pair_act, self.pair_logits_projection)
        # pair_logits_proj = pair_logits_proj.reshape(pair_act.shape[:-1] + (self.config.num_blocks, self.config.attention.num_head))
        # (num_block, num_subsets, num_heads, num_queries, num_keys)
        pair_logits = pair_logits_proj.permute(3, 0, 4, 1, 2) # 使用 transpose 调整维度顺序


        # Cross attention blocks
        cross_att_input = queries_act # 初始化 cross attention input
        for i in range(self.config.num_blocks): # 循环调用 cross attention blocks
            block = self.cross_att_blocks[i] # 获取对应的 cross attention block
            cross_att_input = block(cross_att_input, queries_mask, queries_to_keys, keys_mask, queries_single_cond, keys_single_cond, pair_logits[i]) # 调用 CrossAttTransformerBlock

        return cross_att_input


class CrossAttTransformerBlock(nn.Module):
    """CrossAttTransformer block in PyTorch."""
    def __init__(self, 
                 config, 
                 global_config, 
                 pair_channels: int,
                 single_channels: int, 
                 name):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.cross_attention = CrossAttention( # 使用 PyTorch 版本的 CrossAttention
            config.attention,
            global_config,
            name=f'{name}_cross_attention'
        )
        self.transition_block = TransitionBlock( # 使用 PyTorch 版本的 TransitionBlock
            config.num_intermediate_factor,
            global_config,
            num_channels=single_channels,
            name=f'{name}_transition_block'
        )

    def forward(
        self,
        queries_act: torch.Tensor,  # (num_subsets, num_queries, ch)
        queries_mask: torch.Tensor,  # (num_subsets, num_queries)
        queries_to_keys: atom_layout.GatherInfo,  # (num_subsets, num_keys) # 假设 atom_layout.GatherInfo 已转换为 PyTorch
        keys_mask: torch.Tensor,  # (num_subsets, num_keys)
        queries_single_cond: torch.Tensor,  # (num_subsets, num_queries, ch)
        keys_single_cond: torch.Tensor,  # (num_subsets, num_keys, ch)
        pair_logits: torch.Tensor,
    ):
        """Forward pass of CrossAttTransformerBlock in PyTorch."""
        # copy the queries activations to the keys layout
        keys_act = atom_layout.convert( # 假设 atom_layout.convert 已转换为 PyTorch
            queries_to_keys, queries_act, layout_axes=(-3, -2)
        )
        # cross attention
        queries_act += self.cross_attention( # 使用 self.cross_attention
            x_q=queries_act,
            x_k=keys_act,
            mask_q=queries_mask,
            mask_k=keys_mask,
            pair_logits=pair_logits,
            single_cond_q=queries_single_cond,
            single_cond_k=keys_single_cond,
        )
        queries_act += self.transition_block( # 使用 self.transition_block
            queries_act,
            queries_single_cond,
        )
        return queries_act

