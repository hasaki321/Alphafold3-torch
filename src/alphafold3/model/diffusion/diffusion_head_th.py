# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Diffusion Head."""

from collections.abc import Callable

import torch.utils.dlpack

from alphafold3.common import base_config
from alphafold3.model import feat_batch
from alphafold3.model import model_config
from alphafold3.model.components import haiku_modules as hm
from alphafold3.model.components import utils_th as utils
from alphafold3.model.diffusion import atom_cross_attention_th as atom_cross_attention
from alphafold3.model.diffusion import diffusion_transformer_th as diffusion_transformer
from alphafold3.model.diffusion import featurization_th as featurization
import chex
import haiku as hk
import jax
import jax.numpy as jnp

import torch
import torch.nn as nn
from alphafold3.model.components import torch_modules as tm
from tqdm import trange, tqdm
# Carefully measured by averaging multimer training set.
SIGMA_DATA = 16.0


def fourier_embeddings(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, dim: int) -> torch.Tensor:
    """Fourier embeddings in PyTorch."""
    # 注意w_key与b_key貌似是固定的，因为每次的key都是一样的 [TODO]
    # w_key, b_key = jax.random.split(jax.random.PRNGKey(42))
    # weight = torch.randn(dim) # 使用 torch.randn 初始化 weight
    # bias = torch.rand(dim)  # 使用 torch.rand 初始化 bias
    # weight = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jnp.load('../alphafold3_built/align/fourier_weight.npy')))
    # bias = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jnp.load('../alphafold3_built/align/fourier_bias.npy')))
    return torch.cos(2 * torch.pi * (x[..., None] * weight + bias))


def random_rotation() -> torch.Tensor:
    # Create a random rotation (Gram-Schmidt orthogonalization of two
    # random normal vectors)
    v0, v1 = torch.randn((2, 3)) # 使用 torch.randn
    e0 = v0 / torch.maximum(torch.tensor(1e-10), torch.linalg.norm(v0))
    v1 = v1 - e0 * torch.dot(v1, e0)
    e1 = v1 / torch.maximum(torch.tensor(1e-10), torch.linalg.norm(v1))
    e2 = torch.cross(e0, e1, dim=-1)
    # e2 = torch.linalg.cross(e0, e1)
    return torch.stack([e0, e1, e2])


def random_augmentation(
    positions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
  """Apply random rigid augmentation.

  Args:
    rng_key: random key
    positions: atom positions of shape (<common_axes>, 3)
    mask: per-atom mask of shape (<common_axes>,)

  Returns:
    Transformed positions with the same shape as input positions.
  """
  center = utils.mask_mean( # 假设 utils.mask_mean 也已转换为 PyTorch 版本
      mask[..., None], positions, axis=(-2, -3), keepdims=True, eps=1e-6
  )
  rot = random_rotation()
  translation = torch.randn(3) # 使用 torch.randn for translation

  augmented_positions = (
      torch.einsum(
          '...i,ij->...j',
          positions - center,
          rot,
      )
      + translation
  )
  return augmented_positions * mask[..., None]


def noise_schedule(t, smin=0.0004, smax=160.0, p=7):
  return (
      SIGMA_DATA
      * (smax ** (1 / p) + t * (smin ** (1 / p) - smax ** (1 / p))) ** p
  )


class ConditioningConfig(base_config.BaseConfig):
  pair_channel: int
  seq_channel: int
  prob: float


class SampleConfig(base_config.BaseConfig):
  steps: int
  gamma_0: float = 0.8
  gamma_min: float = 1.0
  noise_scale: float = 1.003
  step_scale: float = 1.5
  num_samples: int = 1


class DiffusionHead(torch.nn.Module):
    """Denoising Diffusion Head."""

    class Config(
        atom_cross_attention.AtomCrossAttEncoderConfig,
        atom_cross_attention.AtomCrossAttDecoderConfig,
    ):
        """Configuration for DiffusionHead."""

        eval_batch_size: int = 5
        eval_batch_dim_shard_size: int = 5
        conditioning: ConditioningConfig = base_config.autocreate(
            prob=0.8, pair_channel=128, seq_channel=384
        )
        eval: SampleConfig = base_config.autocreate(
            num_samples=5,
            steps=200,
        )
        transformer: diffusion_transformer.Transformer.Config = (
            base_config.autocreate()
        )

    def __init__(
        self,
        config: Config,
        global_config: model_config.GlobalConfig,
        pair_channel: int,
        seq_channel: int,
        name='diffusion_head',
    ):
        self.config = config
        self.global_config = global_config
        self.pair_channel = pair_channel
        self.seq_channel = seq_channel
        super().__init__()
        self.create_pararms()

    def create_pararms(self):
        self.pair_cond_initial_norm = tm.LayerNorm( # tm.LayerNorm 假设是转换后的 PyTorch LayerNorm
            normalized_shape = self.pair_channel + 139,
            use_fast_variance=False,
            create_offset=False,
            name='pair_cond_initial_norm',
        )
        self.pair_cond_initial_projection = nn.Linear( # tm.LayerNorm 假设是转换后的 PyTorch LayerNorm
            self.pair_channel + 139,
            self.config.conditioning.pair_channel,bias=False
        )
        self.pair_transition_blocks = nn.ModuleList([ # 使用 nn.ModuleList 管理 transition blocks
            diffusion_transformer.TransitionBlock( # TODO
                num_intermediate_factor=2,
                global_config=self.global_config,
                num_channels = self.config.conditioning.pair_channel,
                name=f'pair_transition_{idx}'
            ) for idx in range(2)
        ])

        self.single_cond_initial_norm = tm.LayerNorm( # tm.LayerNorm 假设是转换后的 PyTorch LayerNorm
            normalized_shape=self.seq_channel + 447,
            eps=1e-5,
            create_offset=False,
            name='single_cond_initial_norm'
        )
        self.single_cond_initial_projection = nn.Linear( # tm.Linear 假设是转换后的 PyTorch Linear
            self.seq_channel + 447,
            self.config.conditioning.seq_channel,bias=False
        )


        self.noise_embedding_initial_norm = tm.LayerNorm( # tm.LayerNorm 假设是转换后的 PyTorch LayerNorm
            normalized_shape=256,
            eps=1e-5,
            create_offset=False,
            name='noise_embedding_initial_norm'
        )
        self.noise_embedding_initial_projection = nn.Linear( # tm.Linear 假设是转换后的 PyTorch Linear
            256,
            self.config.conditioning.seq_channel,bias=False
        )


        self.single_transition_blocks = nn.ModuleList([ # 使用 nn.ModuleList 管理 single transition blocks
            diffusion_transformer.TransitionBlock( # 假设 transition_block 也已转换为 PyTorch
                num_intermediate_factor=2,
                global_config=self.global_config,
                num_channels = self.config.conditioning.seq_channel,
                name=f'single_transition_{idx}'
            ) for idx in range(2)
        ])


        self.diffusion_atom_cross_att_encoder = atom_cross_attention.AtomCrossAttEncoder( # 假设 AtomCrossAttEncoder 也已转换
            config=self.config,
            global_config=self.global_config,
            pair_channel=self.config.conditioning.pair_channel,
            single_channel=self.config.conditioning.seq_channel,
            name='diffusion'
        )
        self.single_cond_embedding_norm = tm.LayerNorm( # tm.LayerNorm 假设是转换后的 PyTorch LayerNorm
            normalized_shape=self.config.conditioning.seq_channel,
            eps=1e-5,
            create_offset=False,
            name='single_cond_embedding_norm'
        )
        self.single_cond_embedding_projection = nn.Linear( # tm.Linear 假设是转换后的 PyTorch Linear
            self.config.conditioning.seq_channel, 
            self.config.per_token_channels,bias=False
        )
        self.transformer = diffusion_transformer.Transformer( # 假设 Transformer 也已转换
            act_channels=self.config.per_token_channels,
            pair_channels=self.config.conditioning.pair_channel,
            single_channels=self.config.conditioning.seq_channel,
            config=self.config.transformer,
            global_config=self.global_config
        )
        self.output_norm = tm.LayerNorm( # tm.LayerNorm 假设是转换后的 PyTorch LayerNorm
            normalized_shape=self.config.per_token_channels,
            eps=1e-5,
            create_offset=False,
            name='output_norm'
        )
        #    "per_atom_channels": 128,
        #   "per_atom_pair_channels": 16,
        #   "per_token_channels": 768,
        self.diffusion_atom_cross_att_decoder = atom_cross_attention.AtomCrossAttDecoder( # 假设 AtomCrossAttDecoder 也已转换
            config=self.config,
            global_config=self.global_config,
            # num_channels=self.config.per_token_channels,
            pair_channel=self.config.per_atom_pair_channels,
            single_channel=self.config.per_atom_channels,
            name='diffusion'
        )
        self.fourier_weight = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jnp.load('./models/fourier_weight.npy')))
        self.fourier_bias = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jnp.load('./models/fourier_bias.npy')))
    
    # @hk.transparent
    def _conditioning(
        self,
        batch: feat_batch.Batch,
        embeddings: dict[str, jnp.ndarray],
        noise_level: jnp.ndarray,
        use_conditioning: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        single_embedding = use_conditioning * embeddings['single']
        pair_embedding = use_conditioning * embeddings['pair']

        rel_features = featurization.create_relative_encoding( # TODO
            batch.token_features, max_relative_idx=32, max_relative_chain=2
        ).to(pair_embedding.dtype)

        features_2d = torch.cat([pair_embedding, rel_features], dim=-1)

        pair_cond = self.pair_cond_initial_projection(
            self.pair_cond_initial_norm(features_2d)
        )


        for block in self.pair_transition_blocks: # 循环调用 pair transition blocks
            pair_cond += block(pair_cond)

        target_feat = embeddings['target_feat']
        features_1d = torch.cat([single_embedding, target_feat], dim=-1) # 使用 torch.cat
        single_cond = self.single_cond_initial_projection(
            self.single_cond_initial_norm(features_1d)
        )


        noise_embedding = fourier_embeddings(
            (1 / 4) * torch.log(noise_level / SIGMA_DATA), self.fourier_weight, self.fourier_bias, dim=256 # 使用 torch.log
        )
        # noise_embedding = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jnp.load('../alphafold3_built/align/noise_embedding.npy')))
        
        # print('noise_level',noise_level.shape)
        # print('single_cond',single_cond.shape)
        # print('noise_embedding',noise_embedding.shape)
        noise_embedding = self.noise_embedding_initial_projection(
            self.noise_embedding_initial_norm(noise_embedding)
            )
        # print('noise_embedding',noise_embedding.shape)

        single_cond = single_cond + noise_embedding



        for block in self.single_transition_blocks: # 循环调用 single transition blocks
            single_cond += block(single_cond)

        return single_cond, pair_cond

    def forward(
        self,
        # positions_noisy.shape: (num_token, max_atoms_per_token, 3)
        positions_noisy: torch.Tensor,
        noise_level: torch.Tensor,
        batch: feat_batch.Batch, 
        embeddings: dict[str, torch.Tensor],
        use_conditioning: bool,
    ) -> jnp.ndarray:
        # TODO No bfloat16 context in PyTorch, handle dtype conversions manually if needed

        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            assert len(noise_level.shape) == 1, noise_level.shape
            # Get conditioning
            trunk_single_cond, trunk_pair_cond = self._conditioning(
                batch=batch,
                embeddings=embeddings,
                noise_level=noise_level,
                use_conditioning=use_conditioning,
            )

            # Extract features
            sequence_mask = batch.token_features.mask
            atom_mask = batch.predicted_structure_info.atom_mask

            # Position features
            act = positions_noisy * atom_mask[..., None]
            act = act / torch.sqrt(noise_level**2 + SIGMA_DATA**2)

            enc = self.diffusion_atom_cross_att_encoder( # TODO
                token_atoms_act=act,
                trunk_single_cond=embeddings['single'],
                trunk_pair_cond=trunk_pair_cond,
                batch=batch,
            )
            act = enc.token_act

            # Token-token attention
            chex.assert_shape(act, (None, self.config.per_token_channels))

            act += self.single_cond_embedding_projection(
                self.single_cond_embedding_norm(trunk_single_cond)
            )

            act = self.transformer(
                act=act,
                single_cond=trunk_single_cond,
                mask=sequence_mask,
                pair_cond=trunk_pair_cond,
            )
            act = self.output_norm(act)
            # (n_tokens, per_token_channels)

            # (Possibly) atom-granularity decoder
            # TODO
            assert isinstance(enc, atom_cross_attention.AtomCrossAttEncoderOutput)
            position_update = self.diffusion_atom_cross_att_decoder(
                token_act=act,
                enc=enc,
                batch=batch,
            )

            skip_scaling = SIGMA_DATA**2 / (noise_level**2 + SIGMA_DATA**2)
            out_scaling = (
                noise_level * SIGMA_DATA / torch.sqrt(noise_level**2 + SIGMA_DATA**2)
            )
        # End `with utils.bfloat16_context()`.

        return (
            skip_scaling * positions_noisy + out_scaling * position_update
        ) * atom_mask[..., None]



def sample(
    denoising_step: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    batch: feat_batch.Batch, 
    # key: torch.Tensor,
    config: SampleConfig,
) -> dict[str, jnp.ndarray]:
    """Sample using denoiser on batch.

    Args:
        denoising_step: the denoising function.
        batch: the batch
        key: random key
        config: config for the sampling process (e.g. number of denoising steps,
        etc.)

    Returns:
        a dict
        {
            'atom_positions': jnp.array(...)       # shape (<common_axes>, 3)
            'mask': jnp.array(...)                 # shape (<common_axes>,)
        }
        where the <common_axes> are
        (num_samples, num_tokens, max_atoms_per_token)
    """
    mask = batch.predicted_structure_info.atom_mask


    def apply_denoising_step(positions, noise_level_prev, noise_level):
        load_jax_tensor = True
        # positions_outs = []
        # noise_levels = []
        # for i in range(config.num_samples): # 5
        # 创建纯净噪声
        # print(positions.shape, noise_level_prev.shape, noise_level.shape)
        positions = random_augmentation(
            positions=positions, mask=mask
        )
        # print(positions.shape)
        # 噪声缩放
        # if load_jax_tensor:
        #     positions = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jnp.load('../alphafold3_built/align/positions.npy')))
        gamma = config.gamma_0 * (noise_level > config.gamma_min)
        t_hat = noise_level_prev * (1 + gamma)

        noise_scale = config.noise_scale * torch.sqrt(t_hat**2 - noise_level_prev**2)
        noise = noise_scale * torch.randn_like(positions)
        # if load_jax_tensor:
        #     noise = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jnp.load('../alphafold3_built/align/noise.npy')))
        positions_noisy = positions + noise
        # print(positions_noisy.shape)

        # 模型去噪
        # print(positions_noisy.shape, t_hat.shape)
        positions_denoised = denoising_step(positions_noisy, t_hat)

        # Classfier-Free guidance 
        grad = (positions_noisy - positions_denoised) / t_hat

        d_t = noise_level - t_hat
        positions_out = positions_noisy + config.step_scale * d_t * grad

        return positions_out, noise_level


    num_samples = config.num_samples

    noise_levels = noise_schedule(torch.linspace(0, 1, config.steps + 1)).unsqueeze(-1)

    #   key, noise_key = jax.random.split(key)
    #   positions = jax.random.normal(noise_key, (num_samples,) + mask.shape + (3,))
    #   positions *= noise_levels[0]

    positions = noise_levels[0] * torch.randn((num_samples,) + tuple(mask.shape) + (3,))
    # print("positions.shape",positions.shape) # torch.Size([5, 37, 24, 3])
    # print('num_samples',num_samples) # 5
    # print('noise_levels',noise_levels.shape) # torch.Size([201])

    # init = (
    #     positions,
    #     noise_levels[None, 0].expand(num_samples, 1), # Use repeat instead of tile TODO
    # )

    noise_levels_prev = noise_levels[None, 0].expand(num_samples, 1)

    apply_denoising_step = torch.vmap(
      apply_denoising_step, in_dims=(0, 0, None), out_dims=0, chunk_size=None, randomness='different'
    )

    for i in trange(1, config.steps + 1,desc='diffusion apply_denoising_step'): # Manual loop for scan
    # for i in trange(1, 1 + 1,desc='diffusion apply_denoising_step'): # Manual loop for scan
        positions, noise_levels_prev = apply_denoising_step(positions, noise_levels_prev, noise_levels[i]) # Call denoising step in loop
        # print(f'loop {i}:',positions.shape, noise_levels_prev.shape)

    # final_dense_atom_mask = jnp.tile(mask[None], (num_samples, 1, 1))
    final_dense_atom_mask = mask[None].repeat((num_samples, 1, 1)) # Use repeat instead of tile TODO

    return {'atom_positions': positions, 'mask': final_dense_atom_mask}

