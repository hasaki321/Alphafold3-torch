# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Per-atom cross attention."""

from alphafold3.common import base_config
from alphafold3.model import feat_batch
from alphafold3.model import model_config
from alphafold3.model.atom_layout import atom_layout_th as atom_layout
# from alphafold3.model.atom_layout import atom_layout as atom_layout
from alphafold3.model.components import utils_th as utils
from alphafold3.model.diffusion import diffusion_transformer_th_batch as diffusion_transformer
from alphafold3.model.components import torch_modules as tm

import chex
import torch
import jax
import jax.numpy as jnp
import torch.nn as nn
import torch.nn.functional as F

class AtomCrossAttEncoderConfig(base_config.BaseConfig):
  per_token_channels: int = 768
  per_atom_channels: int = 128
  atom_transformer: diffusion_transformer.CrossAttTransformer.Config = (
      base_config.autocreate(num_intermediate_factor=2, num_blocks=3)
  )
  per_atom_pair_channels: int = 16

class Batch:
    def __init__(self, ref_structure):
        self.ref_structure = ref_structure

class RefStructure:
    def __init__(self, positions, mask, element, charge, atom_name_chars):
        self.positions = positions
        self.mask = mask
        self.element = element
        self.charge = charge
        self.atom_name_chars = atom_name_chars


class PerAtomConditioning(nn.Module):
    def __init__(self, config: AtomCrossAttEncoderConfig):
        super(PerAtomConditioning, self).__init__()
        self.config = config

        # Initialize modules for single conditioning
        self.embed_ref_pos = nn.Linear(3, config.per_atom_channels, bias=False)
        self.embed_ref_mask = nn.Linear(1, config.per_atom_channels, bias=False)
        self.embed_ref_element = nn.Linear(128, config.per_atom_channels, bias=False)
        self.embed_ref_charge = nn.Linear(1, config.per_atom_channels, bias=False)
        self.embed_ref_atom_name = nn.Linear(64 * 4, config.per_atom_channels, bias=False)  # Assuming 4 characters

        # Initialize modules for pair conditioning
        self.single_to_pair_cond_row = nn.Linear(config.per_atom_channels, config.per_atom_pair_channels, bias=False)
        self.single_to_pair_cond_col = nn.Linear(config.per_atom_channels, config.per_atom_pair_channels, bias=False)
        self.embed_pair_offsets = nn.Linear(3, config.per_atom_pair_channels, bias=False)
        self.embed_pair_distances = nn.Linear(1, config.per_atom_pair_channels, bias=False)

    def forward(self, batch: feat_batch.Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes single and pair conditioning for all atoms in each token."""
        c = self.config

        # Compute per-atom single conditioning
        act = self.embed_ref_pos(batch.ref_structure.positions)
        act += self.embed_ref_mask(batch.ref_structure.mask.float()[:, :, None])
        act += self.embed_ref_element(F.one_hot(batch.ref_structure.element.long(), 128).float())
        act += self.embed_ref_charge(torch.arcsinh(batch.ref_structure.charge)[:, :, None])
        atom_name_chars_1hot = F.one_hot(batch.ref_structure.atom_name_chars.long(), 64)
        num_token, num_dense, _ = act.shape
        act += self.embed_ref_atom_name(atom_name_chars_1hot.reshape(num_token, num_dense, -1).float())
        act *= batch.ref_structure.mask.float()[:, :, None]

        # Compute pair conditioning
        row_act = self.single_to_pair_cond_row(F.relu(act))
        col_act = self.single_to_pair_cond_col(F.relu(act))
        pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]
        pair_act += self.embed_pair_offsets(
            batch.ref_structure.positions[:, :, None, :]
            - batch.ref_structure.positions[:, None, :, :]
        )
        sq_dists = torch.sum(
            torch.square(
                batch.ref_structure.positions[:, :, None, :]
                - batch.ref_structure.positions[:, None, :, :]
            ),
            dim=-1,
        )
        pair_act += self.embed_pair_distances(1.0 / (1 + sq_dists[:, :, :, None]))

        return act, pair_act

@chex.dataclass(mappable_dataclass=False, frozen=True)
class AtomCrossAttEncoderOutput:
  token_act: torch.tensor  # (num_tokens, ch)
  skip_connection: torch.tensor  # (num_subsets, num_queries, ch)
  queries_mask: torch.tensor  # (num_subsets, num_queries)
  queries_single_cond: torch.tensor  # (num_subsets, num_queries, ch)
  keys_mask: torch.tensor  # (num_subsets, num_keys)
  keys_single_cond: torch.tensor  # (num_subsets, num_keys, ch)
  pair_cond: torch.tensor  # (num_subsets, num_queries, num_keys, ch)


class AtomCrossAttEncoder(nn.Module):
    def __init__(
        self,
        config: AtomCrossAttEncoderConfig,
        global_config: model_config.GlobalConfig,
        pair_channel: int, 
        single_channel:int,
        name: str,
    ):
        super(AtomCrossAttEncoder, self).__init__()
        self.config = config
        self.global_config = global_config
        self.name = name

        self._per_atom_conditioning = PerAtomConditioning(config)

        # Initialize MLP for pair activations
        self.single_to_pair_cond_row = nn.Linear(config.per_atom_channels, config.per_atom_pair_channels, bias=False)
        self.single_to_pair_cond_col = nn.Linear(config.per_atom_channels, config.per_atom_pair_channels, bias=False)
        
        self.embed_pair_offsets = nn.Linear(3, config.per_atom_pair_channels, bias=False)
        self.embed_pair_distances = nn.Linear(1, config.per_atom_pair_channels, bias=False)
        self.embed_pair_offsets_valid = nn.Linear(1, config.per_atom_pair_channels, bias=False)
        
        self.pair_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(config.per_atom_pair_channels, config.per_atom_pair_channels, bias=False),
            nn.ReLU(),
            nn.Linear(config.per_atom_pair_channels, config.per_atom_pair_channels, bias=False),
            nn.ReLU(),
            nn.Linear(config.per_atom_pair_channels, config.per_atom_pair_channels, bias=False),
        )

        # Initialize transformer
        self.atom_transformer_encoder = diffusion_transformer.CrossAttTransformer(
            config.atom_transformer, 
            global_config, 
            pair_channels=config.per_atom_pair_channels, 
            single_channels=config.per_atom_channels,
            name=f'{name}_atom_transformer_encoder'
        )

        if name != 'evoformer_conditioning':
            self.atom_positions_to_features = nn.Linear(3, config.per_atom_channels, bias=False)
            
            # single_cond
            self.embed_trunk_single_cond = nn.Linear(single_channel, config.per_atom_channels, bias=False)
            self.lnorm_trunk_single_cond = tm.LayerNorm(single_channel,create_offset=False) 

            # trunk_pair_cond
            self.lnorm_trunk_pair_cond = tm.LayerNorm(pair_channel,create_offset=False)
            self.embed_trunk_pair_cond = nn.Linear(pair_channel, config.per_atom_pair_channels, bias=False)

        # Initialize final projection
        self.project_atom_features_for_aggr = nn.Linear(config.per_atom_channels, config.per_token_channels, bias=False)
    def forward(
        self,
        token_atoms_act: torch.Tensor | None,  # (batch, num_tokens, max_atoms_per_token, 3)
        trunk_single_cond: torch.Tensor | None,  # (num_tokens, ch)
        trunk_pair_cond: torch.Tensor | None,  # (num_tokens, num_tokens, ch)
        batch: feat_batch.Batch,
    ) -> AtomCrossAttEncoderOutput:
        """Cross-attention on flat atom subsets and mapping to per-token features."""

        # Compute single conditioning from atom meta data and convert to queries layout.

        # (num_tokens, max_atoms_per_token, channels)
        token_atoms_single_cond, _ = self._per_atom_conditioning(batch)
        token_atoms_mask = batch.predicted_structure_info.atom_mask # (num_tokens, max_atoms_per_token）
        queries_single_cond = atom_layout.convert(
            batch.atom_cross_att.token_atoms_to_queries,
            token_atoms_single_cond,
            layout_axes=(-3, -2),
        )# (num_res, num_query, channels)

        queries_mask = atom_layout.convert(
            batch.atom_cross_att.token_atoms_to_queries,
            token_atoms_mask,
            layout_axes=(-2, -1),
        )# (num_res, num_query)


        trunk_single_cond = self.embed_trunk_single_cond(
            self.lnorm_trunk_single_cond(trunk_single_cond)
            ) # (num_tokens, channels)
        queries_single_cond += atom_layout.convert(
            batch.atom_cross_att.tokens_to_queries,
            trunk_single_cond,
            layout_axes=(-2,),
        ) # (num_res, num_query, channels) + (num_res, num_query, channels)

        queries_act = atom_layout.convert(
            batch.atom_cross_att.token_atoms_to_queries,
            token_atoms_act,  # (batch, num_tokens, max_atoms_per_token, 3) # [TODO]
            layout_axes=(-3, -2),
        ) # (batch, num_res, num_query, 3)
        queries_act = self.atom_positions_to_features(queries_act) # (batch, num_res, num_query, channels)
        queries_act *= queries_mask[None, ..., None] # (batch, num_res, num_query, channels) *  (1, num_res, num_query, 1)
        queries_act += queries_single_cond[None, ...]# (batch, num_res, num_query, channels) + (1, num_res, num_query, channels)
        
        # Gather the keys from the queries.
        keys_single_cond = atom_layout.convert(
            batch.atom_cross_att.queries_to_keys,
            queries_single_cond,
            layout_axes=(-3, -2),
        ) # (num_res, num_key, channels) [TODO]
        keys_mask = atom_layout.convert(
            batch.atom_cross_att.queries_to_keys, queries_mask, layout_axes=(-2, -1)
        ) # (num_res, num_key, ) [TODO]

        # Embed single features into the pair conditioning.
        row_act = self.single_to_pair_cond_row(F.relu(queries_single_cond)) # (batch, num_res, num_query, channels)

        pair_cond_keys_input = atom_layout.convert(
                    batch.atom_cross_att.queries_to_keys,
                    queries_single_cond,
                    layout_axes=(-3, -2), # [TODO]
                ) # (batch, num_res, num_key, channels)
        
        col_act = self.single_to_pair_cond_col(F.relu(pair_cond_keys_input)) # (num_res, num_key, channels)
        pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]  # (num_res, num_query, num_key, channels)
        # print(pair_act.shape)

        # If provided, broadcast the pair conditioning for the trunk (evoformer
        # pairs) to the atom pair activations. This should boost ligands, but also
        # help for cross attention within proteins, because we always have atoms
        # from multiple residues in a subset.
        # Map trunk pair conditioning to per_atom_pair_channels
        # (num_tokens, num_tokens, per_atom_pair_channels)
        trunk_pair_cond = self.embed_trunk_pair_cond(self.lnorm_trunk_pair_cond(trunk_pair_cond)) # (num_tokens, num_tokens, channels)
        num_tokens = trunk_pair_cond.shape[0]
        tokens_to_queries = batch.atom_cross_att.tokens_to_queries
        tokens_to_keys = batch.atom_cross_att.tokens_to_keys
        trunk_pair_to_atom_pair = atom_layout.GatherInfo(
            gather_idxs=(
                num_tokens * tokens_to_queries.gather_idxs[:, :, None]
                + tokens_to_keys.gather_idxs[:, None, :]
            ),
            gather_mask=(
                tokens_to_queries.gather_mask[:, :, None]
                & tokens_to_keys.gather_mask[:, None, :]
            ),
            input_shape=torch.tensor((num_tokens, num_tokens)),
        )
        pair_act += atom_layout.convert(
            trunk_pair_to_atom_pair, trunk_pair_cond, layout_axes=(-3, -2)
        ) # (num_res, num_query, num_key, channels) +  (num_res, num_query, num_key, channels)

        # Embed pairwise offsets
        queries_ref_pos = atom_layout.convert(
            batch.atom_cross_att.token_atoms_to_queries,
            batch.ref_structure.positions,
            layout_axes=(-3, -2),
        )
        queries_ref_space_uid = atom_layout.convert(
            batch.atom_cross_att.token_atoms_to_queries,
            batch.ref_structure.ref_space_uid,
            layout_axes=(-2, -1),
        )
        keys_ref_pos = atom_layout.convert(
            batch.atom_cross_att.queries_to_keys,
            queries_ref_pos,
            layout_axes=(-3, -2),
        )
        keys_ref_space_uid = atom_layout.convert(
            batch.atom_cross_att.queries_to_keys,
            batch.ref_structure.ref_space_uid,
            layout_axes=(-2, -1),
        )

        offsets_valid = (
            queries_ref_space_uid[:, :, None] == keys_ref_space_uid[:, None, :]
        )
        offsets = queries_ref_pos[:, :, None, :] - keys_ref_pos[:, None, :, :]
        pair_act += (
            self.embed_pair_offsets(offsets) # (num_res, num_query, num_key, channels)
            * offsets_valid[:, :, :, None]  # (num_res, num_query, num_key, 1)
        )# (num_res, num_query, num_key, channels) +  (num_res, num_query, num_key, channels)

        # Embed pairwise inverse squared distances
        sq_dists = torch.sum(torch.square(offsets), dim=-1) # (num_res, num_query, num_key, )
        pair_act += (
            self.embed_pair_distances(1.0 / (1 + sq_dists[:, :, :, None])) # (num_res, num_query, num_key, channels)
            * offsets_valid[:, :, :, None] # (num_res, num_query, num_key, 1)
        )  # (num_res, num_query, num_key, channels) +  (num_res, num_query, num_key, channels)

        # Embed offsets valid mask
        pair_act += self.embed_pair_offsets_valid(offsets_valid[:, :, :, None].float())
        # (num_res, num_query, num_key, channels) +  (num_res, num_query, num_key, channels)

        pair_act = self.pair_mlp(pair_act) # (num_res, num_query, num_key, channels)
        # print(pair_act.shape)
        # Run the atom cross attention transformer.
        queries_act = self.atom_transformer_encoder(
            queries_act=queries_act, # (batch, num_res, num_query, channels)
            queries_mask=queries_mask, # (num_res, num_query)
            queries_to_keys=batch.atom_cross_att.queries_to_keys, # (num_res, num_key)
            keys_mask=keys_mask, # (num_res, num_key, )
            queries_single_cond=queries_single_cond, # (num_res, num_query, channels)
            keys_single_cond=keys_single_cond, # (num_res, num_key, channels)
            pair_cond=pair_act, # (num_res, num_query, num_key, channels)
        ) # (batch, num_res, num_query, channels)
        queries_act *= queries_mask[None, ..., None] # (batch, num_res, num_query, channels) * # (1, num_res, num_query, 1)
        skip_connection = queries_act

        # Convert back to token-atom layout and aggregate to tokens
        queries_act = self.project_atom_features_for_aggr(queries_act)
        token_atoms_act = atom_layout.convert(
            batch.atom_cross_att.queries_to_token_atoms,
            queries_act,
            layout_axes=(-3, -2),
        )# (batch, num_tokens, max_atoms_per_token， channels)
        token_act = utils.mask_mean(
            token_atoms_mask[None, ..., None], F.relu(token_atoms_act), axis=-2
        ) # (1, num_tokens, max_atoms_per_token， 1）, (batch, num_tokens, max_atoms_per_token， channels)

        return AtomCrossAttEncoderOutput(
            token_act=token_act, # (batch, num_tokens, channels)
            skip_connection=skip_connection, # (batch, num_res, num_query, channels)
            queries_mask=queries_mask, # (num_res, num_query)
            queries_single_cond=queries_single_cond, # (num_res, num_query, channels)
            keys_mask=keys_mask, # (num_res, num_key, )
            keys_single_cond=keys_single_cond, # (num_res, num_key, channels)
            pair_cond=pair_act, # (num_res, num_query, num_key, channels)
        )


class AtomCrossAttDecoderConfig(base_config.BaseConfig):
  per_atom_channels: int = 128
  atom_transformer: diffusion_transformer.CrossAttTransformer.Config = (
      base_config.autocreate(num_intermediate_factor=2, num_blocks=3)
  )

class AtomCrossAttDecoder(nn.Module):
    def __init__(
        self,
        config: AtomCrossAttDecoderConfig,
        global_config: model_config.GlobalConfig,
        pair_channel: int,
        single_channel: int,
        name: str,
    ):
        super(AtomCrossAttDecoder, self).__init__()
        self.config = config
        self.global_config = global_config
        self.name = name

        # Project token features to per-atom channels
        self.project_token_features_for_broadcast = nn.Linear(
            768, config.per_atom_channels, bias=False
        )

        # Atom cross attention transformer
        self.atom_transformer_decoder = diffusion_transformer.CrossAttTransformer(
            config.atom_transformer, 
            global_config, 
            pair_channels=pair_channel, 
            single_channels=single_channel,
            name=f'{name}_atom_transformer_decoder'
        )

        # Layer normalization for atom features
        self.atom_features_layer_norm = tm.LayerNorm(config.per_atom_channels,create_offset=False)

        # Linear layer to map atom features to position updates
        self.atom_features_to_position_update = nn.Linear(
            config.per_atom_channels, 3, bias=False
        )

    def forward(
        self,
        token_act: torch.Tensor,  # (batch, num_tokens, channels)
        enc: AtomCrossAttEncoderOutput,
        batch: feat_batch.Batch,
    ) -> torch.Tensor:
        """Mapping to per-atom features and self-attention on subsets."""
        c = self.config

        # Map per-token act down to per_atom channels
        token_act = self.project_token_features_for_broadcast(token_act)

        # Broadcast to token-atoms layout and convert to queries layout
        num_token, max_atoms_per_token = batch.atom_cross_att.queries_to_token_atoms.shape
        token_atom_act = token_act[:, :, None, :].expand(
            token_act.shape[0], num_token, max_atoms_per_token, c.per_atom_channels
        )# (batch, num_tokens, num_atom, channels)
        queries_act = atom_layout.convert(
            batch.atom_cross_att.token_atoms_to_queries,
            token_atom_act,
            layout_axes=(-3, -2),
        )# (batch, num_res, num_query, channels)

        # Add skip connection from encoder
        queries_act += enc.skip_connection # (batch, num_res, num_query, channels) + (batch, num_res, num_query, channels)
        queries_act *= enc.queries_mask[None, ..., None] # (batch, num_res, num_query, channels) * (1, num_res, num_query, 1 )

        # Run the atom cross attention transformer
        
        queries_act = self.atom_transformer_decoder(
            queries_act=queries_act, # (batch, num_res, num_query, channels)
            queries_mask=enc.queries_mask, # (num_res, num_query)
            queries_to_keys=batch.atom_cross_att.queries_to_keys, # (num_res, num_key)
            keys_mask=enc.keys_mask, # (num_res, num_key, )
            queries_single_cond=enc.queries_single_cond, # (num_res, num_query, channels)
            keys_single_cond=enc.keys_single_cond, # (num_res, num_key, channels)
            pair_cond=enc.pair_cond, # (num_res, num_query, num_key, channels)
        )
        queries_act *= enc.queries_mask[None, ..., None] # (batch, num_res, num_query, channels) * (1, num_res, num_query, 1 )

        # Apply layer normalization
        queries_act = self.atom_features_layer_norm(queries_act)

        # Map atom features to position updates
        queries_position_update = self.atom_features_to_position_update(queries_act)

        # Convert back to token-atoms layout
        position_update = atom_layout.convert(
            batch.atom_cross_att.queries_to_token_atoms,
            queries_position_update,
            layout_axes=(-3, -2),
        )# (batch, num_token, num_atom, 3)

        return position_update