import torch
import torch_scatter
import jax
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import functools
import numpy as np

import functools

from alphafold3.constants import residue_names
from alphafold3.model import feat_batch
from alphafold3.model import features
from alphafold3.model.components import utils
import chex

def _grid_keys(key: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """Generate a grid of rng keys that is consistent with different padding.

    Args:
        key: A PRNG key.
        shape: The shape of the output array of keys that will be generated.

    Returns:
        An array of shape `shape` consisting of random keys.
    """
    if not shape:
        return key
    new_keys = torch.stack([torch.fmod(key + i, 2**32) for i in range(shape[0])])
    return torch.stack([_grid_keys(k, shape[1:]) for k in new_keys])


def _padding_consistent_rng(f):
    """Modify any element-wise random function to be consistent with padding.

    Args:
        f: Any element-wise function that takes (PRNG key, shape) as the first 2 arguments.

    Returns:
        An equivalent function to f, that is now consistent for different amounts of padding.
    """
    def inner(key: torch.Tensor, shape: Tuple[int, ...], **kwargs):
        keys = _grid_keys(key, shape)
        return torch.vmap(functools.partial(f, shape=(), **kwargs))(keys)
    return inner


def gumbel_argsort_sample_idx(logits: torch.Tensor) -> torch.Tensor:
    """Samples with replacement from a distribution given by 'logits'.

    Args:
        key: prng key
        logits: logarithm of probabilities to sample from, probabilities can be unnormalized.

    Returns:
        Sample from logprobs in one-hot form.
    """
    z = -torch.log(-torch.log(torch.rand_like(logits)))
    
    # 对 logits + z 进行排序
    axis = len(logits.shape) - 1
    _, perm = torch.sort(logits + z, dim=axis, stable=False)
    
    # 返回反向排序后的索引
    return torch.flip(perm, dims=[axis])


def create_msa_feat(msa: Dict) -> torch.Tensor:
    """Create and concatenate MSA features."""
    msa_1hot = F.one_hot(msa.rows.to(torch.long), num_classes=residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP + 1)
    deletion_matrix = msa.deletion_matrix
    has_deletion = torch.clamp(deletion_matrix, 0.0, 1.0).unsqueeze(-1)
    deletion_value = (torch.atan(deletion_matrix / 3.0) * (2.0 / np.pi)).unsqueeze(-1)

    msa_feat = [
        msa_1hot,
        has_deletion,
        deletion_value,
    ]

    return torch.cat(msa_feat, dim=-1)


def truncate_msa_batch(msa: features.MSA, num_msa: int) -> features.MSA:
    indices = torch.arange(num_msa, device=msa.rows.device)
    return msa.index_msa_rows(indices)


def create_target_feat(
    batch: Dict,
    append_per_atom_features: bool,
) -> torch.Tensor:
    """Make target feat."""
    token_features = batch.token_features
    target_features = []
    target_features.append(
        F.one_hot(
            token_features.aatype.long(),
            num_classes=residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP,
        )
    )
    target_features.append(batch.msa.profile)
    target_features.append(batch.msa.deletion_mean.unsqueeze(-1))

    # Reference structure features
    if append_per_atom_features:
        ref_mask = batch.ref_structure.mask
        element_feat = F.one_hot(batch.ref_structure.element, num_classes=128)
        element_feat = (element_feat * ref_mask.unsqueeze(-1)).sum(dim=-2) / (ref_mask.sum(dim=-1, keepdim=True) + 1e-6)
        target_features.append(element_feat)
        pos_feat = batch.ref_structure.positions
        pos_feat = pos_feat.reshape([pos_feat.shape[0], -1])
        target_features.append(pos_feat)
        target_features.append(ref_mask)

    return torch.cat(target_features, dim=-1)


def create_relative_encoding(
    seq_features: Dict,
    max_relative_idx: int,
    max_relative_chain: int,
) -> torch.Tensor:
    """Add relative position encodings."""
    rel_feats = []
    token_index = seq_features.token_index
    residue_index = seq_features.residue_index
    asym_id = seq_features.asym_id
    entity_id = seq_features.entity_id
    sym_id = seq_features.sym_id

    left_asym_id = asym_id.unsqueeze(1)
    right_asym_id = asym_id.unsqueeze(0)

    left_residue_index = residue_index.unsqueeze(1)
    right_residue_index = residue_index.unsqueeze(0)

    left_token_index = token_index.unsqueeze(1)
    right_token_index = token_index.unsqueeze(0)

    left_entity_id = entity_id.unsqueeze(1)
    right_entity_id = entity_id.unsqueeze(0)

    left_sym_id = sym_id.unsqueeze(1)
    right_sym_id = sym_id.unsqueeze(0)

    # Embed relative positions using a one-hot embedding of distance along chain
    offset = left_residue_index - right_residue_index
    clipped_offset = torch.clamp(offset + max_relative_idx, min=0, max=2 * max_relative_idx)
    asym_id_same = left_asym_id == right_asym_id
    final_offset = torch.where(
        asym_id_same,
        clipped_offset,
        (2 * max_relative_idx + 1) * torch.ones_like(clipped_offset),
    )
    rel_pos = F.one_hot(final_offset.long(), num_classes=2 * max_relative_idx + 2)
    rel_feats.append(rel_pos)

    # Embed relative token index as a one-hot embedding of distance along residue
    token_offset = left_token_index - right_token_index
    clipped_token_offset = torch.clamp(
        token_offset + max_relative_idx, min=0, max=2 * max_relative_idx
    )
    residue_same = (left_asym_id == right_asym_id) & (left_residue_index == right_residue_index)
    final_token_offset = torch.where(
        residue_same,
        clipped_token_offset,
        (2 * max_relative_idx + 1) * torch.ones_like(clipped_token_offset),
    )
    # print(final_token_offset)
    rel_token = F.one_hot(final_token_offset.long(), num_classes=2 * max_relative_idx + 2)
    rel_feats.append(rel_token)

    # Embed same entity ID
    entity_id_same = left_entity_id == right_entity_id
    rel_feats.append(entity_id_same.float().unsqueeze(-1))

    # Embed relative chain ID inside each symmetry class
    rel_sym_id = left_sym_id - right_sym_id
    max_rel_chain = max_relative_chain
    clipped_rel_chain = torch.clamp(rel_sym_id + max_rel_chain, min=0, max=2 * max_rel_chain)
    final_rel_chain = torch.where(
        entity_id_same,
        clipped_rel_chain,
        (2 * max_rel_chain + 1) * torch.ones_like(clipped_rel_chain),
    )
    rel_chain = F.one_hot(final_rel_chain.long(), num_classes=2 * max_relative_chain + 2)
    rel_feats.append(rel_chain)

    return torch.cat(rel_feats, dim=-1)

def create_relative_encoding_scatter(
    seq_features: Dict,
    max_relative_idx: int,
    max_relative_chain: int,
) -> torch.Tensor:
    rel_feats = []
    token_index = seq_features.token_index
    residue_index = seq_features.residue_index
    asym_id = seq_features.asym_id
    entity_id = seq_features.entity_id
    sym_id = seq_features.sym_id

    left_asym_id = asym_id.unsqueeze(1)
    right_asym_id = asym_id.unsqueeze(0)

    left_residue_index = residue_index.unsqueeze(1)
    right_residue_index = residue_index.unsqueeze(0)

    left_token_index = token_index.unsqueeze(1)
    right_token_index = token_index.unsqueeze(0)

    left_entity_id = entity_id.unsqueeze(1)
    right_entity_id = entity_id.unsqueeze(0)

    left_sym_id = sym_id.unsqueeze(1)
    right_sym_id = sym_id.unsqueeze(0)

    # Embed relative positions using sparse operations
    offset = left_residue_index - right_residue_index
    clipped_offset = torch.clamp(offset + max_relative_idx, min=0, max=2 * max_relative_idx)
    
    # Use torch_scatter to efficiently compute final offset
    asym_id_same = (left_asym_id == right_asym_id).to(torch.long)
    final_offset = torch_scatter.scatter(
        clipped_offset, 
        asym_id_same, 
        dim=0, 
        reduce='max'  # Use a max reduction, or customize depending on need
    )

    rel_pos = F.one_hot(final_offset.long(), num_classes=2 * max_relative_idx + 2)
    rel_feats.append(rel_pos)

    # Embed relative token index using similar sparse approach
    token_offset = left_token_index - right_token_index
    clipped_token_offset = torch.clamp(
        token_offset + max_relative_idx, min=0, max=2 * max_relative_idx
    )

    residue_same = ((left_asym_id == right_asym_id) & (left_residue_index == right_residue_index)).to(torch.long)
    final_token_offset = torch_scatter.scatter(
        clipped_token_offset,
        residue_same,
        dim=0,
        reduce='max'
    )

    rel_token = F.one_hot(final_token_offset.long(), num_classes=2 * max_relative_idx + 2)
    rel_feats.append(rel_token)

    # Embed same entity ID using sparse scatter for efficiency
    entity_id_same = (left_entity_id == right_entity_id).to(torch.long)
    rel_feats.append(entity_id_same.float().unsqueeze(-1))

    # Relative chain ID inside each symmetry class
    rel_sym_id = left_sym_id - right_sym_id
    clipped_rel_chain = torch.clamp(rel_sym_id + max_relative_chain, min=0, max=2 * max_relative_chain)

    final_rel_chain = torch_scatter.scatter(
        clipped_rel_chain,
        entity_id_same,
        dim=0,
        reduce='max'
    )

    rel_chain = F.one_hot(final_rel_chain.long(), num_classes=2 * max_relative_chain + 2)
    rel_feats.append(rel_chain)
    # print(jax.tree_map(lambda x:x.shape, rel_feats))
    # quit()
    return torch.cat(rel_feats, dim=-1)

def shuffle_msa(
    msa: Dict
) -> Tuple[Dict, torch.Tensor]:
    """Shuffle MSA randomly, return batch with shuffled MSA.

    Args:
        key: rng key for random number generation.
        msa: MSA object to sample msa from.

    Returns:
        Protein with sampled msa.
    """
    logits = (torch.clamp(msa.mask.sum(dim=-1), 0.0, 1.0) - 1.0) * 1e6
    index_order = gumbel_argsort_sample_idx(logits)

    return msa.index_msa_rows(index_order)

if __name__ == '__main__':
    # 假设输入数据
    msa = {
        'rows': torch.randint(0, 20, (10, 20)),  # (num_seq, num_res)
        'deletion_matrix': torch.rand(10, 20),   # (num_seq, num_res)
    }
    batch = {
        'token_features': {
            'aatype': torch.randint(0, 20, (20,)),  # (num_res,)
        },
        'msa': {
            'profile': torch.rand(20, 20),  # (num_res, num_classes)
            'deletion_mean': torch.rand(20),  # (num_res,)
        },
        'ref_structure': {
            'mask': torch.ones(20, 10),  # (num_res, num_atoms)
            'element': torch.randint(0, 128, (20, 10)),  # (num_res, num_atoms)
            'positions': torch.rand(20, 10, 3),  # (num_res, num_atoms, 3)
        },
    }
    seq_features = {
        'token_index': torch.arange(20),  # (num_res,)
        'residue_index': torch.arange(20),  # (num_res,)
        'asym_id': torch.randint(0, 5, (20,)),  # (num_res,)
        'entity_id': torch.randint(0, 5, (20,)),  # (num_res,)
        'sym_id': torch.randint(0, 5, (20,)),  # (num_res,)
    }

    # 创建 MSA 特征
    msa_feat = create_msa_feat(msa)

    # 创建目标特征
    target_feat = create_target_feat(batch, append_per_atom_features=True)

    # 创建相对位置编码
    rel_encoding = create_relative_encoding(seq_features, max_relative_idx=32, max_relative_chain=5)

    # 打乱 MSA
    key = torch.randint(0, 2**32, (2,))
    shuffled_msa, key = shuffle_msa(key, msa)