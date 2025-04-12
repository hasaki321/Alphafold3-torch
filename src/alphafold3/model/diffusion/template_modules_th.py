import jax
import torch
import torch.nn as nn
import torch.nn.functional as F
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple, Union
from alphafold3.common import base_config

from alphafold3.common.base_config import BaseConfig
from alphafold3.constants import residue_names
from alphafold3.jax import geometry  # 假设 geometry 模块已转换为 PyTorch
from alphafold3.model import features, model_config, protein_data_processing
from alphafold3.model.components import torch_modules as tm
# from alphafold3.model.diffusion import modules  # 假设 modules 已转换为 PyTorch
from alphafold3.model.diffusion import modules_th as modules  # 假设 modules 已转换为 PyTorch
from alphafold3.model.scoring import scoring  # 假设 scoring 模块已转换为 PyTorchArray = jnp.ndarray | np.ndarray
from alphafold3.model.diffusion.template_utils import *

class TemplateEmbedding(nn.Module):
  """Embed a set of templates in PyTorch."""

#   class Config(BaseConfig):  # 假设 BaseConfig 已转换为 PyTorch
#     """Config for TemplateEmbedding in PyTorch."""
#     num_channels: int = 64
#     template_stack: modules.PairFormerIteration.Config = modules.PairFormerIteration.Config(num_layer=2, pair_transition=modules.PairFormerIteration.Config.pair_transition(num_intermediate_factor=2)) # 使用 PyTorch 版本的 PairFormerIteration.Config
#     dgram_features: DistogramFeaturesConfig = DistogramFeaturesConfig() # 使用 PyTorch 版本的 DistogramFeaturesConfig
  class Config(base_config.BaseConfig):
    num_channels: int = 64
    template_stack: modules.PairFormerIteration.Config = base_config.autocreate(
        num_layer=2,
        pair_transition=base_config.autocreate(num_intermediate_factor=2),
    )
    dgram_features: DistogramFeaturesConfig = base_config.autocreate()

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      input_channels: int,
      name='template_embedding',
  ):
    super().__init__()
    self.config = config
    self.global_config = global_config
    self.single_template_embedding = SingleTemplateEmbedding(config, global_config, input_channels=input_channels, name='single_template_embedding') # 实例化 SingleTemplateEmbedding
    self.output_linear = nn.Linear(config.num_channels, input_channels,bias=False) # tm.Linear 假设为 PyTorch Linear


  def forward(
      self,
      query_embedding: torch.Tensor,
      templates: features.Templates, # 使用 PyTorch 版本的 features.Templates
      padding_mask_2d: torch.Tensor,
      multichain_mask_2d: torch.Tensor,
  ) -> torch.Tensor:
    """Generate an embedding for a set of templates.

    Args:
      query_embedding: [num_res, num_res, num_channel] a query tensor that will
        be used to attend over the templates to remove the num_templates
        dimension.
      templates: A 'Templates' object.
      padding_mask_2d: [num_res, num_res] Pair mask for attention operations.
      multichain_mask_2d: [num_res, num_res] Pair mask for multichain.
      key: random key generator.

    Returns:
      An embedding of size [num_res, num_res, num_channels]
    """
    config = self.config
    num_residues = query_embedding.shape[0]
    num_templates = templates.aatype.shape[0]
    query_num_channels = query_embedding.shape[2]
    num_atoms = 24
    assert query_embedding.shape == (
        num_residues,
        num_residues,
        query_num_channels,
    )
    assert templates.aatype.shape == (num_templates, num_residues)
    assert templates.atom_positions.shape == (
        num_templates,
        num_residues,
        num_atoms,
        3,
    )
    assert templates.atom_mask.shape == (num_templates, num_residues, num_atoms)
    assert padding_mask_2d.shape == (num_residues, num_residues)

    num_templates = templates.aatype.shape[0]
    num_res, _, query_num_channels = query_embedding.shape

    # Embed each template separately.
    template_embedder = self.single_template_embedding # 使用 self.single_template_embedding

    summed_template_embeddings = torch.zeros(
        (num_res, num_res, config.num_channels), dtype=query_embedding.dtype
    ) # 使用 torch.zeros 初始化

    # print(templates)
    # templates = jax.tree_map(np.array, templates)
    # multichain_mask_2d = np.array(multichain_mask_2d.float())
    for i in range(num_templates): # 循环处理模板
        class current_template:
          aatype =  templates.aatype[i]
          atom_positions =  templates.atom_positions[i]
          atom_mask =  templates.atom_mask[i]
        

        # 调用 template_embedder 函数
        embedding = template_embedder(
            query_embedding,
            current_template,
            padding_mask_2d,
            multichain_mask_2d,
        )
        summed_template_embeddings += embedding # 累加 embedding

    embedding = summed_template_embeddings / (1e-7 + num_templates)
    embedding = F.relu(embedding) # 使用 F.relu
    embedding = self.output_linear(embedding) # 使用 self.output_linear

    assert embedding.shape == (num_residues, num_residues, query_num_channels)
    return embedding


class SingleTemplateEmbedding(nn.Module):
  """Embed a single template in PyTorch."""

  def __init__(
      self,
      config: TemplateEmbedding.Config,
      global_config: model_config.GlobalConfig,
      input_channels: int,
      name='single_template_embedding',
  ):
    super().__init__()
    self.config = config
    self.global_config = global_config
    self.query_embedding_norm = tm.LayerNorm(normalized_shape=input_channels, name='query_embedding_norm') # tm.LayerNorm 假设为 PyTorch LayerNorm
    self.template_pair_embedding_layers = nn.ModuleList([\
      nn.Linear(input_dim, config.num_channels,bias=False)\
        for i, input_dim in\
            enumerate([39,1,31,31,1,1,1,1,input_channels])]) # 使用 nn.ModuleList 和 tm.Linear
    self.template_embedding_iteration_layers = nn.ModuleList([\
      modules.PairFormerIteration(config.template_stack, global_config, config.num_channels, name=f'template_embedding_iteration_layer_{i}') \
        for i in range(config.template_stack.num_layer)]) # 使用 nn.ModuleList 和 PyTorch 版本的 PairFormerIteration
    self.output_layer_norm = tm.LayerNorm(normalized_shape=config.num_channels, name='output_layer_norm') # tm.LayerNorm 假设为 PyTorch LayerNorm


  def forward(
      self,
      query_embedding: torch.Tensor,
      templates: features.Templates, # 使用 PyTorch 版本的 features.Templates
      padding_mask_2d: torch.Tensor,
      multichain_mask_2d: torch.Tensor,
  ) -> torch.Tensor:
    """Build the single template embedding graph.

    Args:
      query_embedding: (num_res, num_res, num_channels) - embedding of the query
        sequence/msa.
      templates: 'Templates' object containing single Template.
      padding_mask_2d: Padding mask (Note: this doesn't care if a template
        exists, unlike the template_pseudo_beta_mask).
      multichain_mask_2d: A mask indicating intra-chain residue pairs, used to
        mask out between chain distances/features when templates are for single
        chains.
      key: Random key generator.

    Returns:
      A template embedding (num_res, num_res, num_channels).
    """
    config = self.config
    # assert padding_mask_2d.dtype == query_embedding.dtype

    def construct_input(
        query_embedding, 
        aatype: torch.Tensor,
        atom_positions: torch.Tensor,
        atom_mask: torch.Tensor,
        multichain_mask_2d
    ):
      # Compute distogram feature for the template.
      dtype = torch.float32
      input_aatype = aatype.long()

      dense_atom_positions = atom_positions
      dense_atom_positions *= atom_mask[..., None]

      pseudo_beta_positions, pseudo_beta_mask = pseudo_beta_fn_th(
          input_aatype, dense_atom_positions, atom_mask
      )
      pseudo_beta_mask_2d = (
          pseudo_beta_mask[:, None] * pseudo_beta_mask[None, :]
      )
      pseudo_beta_mask_2d *= multichain_mask_2d

      dgram = dgram_from_positions_th(
          pseudo_beta_positions, self.config.dgram_features
      )
      dgram *= pseudo_beta_mask_2d[..., None]

      # dgram = dgram.astype(dtype)

      # pseudo_beta_mask_2d = pseudo_beta_mask_2d.astype(dtype)

      to_concat = [(dgram, 1), (pseudo_beta_mask_2d, 0)]

      aatype = F.one_hot(
          input_aatype,
          num_classes = residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP,
      ).float()
      to_concat.append((aatype[None, :, :], 1))
      to_concat.append((aatype[:, None, :], 1))

      # Compute a feature representing the normalized vector between each
      # backbone affine - i.e. in each residues local frame, what direction are
      # each of the other residues.

      template_group_indices = torch.tensor(protein_data_processing.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX)[input_aatype]
          
      # unit_vector, backbone_mask = make_backbone_rigid_th(
      #     dense_atom_positions,
      #     atom_mask,
      #     template_group_indices.to(torch.int32),
      # )
      unit_vector, backbone_mask = make_backbone_vectors_th(
          dense_atom_positions,
          atom_mask,
          template_group_indices.to(torch.int32),
      )
      # print(unit_vector[0].shape)
      # print(unit_vector1[0].shape)
      # assert torch.allclose(backbone_mask, backbone_mask1)
      # assert torch.allclose(unit_vector[0], unit_vector1[0])
      unit_vector = [x.to(dtype) for x in unit_vector]
      backbone_mask = backbone_mask.to(dtype)

      backbone_mask_2d = backbone_mask[:, None] * backbone_mask[None, :]
      backbone_mask_2d *= multichain_mask_2d
      unit_vector = [x * backbone_mask_2d for x in unit_vector]

      # Note that the backbone_mask takes into account C, CA and N (unlike
      # pseudo beta mask which just needs CB) so we add both masks as features.
      to_concat.extend([(x, 0) for x in unit_vector])
      to_concat.append((backbone_mask_2d, 0))

      query_embedding = self.query_embedding_norm( # 使用 self.query_embedding_norm
          query_embedding
      )
      # Allow the template embedder to see the query embedding.  Note this
      # contains the position relative feature, so this is how the network knows
      # which residues are next to each other.
      to_concat.append((query_embedding, 1))

      act = 0

      for i, (x, n_input_dims) in enumerate(to_concat):
        inputs = x.float()
        inputs = inputs.unsqueeze(-1) if len(inputs.shape) < 3 else inputs
        act += self.template_pair_embedding_layers[i](inputs) # 循环调用 template_pair_embedding_layers
      return act

    # act = construct_input(query_embedding, templates, multichain_mask_2d)
    aatype = templates.aatype
    dense_atom_positions = templates.atom_positions
    dense_atom_mask = templates.atom_mask
    act = construct_input(query_embedding, 
                          aatype,
                          dense_atom_positions,
                          dense_atom_mask,
                          multichain_mask_2d)
    # torch.save(act,'align/construct_input')
    # quit()
    if config.template_stack.num_layer:
      for layer in self.template_embedding_iteration_layers: # 循环调用 template_embedding_iteration_layers
          act = layer(act=act, pair_mask=padding_mask_2d) # 调用 PairFormerIteration


    act = self.output_layer_norm(act) # 使用 self.output_layer_norm
    return act