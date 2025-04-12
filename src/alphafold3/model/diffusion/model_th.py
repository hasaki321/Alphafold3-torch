# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Diffusion model."""


from collections.abc import Iterable, Mapping
import concurrent
import dataclasses
import functools
from typing import Any, TypeAlias, Tuple, Optional, Dict
from alphafold3.model.components import torch_modules as tm

import time
import torch
import torch.nn as nn
from absl import logging
from alphafold3 import structure
from alphafold3.common import base_config
from alphafold3.model import confidences
# from alphafold3.model import confidences_th as confidences
from alphafold3.model import feat_batch
from alphafold3.model import features
from alphafold3.model import model_config
from alphafold3.model.atom_layout import atom_layout
# from alphafold3.model.components import haiku_modules as hm
from alphafold3.model.components import torch_modules as tm
from alphafold3.model.components import mapping_th
from alphafold3.model.components import utils
from alphafold3.model.diffusion import atom_cross_attention_th as atom_cross_attention
from alphafold3.model.diffusion import confidence_head_th as confidence_head
# from alphafold3.model.diffusion import diffusion_head

from alphafold3.model.diffusion import distogram_head_th as distogram_head
from alphafold3.model.diffusion import featurization_th as featurization
# from alphafold3.model.diffusion import featurization
from alphafold3.model.diffusion import modules_th as modules
from alphafold3.model.diffusion import template_modules_th as template_modules
from alphafold3.structure import mmcif
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

import os
BATCH_DIFFUSION = os.getenv('BATCH') 
if BATCH_DIFFUSION == 'true':
    from alphafold3.model.diffusion import diffusion_head_th_batch as diffusion_head
    print("Enable Batched diffusion")
else:
    from alphafold3.model.diffusion import diffusion_head_th as diffusion_head
    print("Enable Vmap diffusion")

ModelResult: TypeAlias = Mapping[str, Any]
_ScalarNumberOrArray: TypeAlias = Mapping[str, float | int | np.ndarray]


@dataclasses.dataclass(frozen=True)
class InferenceResult:
  """Postprocessed model result.

  Attributes:
    predicted_structure: Predicted protein structure.
    numerical_data: Useful numerical data (scalars or arrays) to be saved at
      inference time.
    metadata: Smaller numerical data (usually scalar) to be saved as inference
      metadata.
    debug_outputs: Additional dict for debugging, e.g. raw outputs of a model
      forward pass.
    model_id: Model identifier.
  """

  predicted_structure: structure.Structure = dataclasses.field()
  numerical_data: _ScalarNumberOrArray = dataclasses.field(default_factory=dict)
  metadata: _ScalarNumberOrArray = dataclasses.field(default_factory=dict)
  debug_outputs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  model_id: bytes = b''


def get_predicted_structure(
    result: ModelResult, batch: feat_batch.Batch
) -> structure.Structure:
  """Creates the predicted structure and ion preditions.

  Args:
    result: model output in a model specific layout
    batch: model input batch

  Returns:
    Predicted structure.
  """
  model_output_coords = result['diffusion_samples']['atom_positions']

  # Rearrange model output coordinates to the flat output layout.
  # TODO
  model_output_to_flat = atom_layout.compute_gather_idxs(
      source_layout=batch.convert_model_output.token_atoms_layout,
      target_layout=batch.convert_model_output.flat_output_layout,
  )
  pred_flat_atom_coords = atom_layout.convert(
      gather_info=model_output_to_flat,
      arr=model_output_coords,
      layout_axes=(-3, -2),
  )

  predicted_lddt = result.get('predicted_lddt')

  if predicted_lddt is not None:
    pred_flat_b_factors = atom_layout.convert(
        gather_info=model_output_to_flat,
        arr=predicted_lddt,
        layout_axes=(-2, -1),
    )
  else:
    # Handle models which don't have predicted_lddt outputs.
    pred_flat_b_factors = np.zeros(pred_flat_atom_coords.shape[:-1])

  (missing_atoms_indices,) = np.nonzero(model_output_to_flat.gather_mask == 0)
  if missing_atoms_indices.shape[0] > 0:
    missing_atoms_flat_layout = batch.convert_model_output.flat_output_layout[
        missing_atoms_indices
    ]
    missing_atoms_uids = list(
        zip(
            missing_atoms_flat_layout.chain_id,
            missing_atoms_flat_layout.res_id,
            missing_atoms_flat_layout.res_name,
            missing_atoms_flat_layout.atom_name,
        )
    )
    logging.warning(
        'Target %s: warning: %s atoms were not predicted by the '
        'model, setting their coordinates to (0, 0, 0). '
        'Missing atoms: %s',
        batch.convert_model_output.empty_output_struc.name,
        missing_atoms_indices.shape[0],
        missing_atoms_uids,
    )

  # Put them into a structure
  pred_struc = batch.convert_model_output.empty_output_struc
  pred_struc = pred_struc.copy_and_update_atoms(
      atom_x=pred_flat_atom_coords[..., 0],
      atom_y=pred_flat_atom_coords[..., 1],
      atom_z=pred_flat_atom_coords[..., 2],
      atom_b_factor=pred_flat_b_factors,
      atom_occupancy=np.ones(pred_flat_atom_coords.shape[:-1]),  # Always 1.0.
  )
  # Set manually/differently when adding metadata.
  pred_struc = pred_struc.copy_and_update_globals(release_date=None)
  return pred_struc




def _compute_ptm(
    result: ModelResult,
    num_tokens: int,
    asym_id: np.ndarray,
    pae_single_mask: np.ndarray,
    interface: bool,
) -> np.ndarray:
  """Computes the pTM metrics from PAE."""
  return np.stack(
      [
          confidences.predicted_tm_score(
              tm_adjusted_pae=tm_adjusted_pae[:num_tokens, :num_tokens],
              asym_id=asym_id,
              pair_mask=pae_single_mask[:num_tokens, :num_tokens],
              interface=interface,
          )
          for tm_adjusted_pae in result['tmscore_adjusted_pae_global']
      ],
      axis=0,
  )


def _compute_chain_pair_iptm(
    num_tokens: int,
    asym_ids: np.ndarray,
    mask: np.ndarray,
    tm_adjusted_pae: np.ndarray,
) -> np.ndarray:
  """Computes the chain pair ipTM metrics from PAE."""
  return np.stack(
      [
          confidences.chain_pairwise_predicted_tm_scores(
              tm_adjusted_pae=sample_tm_adjusted_pae[:num_tokens],
              asym_id=asym_ids[:num_tokens],
              pair_mask=mask[:num_tokens, :num_tokens],
          )
          for sample_tm_adjusted_pae in tm_adjusted_pae
      ],
      axis=0,
  )

class HeadsConfig(base_config.BaseConfig):
    diffusion: diffusion_head.DiffusionHead.Config = base_config.autocreate()
    confidence = base_config.autocreate()
    distogram = base_config.autocreate()
    # confidence: confidence_head.ConfidenceHead.Config = base_config.autocreate()
    # distogram: distogram_head.DistogramHead.Config = base_config.autocreate()

class Config(base_config.BaseConfig):
    evoformer: 'Evoformer.Config' = base_config.autocreate()
    global_config: model_config.GlobalConfig = base_config.autocreate()
    heads: 'Diffuser.HeadsConfig' = base_config.autocreate()
    num_recycles: int = 10
    return_embeddings: bool = False

class Diffuser(torch.nn.Module):
  """Full Diffusion network."""
  class HeadsConfig(base_config.BaseConfig):
    diffusion: diffusion_head.DiffusionHead.Config = base_config.autocreate()
    # confidence = base_config.autocreate()
    # distogram = base_config.autocreate()
    confidence: confidence_head.ConfidenceHead.Config = base_config.autocreate()
    distogram: distogram_head.DistogramHead.Config = base_config.autocreate()

  class Config(base_config.BaseConfig):
    evoformer: 'Evoformer.Config' = base_config.autocreate()
    global_config: model_config.GlobalConfig = base_config.autocreate()
    heads: 'Diffuser.HeadsConfig' = base_config.autocreate()
    num_recycles: int = 10
    return_embeddings: bool = False

  def __init__(self, config: Config, name: str = 'diffuser'):
    super().__init__()
    self.config = config
    self.global_config = config.global_config
    self.name = name

    self.use_compile = True

    self.diffusion_head = diffusion_head.DiffusionHead( 
        self.config.heads.diffusion, self.global_config, self.config.evoformer.pair_channel, self.config.evoformer.seq_channel
    )

    self.confidence_head = confidence_head.ConfidenceHead(
        self.config.heads.confidence, self.global_config
    )

    self.evoformer_conditioning = atom_cross_attention.AtomCrossAttEncoder(
        config.evoformer.per_atom_conditioning, 
        config.global_config, 
        pair_channel=self.config.evoformer.pair_channel,
        single_channel=self.config.evoformer.seq_channel,
        name='evoformer_conditioning')
    
    self.evoformer = Evoformer(self.config.evoformer, self.global_config) # embedding_module

    self.distogram = distogram_head.DistogramHead(
        self.config.heads.distogram, self.global_config
    )
    # self.test_align = False
    # if self.use_compile:
    #    self.diffusion_head = torch.compile(self.diffusion_head)
    #    self.confidence_head = torch.compile(self.confidence_head)
    #    self.evoformer_conditioning = torch.compile(self.evoformer_conditioning)
    #    self.evoformer = torch.compile(self.evoformer)
    
  def _sample_diffusion(
      self,
      batch: feat_batch.Batch,
      embeddings: dict[str, torch.Tensor],
      *,
      sample_config: diffusion_head.SampleConfig,
  ) -> dict[str, jnp.ndarray]:
    denoising_step = functools.partial(
        self.diffusion_head,
        batch=batch,
        embeddings=embeddings,
        use_conditioning=True,
    )
    # if self.test_align:
    #     sample_config.num_samples = 1
    #     sample_config.steps = 1
    # sample_config.steps = 3
    sample = diffusion_head.sample(
        denoising_step=denoising_step,
        batch=batch,
        # key=hk.next_rng_key(),
        config=sample_config,
    )
    return sample

  def forward(self, batch: features.BatchDict) -> ModelResult:
    # if key is None:
    #   key = hk.next_rng_key()

    # import time
    

    embedding_module = self.evoformer
    # start_time = time.time()
    target_feat = self.create_target_feat_embedding(
        batch=batch,
        config=embedding_module.config,
        global_config=self.global_config,
    )
    # print(f"create_target_feat_embedding ending with {(time.time() - start_time):.4f} sec")
    # if self.test_align:
    #    torch.save(target_feat,'align/target_feat_cat')
    #    print('save target_feat_cat and quit');quit()
    def recycle_body(prev):
      embeddings = embedding_module(
          batch=batch,
          prev=prev,
          target_feat=target_feat,
      )
      embeddings['pair'] = embeddings['pair'].to(torch.float32)
      embeddings['single'] = embeddings['single'].to(torch.float32)
      return embeddings

    num_res = batch.num_res

    embeddings = {
        'pair': torch.zeros(
            (num_res, num_res, self.config.evoformer.pair_channel),
            dtype=torch.float32,
        ),
        'single': torch.zeros(
            (num_res, self.config.evoformer.seq_channel), dtype=torch.float32
        ),
        'target_feat': target_feat,
    }
    # if self.test_align:
    #     torch.save(target_feat,'target_feat_cat')
    # if not self.test_align:
        # print("EvoFormer start")
        # start_evo = time.time()
    num_iter = self.config.num_recycles
    # num_iter = 3
    for _ in range(num_iter):
        # start_time = time.time()
        embeddings= recycle_body(embeddings)
            # print(f"Loop ending with {(time.time() - start_time):.4f} sec")
            # if self.test_align:
            #     torch.save(embeddings['pair'],f'./align/embeddings_pair_{_}')
            #     torch.save(embeddings['single'],f'./align/embeddings_single_{_}')
        # print(f"EvoFormer ending with {(time.time() - start_evo):.4f} sec")


    # print("Diffusion start")
    # start_diff = time.time()
    samples = self._sample_diffusion(
        batch,
        embeddings,
        sample_config=self.config.heads.diffusion.eval,
    )
    # print(f"EvoFormer ending with {(time.time() - start_diff):.4f} sec")
    # print('save and quit after samples');quit()

    # Compute dist_error_fn over all samples for distance error logging.
    # print("confidence_output start"); start_confidence = time.time()
    confidence_output = mapping_th.sharded_map(
        lambda dense_atom_positions: self.confidence_head(
            dense_atom_positions=dense_atom_positions,
            embeddings=embeddings,
            seq_mask=batch.token_features.mask,
            token_atoms_to_pseudo_beta=batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
            asym_id=batch.token_features.asym_id,
        ),
        in_axes=0,
    )(samples['atom_positions'])
    # print(f"confidence head took {time.time()-start_confidence:.4f}s") 
    # confidence_output = self.confidence_head(
    #         dense_atom_positions=samples['atom_positions'],
    #         embeddings=embeddings,
    #         seq_mask=batch.token_features.mask,
    #         token_atoms_to_pseudo_beta=batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
    #         asym_id=batch.token_features.asym_id,
    # )
    distogram = self.distogram(batch, embeddings)

    # print('keys in confidence_output:')
    # for k in confidence_output:
    #     print('  ',k)
    #     torch.save(confidence_output[k], f'align/{k}')
    # print('keys in distogram:')
    # for k,v in distogram.items():
    #     print('  ',k)
    #     torch.save(distogram[k], f'align/{k}')
    # print('save and quit after confidence output'); quit()

    output = {
        'diffusion_samples': samples,
        'distogram': distogram,
        **confidence_output,
    }
    if self.config.return_embeddings:
      output['single_embeddings'] = embeddings['single']
      output['pair_embeddings'] = embeddings['pair']
    return output

  def create_target_feat_embedding(
        self,
        batch: feat_batch.Batch,
        config: 'Evoformer.Config',
        global_config: model_config.GlobalConfig,
    ) -> jnp.ndarray:
        """Create target feature embedding."""

    #   dtype = jnp.bfloat16 if global_config.bfloat16 == 'all' else jnp.float32
        dtype = torch.bfloat16
    #   with utils.bfloat16_context():
        target_feat = featurization.create_target_feat(
            batch,
            append_per_atom_features=False,
        ).to(dtype)
        # target_feat = target_feat.

        enc = self.evoformer_conditioning(
            token_atoms_act=None,
            trunk_single_cond=None,
            trunk_pair_cond=None,
            batch=batch,
        )
        # print(target_feat.shape, enc.token_act.shape)
        # quit()
        target_feat = torch.concatenate([target_feat, enc.token_act], axis=-1).to(
            dtype
        )

        return target_feat


  @classmethod
  def get_inference_result(
      cls,
      batch: features.BatchDict,
      result: ModelResult,
      target_name: str = '',
  ) -> Iterable[InferenceResult]:
    """Get the predicted structure, scalars, and arrays for inference.

    This function also computes any inference-time quantities, which are not a
    part of the forward-pass, e.g. additional confidence scores. Note that this
    function is not serialized, so it should be slim if possible.

    Args:
      batch: data batch used for model inference, incl. TPU invalid types.
      result: output dict from the model's forward pass.
      target_name: target name to be saved within structure.

    Yields:
      inference_result: dataclass object that contains a predicted structure,
      important inference-time scalars and arrays, as well as a slightly trimmed
      dictionary of raw model result from the forward pass (for debugging).
    """
    del target_name
    batch = feat_batch.Batch.from_data_dict(batch)

    # Retrieve structure and construct a predicted structure.
    pred_structure = get_predicted_structure(result=result, batch=batch)

    num_tokens = batch.token_features.seq_length.item()

    pae_single_mask = np.tile(
        batch.frames.mask[:, None],
        [1, batch.frames.mask.shape[0]],
    )
    ptm = _compute_ptm(
        result=result,
        num_tokens=num_tokens,
        asym_id=batch.token_features.asym_id[:num_tokens],
        pae_single_mask=pae_single_mask,
        interface=False,
    )
    iptm = _compute_ptm(
        result=result,
        num_tokens=num_tokens,
        asym_id=batch.token_features.asym_id[:num_tokens],
        pae_single_mask=pae_single_mask,
        interface=True,
    )
    ptm_iptm_average = 0.8 * iptm + 0.2 * ptm

    asym_ids = batch.token_features.asym_id[:num_tokens]
    chain_ids = [mmcif.int_id_to_str_id(asym_id) for asym_id in asym_ids]
    res_ids = batch.token_features.residue_index[:num_tokens]

    if len(np.unique(asym_ids[:num_tokens])) > 1:
      # There is more than one chain, hence interface pTM (i.e. ipTM) defined,
      # so use it.
      ranking_confidence = ptm_iptm_average
    else:
      # There is only one chain, hence ipTM=NaN, so use just pTM.
      ranking_confidence = ptm

    contact_probs = result['distogram']['contact_probs']
    # Compute PAE related summaries.
    _, chain_pair_pae_min, _ = confidences.chain_pair_pae(
        num_tokens=num_tokens,
        asym_ids=batch.token_features.asym_id,
        full_pae=result['full_pae'],
        mask=pae_single_mask,
    )
    chain_pair_pde_mean, chain_pair_pde_min = confidences.chain_pair_pde(
        num_tokens=num_tokens,
        asym_ids=batch.token_features.asym_id,
        full_pde=result['full_pde'],
    )
    intra_chain_single_pde, cross_chain_single_pde, _ = confidences.pde_single(
        num_tokens,
        batch.token_features.asym_id,
        result['full_pde'],
        contact_probs,
    )
    pae_metrics = confidences.pae_metrics(
        num_tokens=num_tokens,
        asym_ids=batch.token_features.asym_id,
        full_pae=result['full_pae'],
        mask=pae_single_mask,
        contact_probs=contact_probs,
        tm_adjusted_pae=result['tmscore_adjusted_pae_interface'],
    )
    ranking_confidence_pae = confidences.rank_metric(
        result['full_pae'],
        contact_probs * batch.frames.mask[:, None].astype(float),
    )
    chain_pair_iptm = _compute_chain_pair_iptm(
        num_tokens=num_tokens,
        asym_ids=batch.token_features.asym_id,
        mask=pae_single_mask,
        tm_adjusted_pae=result['tmscore_adjusted_pae_interface'],
    )
    # iptm_ichain is a vector of per-chain ptm values. iptm_ichain[0],
    # for example, is just the zeroth diagonal entry of the chain pair iptm
    # matrix:
    # [[x, , ],
    #  [ , , ],
    #  [ , , ]]]
    iptm_ichain = chain_pair_iptm.diagonal(axis1=-2, axis2=-1)
    # iptm_xchain is a vector of cross-chain interactions for each chain.
    # iptm_xchain[0], for example, is an average of chain 0's interactions with
    # other chains:
    # [[ ,x,x],
    #  [x, , ],
    #  [x, , ]]]
    iptm_xchain = confidences.get_iptm_xchain(chain_pair_iptm)

    predicted_distance_errors = result['average_pde']

    # Computing solvent accessible area with dssp can be slow for large
    # structures with lots of chains, so we parallelize the call.
    pred_structures = pred_structure.unstack()
    num_workers = len(pred_structures)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:
      has_clash = list(executor.map(confidences.has_clash, pred_structures))
      fraction_disordered = list(
          executor.map(confidences.fraction_disordered, pred_structures)
      )

    for idx, pred_structure in enumerate(pred_structures):
      ranking_score = confidences.get_ranking_score(
          ptm=ptm[idx],
          iptm=iptm[idx],
          fraction_disordered_=fraction_disordered[idx],
          has_clash_=has_clash[idx],
      )
      yield InferenceResult(
          predicted_structure=pred_structure,
          numerical_data={
              'full_pde': result['full_pde'][idx, :num_tokens, :num_tokens],
              'full_pae': result['full_pae'][idx, :num_tokens, :num_tokens],
              'contact_probs': contact_probs[:num_tokens, :num_tokens],
          },
          metadata={
              'predicted_distance_error': predicted_distance_errors[idx],
              'ranking_score': ranking_score,
              'fraction_disordered': fraction_disordered[idx],
              'has_clash': has_clash[idx],
              'predicted_tm_score': ptm[idx],
              'interface_predicted_tm_score': iptm[idx],
              'chain_pair_pde_mean': chain_pair_pde_mean[idx],
              'chain_pair_pde_min': chain_pair_pde_min[idx],
              'chain_pair_pae_min': chain_pair_pae_min[idx],
              'ptm': ptm[idx],
              'iptm': iptm[idx],
              'ptm_iptm_average': ptm_iptm_average[idx],
              'intra_chain_single_pde': intra_chain_single_pde[idx],
              'cross_chain_single_pde': cross_chain_single_pde[idx],
              'pae_ichain': pae_metrics['pae_ichain'][idx],
              'pae_xchain': pae_metrics['pae_xchain'][idx],
              'ranking_confidence': ranking_confidence[idx],
              'ranking_confidence_pae': ranking_confidence_pae[idx],
              'chain_pair_iptm': chain_pair_iptm[idx],
              'iptm_ichain': iptm_ichain[idx],
              'iptm_xchain': iptm_xchain[idx],
              'token_chain_ids': chain_ids,
              'token_res_ids': res_ids,
          },
        #   model_id=result['__identifier__'],
          debug_outputs={},
      )


class Evoformer(nn.Module):
    """Creates 'single' and 'pair' embeddings."""

    class PairformerConfig(modules.PairFormerIteration.Config):  # pytype: disable=invalid-function-definition
        block_remat: bool = False
        remat_block_size: int = 8

    class Config(base_config.BaseConfig):
        """Configuration for Evoformer."""

        max_relative_chain: int = 2
        msa_channel: int = 64
        seq_channel: int = 384
        max_relative_idx: int = 32
        num_msa: int = 1024
        pair_channel: int = 128
        pairformer: 'Evoformer.PairformerConfig' = base_config.autocreate(
            single_transition=base_config.autocreate(),
            single_attention=base_config.autocreate(),
            num_layer=48,
        )
        per_atom_conditioning: atom_cross_attention.AtomCrossAttEncoderConfig = (
            base_config.autocreate(
                per_token_channels=384,
                per_atom_channels=128,
                atom_transformer=base_config.autocreate(
                    num_intermediate_factor=2,
                    num_blocks=3,
                ),
                per_atom_pair_channels=16,
            )
        )
        template: template_modules.TemplateEmbedding.Config = (
            base_config.autocreate()
        )
        msa_stack: modules.EvoformerIteration.Config = (base_config.autocreate(num_layer=4))

    def __init__(
        self,
        config: Config,
        global_config: model_config.GlobalConfig,
        name='evoformer',
    ):
        super().__init__()
        self.name = name
        self.config = config
        self.global_config = global_config
        self.create_param()

    def create_param(self):
        token_class = 32 * 2 + 2
        rel_chain_class =  2 * 2 + 2
        self.pairformer_layers = self.config.pairformer.num_layer

        self.position_activations = nn.Linear(token_class + token_class + 1 + rel_chain_class,self.config.pair_channel, bias=False) 
        self.left_single = nn.Linear(447, self.config.pair_channel, bias=False)       
        self.right_single = nn.Linear(447, self.config.pair_channel, bias=False)     
        self.bond_embedding = nn.Linear(1, self.config.pair_channel, bias=False)   

        self.msa_activations = nn.Linear(34,self.config.msa_channel, bias=False) 
        self.extra_msa_target_feat = nn.Linear(447,self.config.msa_channel, bias=False) 
        
        self.prev_embedding_layer_norm = tm.LayerNorm(normalized_shape=self.config.pair_channel, 
                                                        name='prev_embedding_layer_norm'
                                                        )
        self.prev_embedding = nn.Linear(self.config.pair_channel, self.config.pair_channel, bias=False) 
        self.single_activations = nn.Linear(447, self.config.seq_channel, bias=False) 
        self.prev_single_embedding_layer_norm = tm.LayerNorm(normalized_shape=self.config.seq_channel, 
                                                        name='prev_single_embedding_layer_norm'
                                                        ) 
        self.prev_single_embedding = nn.Linear(self.config.seq_channel, self.config.seq_channel, bias=False) 

        self.template_embedding = template_modules.TemplateEmbedding(self.config.template, self.global_config, input_channels=self.config.pair_channel) # 使用 PyTorch 版本的 TemplateEmbedding
        self.msa_stack_layers = nn.ModuleList([
           modules.EvoformerIteration(self.config.msa_stack, self.global_config, msa_channels=self.config.msa_channel, pair_channels=self.config.pair_channel,) 
           for i in range(self.config.msa_stack.num_layer)])
        
        self.pairformer_stack_layers = nn.ModuleList([
           modules.PairFormerIteration(self.config.pairformer, self.global_config, num_channels=self.config.pair_channel, seq_channel=self.config.seq_channel,with_single=True, name=f'trunk_pairformer_layer_{i}'
        ) for i in range(self.pairformer_layers)]) # 使用 nn.ModuleList 和 PyTorch 版本的 PairFormerIteration

    def _relative_encoding(
        self, batch: feat_batch.Batch, pair_activations: torch.Tensor
    ) -> torch.Tensor:
        import time
        """Add relative position encodings in PyTorch."""
        # start_time = time.time()
        rel_feat = featurization.create_relative_encoding( # 假设 featurization.create_relative_encoding 已转换
            batch.token_features,
            self.config.max_relative_idx,
            self.config.max_relative_chain,
        )

        # assert ~((rel_feat != rel_feat_s).any())
        rel_feat = rel_feat.to(pair_activations.dtype) # 转换 dtype

        pair_activations += self.position_activations(rel_feat)
        return pair_activations

    def _seq_pair_embedding(
        self,
        token_features: features.TokenFeatures, # 使用 PyTorch 版本的 features.TokenFeatures
        target_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generated Pair embedding from sequence in PyTorch."""
        left_single = self.left_single(target_feat)[:, None, :] # 使用 self.left_single
        right_single = self.right_single(target_feat)[None, :, :] # 使用 self.right_single
        # dtype = left_single.dtype
        pair_activations = left_single + right_single
        num_residues = pair_activations.shape[0]
        assert pair_activations.shape == (
            num_residues,
            num_residues,
            self.config.pair_channel,
        )
        mask = token_features.mask
        pair_mask = (mask[:, None] * mask[None, :])
        # .to(dtype) # 转换 mask dtype
        assert pair_mask.shape == (num_residues, num_residues)
        return pair_activations, pair_mask

    def _embed_bonds(
        self,
        batch: feat_batch.Batch, # 使用 PyTorch 版本的 feat_batch.Batch
        pair_activations: torch.Tensor,
    ) -> torch.Tensor:
        """Embeds bond features and merges into pair activations in PyTorch."""
        # Construct contact matrix.
        num_tokens = batch.token_features.token_index.shape[0]
        contact_matrix = torch.zeros((num_tokens, num_tokens)) # 使用 torch.zeros

        tokens_to_polymer_ligand_bonds = (
            batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds
        )
        # TODO [Acc]
        # print(tokens_to_polymer_ligand_bonds.gather_idxs.shape)
        # print(tokens_to_polymer_ligand_bonds.gather_mask.shape)
        gather_idxs_polymer_ligand = tokens_to_polymer_ligand_bonds.gather_idxs
        gather_mask_polymer_ligand = (
            tokens_to_polymer_ligand_bonds.gather_mask.prod(axis=1).to(
                gather_idxs_polymer_ligand.dtype
            )[:, None]
        )
        # gather_mask_polymer_ligand = tokens_to_polymer_ligand_bonds.gather_mask.all(axis=1, keepdims=True, dtype=gather_idxs_polymer_ligand.dtype)

        # If valid mask then it will be all 1's, so idxs should be unchanged.
        gather_idxs_polymer_ligand = (
            gather_idxs_polymer_ligand * gather_mask_polymer_ligand
        )

        tokens_to_ligand_ligand_bonds = (
            batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds
        )
        gather_idxs_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_idxs
        gather_mask_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_mask.prod(
            axis=1
        ).to(gather_idxs_ligand_ligand.dtype)[:, None]
        gather_idxs_ligand_ligand = (
            gather_idxs_ligand_ligand * gather_mask_ligand_ligand
        )

        gather_idxs = torch.cat(
            [gather_idxs_polymer_ligand, gather_idxs_ligand_ligand], dim=0 # 使用 torch.cat
        )
        contact_matrix[
            gather_idxs[:, 0], gather_idxs[:, 1]
        ] = 1.0 # 使用 torch 张量赋值

        # Because all the padded index's are 0's.
        contact_matrix[0, 0] = 0.0 # 使用 torch 张量赋值

        bonds_act = self.bond_embedding(
            contact_matrix[:, :, None].to(pair_activations.dtype) # 转换 dtype
        )
        return pair_activations + bonds_act

    def _embed_template_pair(
        self,
        batch: feat_batch.Batch, # 使用 PyTorch 版本的 feat_batch.Batch
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embeds Templates and merges into pair activations in PyTorch."""
        dtype = pair_activations.dtype
        # key, subkey = jax.random.split(key) # Key splitting needs PyTorch equivalent
        templates = batch.templates
        asym_id = batch.token_features.asym_id
        # Construct a mask such that only intra-chain template features are
        # computed, since all templates are for each chain individually.
        multichain_mask = (asym_id[:, None] == asym_id[None, :]).to(dtype) # 转换 dtype

        template_act = self.template_embedding( # 使用 self.template_embedding
            query_embedding=pair_activations,
            templates=templates,
            multichain_mask_2d=multichain_mask,
            padding_mask_2d=pair_mask,
        )
        return pair_activations + template_act

    def _embed_process_msa(
        self,
        msa_batch: features.MSA, # 使用 PyTorch 版本的 features.MSA
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor,
        target_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processes MSA and returns updated pair activations in PyTorch."""
        dtype = pair_activations.dtype
        msa_batch = featurization.shuffle_msa(msa_batch) # 假设 featurization.shuffle_msa 已转换，并接受 PyTorch key
        msa_batch = featurization.truncate_msa_batch(msa_batch, self.config.num_msa) # 假设 featurization.truncate_msa_batch 已转换
        msa_feat = featurization.create_msa_feat(msa_batch).to(dtype) # 假设 featurization.create_msa_feat 已转换, 并转换 dtype

        msa_activations = self.msa_activations(msa_feat) # 使用 self.msa_activations_proj

        msa_activations += self.extra_msa_target_feat(target_feat)[None] # 使用 self.extra_msa_target_feat_proj
        msa_mask = msa_batch.mask.to(dtype) # 转换 msa_mask dtype

        # Evoformer MSA stack.
        evoformer_input = {'msa': msa_activations, 'pair': pair_activations}
        masks = {'msa': msa_mask, 'pair': pair_mask}

        activations = evoformer_input # 初始化 activations
        for layer in self.msa_stack_layers: # 循环调用 msa_stack_layers
            activations = layer(activations=activations, masks=masks) # 调用 EvoformerIteration

        evoformer_output = activations # 获取最终 activations

        return evoformer_output['pair']

    def forward(
        self,
        batch: feat_batch.Batch, # 使用 PyTorch 版本的 feat_batch.Batch
        prev: Dict[str, torch.Tensor],
        target_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        num_residues = target_feat.shape[0]

        assert self.global_config.bfloat16 in {'all', 'none'}
        assert batch.token_features.aatype.shape == (num_residues,)

        # TODO
        dtype = (
            torch.bfloat16 if self.global_config.bfloat16 == 'all' else torch.float32
        )

        # TODO
        # print(jax.tree_map(lambda x:x.dtype if isinstance(x, torch.Tensor) else x.shape, batch))
        # print(target_feat.shape, target_feat.dtype)
        # with utils.float32_context():
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            pair_activations, pair_mask = self._seq_pair_embedding(
                batch.token_features, target_feat
            )

            #   pair_activations += hm.Linear(
            #       pair_activations.shape[-1],
            #       name='prev_embedding',
            #       initializer=self.global_config.final_init,
            #   )(
            #       hm.LayerNorm(name='prev_embedding_layer_norm')(
            #           prev['pair'].astype(pair_activations.dtype)
            #       )
            #   )

            pair_activations += self.prev_embedding(
                self.prev_embedding_layer_norm(
                    prev['pair'].to(pair_activations.dtype) # 转换 dtype
                )
            )
            
            pair_activations = self._relative_encoding(batch, pair_activations)

            pair_activations = self._embed_bonds(
                batch=batch, pair_activations=pair_activations
            )
            # start_tmp = time.time()
            pair_activations = self._embed_template_pair(
                batch=batch,
                pair_activations=pair_activations,
                pair_mask=pair_mask,
            )
            # print(f"Template embedding took {(time.time() - start_tmp):.5f}")
            # start_msa = time.time()
            pair_activations = self._embed_process_msa(
                msa_batch=batch.msa,
                pair_activations=pair_activations,
                pair_mask=pair_mask,
                target_feat=target_feat,
            )
            # print(f"MSA embedding took {(time.time() - start_msa):.5f}")

            single_activations = self.single_activations(target_feat)
            single_activations += self.prev_single_embedding(
                self.prev_single_embedding_layer_norm(
                    prev['single']
                )
            )


            # def pairformer_fn(x):
            #     pairformer_iteration = modules.PairFormerIteration(
            #         self.config.pairformer,
            #         self.global_config,
            #         with_single=True,
            #         name='trunk_pairformer',
            #     )
            #     pair_act, single_act = x
            #     return pairformer_iteration(
            #         act=pair_act,
            #         single_act=single_act,
            #         pair_mask=pair_mask,
            #         seq_mask=batch.token_features.mask.astype(dtype),
            #     )

            # pairformer_stack = hk.experimental.layer_stack(
            #     self.config.pairformer.num_layer
            # )(pairformer_fn)

            # pair_activations, single_activations = pairformer_stack(
            #     (pair_activations, single_activations)
            # )

            # start_pair = time.time()
            for i in range(self.pairformer_layers):
                pair_activations, single_activations = self.pairformer_stack_layers[i](
                    act=pair_activations,
                    single_act=single_activations,
                    pair_mask=pair_mask,
                    seq_mask=batch.token_features.mask.to(dtype),
                )
            # print(f"PairFormer took {(time.time() - start_pair):.5f}")


            assert pair_activations.shape == (
                num_residues,
                num_residues,
                self.config.pair_channel,
            )
            assert single_activations.shape == (num_residues, self.config.seq_channel)
            assert len(target_feat.shape) == 2
            assert target_feat.shape[0] == num_residues
            output = {
                'single': single_activations,
                'pair': pair_activations,
                'target_feat': target_feat,
            }

        return output
if __name__ == "__main__":
    from alphafold3.model.pipeline.pipeline import get_null_batch
    batch = get_null_batch(38)

    # print(batch)