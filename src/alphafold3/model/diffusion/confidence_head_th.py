import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from alphafold3.model.atom_layout import atom_layout_th as atom_layout
from alphafold3.model.diffusion import modules_th as modules
from alphafold3.model.diffusion import template_modules_th_jax as template_modules
from alphafold3.model.components import torch_modules as tm
from alphafold3.common import base_config

class ConfidenceHead(nn.Module):
    """Head to predict the distance errors in a prediction."""

    class PAEConfig(base_config.BaseConfig):
        max_error_bin: float = 31.0
        num_bins: int = 64

    class Config(base_config.BaseConfig):
        """Configuration for ConfidenceHead."""

        pairformer: modules.PairFormerIteration.Config = base_config.autocreate(
            single_attention=base_config.autocreate(),
            single_transition=base_config.autocreate(),
            num_layer=4,
        )
        max_error_bin: float = 31.0
        num_plddt_bins: int = 50
        num_bins: int = 64
        no_embedding_prob: float = 0.2
        pae: 'ConfidenceHead.PAEConfig' = base_config.autocreate()
        dgram_features: template_modules.DistogramFeaturesConfig = (
            base_config.autocreate()
        )
    
    def __init__(
        self,
        config: Config,
        global_config: Dict,
        name: str = 'confidence_head',
    ):
        super(ConfidenceHead, self).__init__()
        self.config = config
        self.global_config = global_config

        # Linear layers for feature embedding
        self.left_target_feat_project = nn.Linear(447, 128,bias=False)
        self.right_target_feat_project = nn.Linear(447, 128,bias=False)
        self.distogram_feat_project = nn.Linear(39, 128, bias=False)

        # PairFormer stack;  in jax: template_embedding_iteration
        self.pairformer_stack = nn.Sequential(
            *[modules.PairFormerIteration(config.pairformer, global_config, 128, 384, with_single=True,name='template_embedding_iteration') for _ in range(config.pairformer.num_layer)]
        )
        # config.num_bins = 128
        # config.pae.num_bins = 128
        # config.num_plddt_bins = 384

        # Linear layers for distance logits
        self.left_distance_logits = nn.Linear(128, config.num_bins,  bias=False)
        self.logits_ln = tm.LayerNorm(128)

        # Linear layers for PAE logits
        self.pae_logits = nn.Linear(128, config.num_bins,bias=False)
        self.pae_logits_ln = tm.LayerNorm(128)

        # Linear layers for pLDDT logits
        # self.plddt_logits = nn.Linear(config.num_bins, config.num_plddt_bins,bias=False)
        self.plddt_logits = nn.Parameter(torch.zeros(384,24,50))
        self.plddt_logits_ln = tm.LayerNorm(384)

        # Linear layers for experimentally resolved logits
        # self.experimentally_resolved_logits = nn.Linear(config.num_bins, 2,bias=False)
        self.experimentally_resolved_logits = nn.Parameter(torch.zeros((384,24,2)))
        self.experimentally_resolved_ln = tm.LayerNorm(384)

    def _embed_features(
        self,
        dense_atom_positions: torch.Tensor,
        token_atoms_to_pseudo_beta: torch.Tensor,
        pair_mask: torch.Tensor,
        pair_act: torch.Tensor,
        target_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Embed features for the confidence head."""
        out = self.left_target_feat_project(target_feat).to(pair_act.dtype)
        out = out + self.right_target_feat_project(target_feat).to(pair_act.dtype)[:, None]
        # out1 = self.right_target_feat_project(target_feat).to(pair_act.dtype)[:, None]
        # print('out.shape',out.shape)
        # print('out1.shape',out1.shape)
        # out = out + out1
        positions = atom_layout.convert(
            token_atoms_to_pseudo_beta,
            dense_atom_positions,
            layout_axes=(-3, -2),
        )
        dgram = template_modules.dgram_from_positions_th(
            positions, self.config.dgram_features
        )
        dgram *= pair_mask[..., None]

        out = out + self.distogram_feat_project(dgram.to(pair_act.dtype))
        return out

    def forward(
        self,
        dense_atom_positions: torch.Tensor,
        embeddings: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
        token_atoms_to_pseudo_beta: torch.Tensor,
        asym_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Builds ConfidenceHead module.
        Arguments:
        dense_atom_positions: [N_res, N_atom, 3] array of positions.
        embeddings: Dictionary of representations.
        seq_mask: Sequence mask.
        token_atoms_to_pseudo_beta: Pseudo beta info for atom tokens.
        asym_id: Asym ID token features.

        Returns:
        Dictionary of results.
        """
        dtype = torch.bfloat16 if self.global_config.bfloat16 == 'all' else torch.float32
        with torch.amp.autocast('cpu',dtype=dtype):
            seq_mask_cast = seq_mask.to(dtype)
            pair_mask = seq_mask_cast[:, None] * seq_mask_cast[None, :]
            pair_mask = pair_mask.to(dtype)

            pair_act = embeddings['pair'].to(dtype)
            single_act = embeddings['single'].to(dtype)
            target_feat = embeddings['target_feat'].to(dtype)

            num_residues = seq_mask.shape[0]
            num_pair_channels = pair_act.shape[2]

            pair_act = pair_act + self._embed_features(
                dense_atom_positions,
                token_atoms_to_pseudo_beta,
                pair_mask,
                pair_act,
                target_feat,
            )
            pair_act = torch.squeeze(pair_act,0)
            # print('pair_act.shape',pair_act.shape)
            # PairFormer stack
            for pairformer_layer in self.pairformer_stack:
               pair_act, single_act =  pairformer_layer(act=pair_act,
                                        single_act=single_act,
                                        pair_mask=pair_mask,
                                        seq_mask=seq_mask,)
            pair_act = pair_act.to(torch.float32)
            assert pair_act.shape == (num_residues, num_residues, num_pair_channels)

            # Distance logits
            left_distance_logits = self.left_distance_logits(self.logits_ln(pair_act))
            right_distance_logits = left_distance_logits
            distance_logits = left_distance_logits + right_distance_logits.transpose(-2, -3)  # Symmetrize

            # Distance breaks and bin centers
            distance_breaks = torch.linspace(
                0.0, self.config.max_error_bin, self.config.num_bins - 1, device=pair_act.device
            )
            step = distance_breaks[1] - distance_breaks[0]
            bin_centers = distance_breaks + step / 2
            bin_centers = torch.cat([bin_centers, bin_centers[-1:] + step], dim=0)

            # Distance probabilities
            distance_probs = F.softmax(distance_logits, dim=-1)
            # print('distance_probs.shape',distance_probs.shape)
            # print('bin_centers.shape',bin_centers.shape)
            # breakpoint()
            pred_distance_error = (distance_probs * bin_centers).sum(dim=-1) * pair_mask
            average_pred_distance_error = pred_distance_error.sum(dim=[-2, -1]) / pair_mask.sum(dim=[-2, -1])

            # PAE outputs
            pae_outputs = {}
            pae_logits = self.pae_logits(self.pae_logits_ln(pair_act))
            pae_breaks = torch.linspace(
                0.0, self.config.pae.max_error_bin, self.config.pae.num_bins - 1, device=pair_act.device
            )
            step = pae_breaks[1] - pae_breaks[0]
            bin_centers = pae_breaks + step / 2
            bin_centers = torch.cat([bin_centers, bin_centers[-1:] + step], dim=0)
            pae_probs = F.softmax(pae_logits, dim=-1)

            seq_mask_bool = seq_mask.to(torch.bool)
            pair_mask_bool = seq_mask_bool[:, None] * seq_mask_bool[None, :]
            pae = (pae_probs * bin_centers).sum(dim=-1) * pair_mask_bool
            pae_outputs.update({
                'full_pae': pae,
            })

        # pTM computation
        tmscore_adjusted_pae_global, tmscore_adjusted_pae_interface = self._get_tmscore_adjusted_pae(
            asym_id=asym_id,
            seq_mask=seq_mask,
            pair_mask=pair_mask_bool,
            bin_centers=bin_centers,
            pae_probs=pae_probs,
        )
        pae_outputs.update({
            'tmscore_adjusted_pae_global': tmscore_adjusted_pae_global,
            'tmscore_adjusted_pae_interface': tmscore_adjusted_pae_interface,
        })
        single_act = single_act.to(torch.float32)

        # print('single_act.shape in confidence head',single_act.shape)    # (37, 384)   (num_res, num_channels)  
        # print('self.plddt_logits_ln(single_act).shape in confidence head',self.plddt_logits_ln(single_act).shape)    # 
        # pLDDT
        # Shape (num_res, num_atom, num_bins)
        #   (num_channels, num_atom, num_bins)  x (num_res, num_channels)   ->   (num_res, num_atom, num_bins)  
        #   (384, 24, 50)
        plddt_logits = torch.einsum('cab,rc->rab',self.plddt_logits,self.plddt_logits_ln(single_act))
        bin_width = 1.0 / self.config.num_plddt_bins
        bin_centers = torch.arange(0.5 * bin_width, 1.0, bin_width, device=plddt_logits.device)
        predicted_lddt = (F.softmax(plddt_logits, dim=-1) * bin_centers).sum(dim=-1) * 100.0

        # Experimentally resolved
        # Shape (num_res, num_atom, 2)
        #   (num_channels, num_atom, 2) (num_res, num_channels)-> (num_res, num_atom, 2)
        #   (384, 24, 2) x  
        experimentally_resolved_logits = torch.einsum('cat,rc->rat',self.experimentally_resolved_logits,self.experimentally_resolved_ln(single_act))
        predicted_experimentally_resolved = F.softmax(experimentally_resolved_logits, dim=-1)[..., 1]

        return {
            'predicted_lddt': predicted_lddt,
            'predicted_experimentally_resolved': predicted_experimentally_resolved,
            'full_pde': pred_distance_error,
            'average_pde': average_pred_distance_error,
            **pae_outputs,
        }

    def _get_tmscore_adjusted_pae(
        self,
        asym_id: torch.Tensor,
        seq_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        bin_centers: torch.Tensor,
        pae_probs: torch.Tensor,
    ):
        def get_tmscore_adjusted_pae(num_interface_tokens, bin_centers, pae_probs):
            # Clip to avoid negative/undefined d0.
            clipped_num_res = torch.clamp(num_interface_tokens, min=19)

            # Compute d_0(num_res) as defined by TM-score
            d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

            # Make compatible with [num_tokens, num_tokens, num_bins]
            d0 = d0[:, :, None]
            bin_centers = bin_centers[None, None, :]

            # TM-Score term for every bin.
            tm_per_bin = 1.0 / (1 + (bin_centers / d0) ** 2)
            # E_distances tm(distance).
            predicted_tm_term = (pae_probs * tm_per_bin).sum(dim=-1)
            return predicted_tm_term

        # Interface version
        x = asym_id[None, :] == asym_id[:, None]
        num_chain_tokens = (x * pair_mask).sum(dim=-1)
        num_interface_tokens = num_chain_tokens[None, :] + num_chain_tokens[:, None]
        # Don't double-count within a single chain
        num_interface_tokens -= x * (num_interface_tokens // 2)
        num_interface_tokens = num_interface_tokens * pair_mask

        num_global_tokens = torch.full_like(pair_mask, seq_mask.sum(), dtype=torch.int32)

        global_apae = get_tmscore_adjusted_pae(num_global_tokens, bin_centers, pae_probs)
        interface_apae = get_tmscore_adjusted_pae(num_interface_tokens, bin_centers, pae_probs)
        return global_apae, interface_apae