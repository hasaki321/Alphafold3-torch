import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Final
from alphafold3.common import base_config
from alphafold3.model.components import torch_modules as tm

_CONTACT_THRESHOLD: Final[float] = 8.0
_CONTACT_EPSILON: Final[float] = 1e-3


class DistogramHead(nn.Module):
    """Distogram head."""

    class Config(base_config.BaseConfig):
        first_break: float = 2.3125
        last_break: float = 21.6875
        num_bins: int = 64

    def __init__(
        self,
        config: Config,
        global_config: Dict,
        name: str = 'distogram_head',
    ):
        super(DistogramHead, self).__init__()
        self.config = config
        self.global_config = global_config

        # Linear layer for half logits
        self.half_logits = nn.Linear(128,64,bias=False)
        # self.left_half_logits = tm.Linear(
        #     self.config.num_bins,
        #     initializer=self.global_config.final_init,
        #     name='half_logits',
        # )
    def forward(
        self,
        batch: Dict,
        embeddings: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for DistogramHead.

        Args:
            batch: Dictionary containing token features and masks.
            embeddings: Dictionary containing pair activations.

        Returns:
            Dictionary containing bin edges and contact probabilities.
        """
        pair_act = embeddings['pair']
        # seq_mask = torch.from_numpy(batch.token_features.mask).to(embeddings['pair'].device)
        seq_mask = batch.token_features.mask.to(embeddings['pair'].device)
        pair_mask = seq_mask[:, None] * seq_mask[None, :]

        # Compute half logits
        left_half_logits = self.half_logits(pair_act)
        right_half_logits = left_half_logits
        logits = left_half_logits + right_half_logits.transpose(-2, -3)

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)

        # Compute bin edges
        breaks = torch.linspace(
            self.config.first_break,
            self.config.last_break,
            self.config.num_bins - 1,
            device=pair_act.device,
        )

        # Compute bin tops
        bin_tops = torch.cat([breaks, breaks[-1:] + (breaks[-1] - breaks[-2])], dim=0)

        # Compute contact probabilities
        threshold = _CONTACT_THRESHOLD + _CONTACT_EPSILON
        is_contact_bin = (bin_tops <= threshold).float()
        contact_probs = torch.einsum('ijk,k->ij', probs, is_contact_bin)
        contact_probs = pair_mask * contact_probs

        return {
            'bin_edges': breaks,
            'contact_probs': contact_probs,
        }

from alphafold3.model.pipeline__ import get_null_batch

def create_sample_batch(seq_len):
    """Generates a sample batch for testing."""
    return get_null_batch(seq_len)

def create_sample_embeddings(seq_len, pair_dim):
    """Generates sample embeddings with pair activations."""
    return {
        'pair': torch.randn(seq_len, seq_len, pair_dim),
    }


def distogram_head():
    """Fixture to create a DistogramHead instance."""
    config = DistogramHead.Config(first_break=2.3125, last_break=21.6875, num_bins=64)
    global_config = {'final_init': None}  # Placeholder for global config
    return DistogramHead(config, global_config)


def test_distogram_head_forward(distogram_head):
    """Tests the forward pass of the DistogramHead."""
    seq_len = 40
    pair_dim = 64

    # Create sample data
    batch = create_sample_batch(seq_len)
    embeddings = create_sample_embeddings(seq_len, pair_dim)

    # Forward pass
    output = distogram_head(batch, embeddings)

    # Assertions
    assert 'bin_edges' in output, "Output should contain 'bin_edges'."
    assert 'contact_probs' in output, "Output should contain 'contact_probs'."

    # Validate shapes
    assert output['bin_edges'].shape == (distogram_head.config.num_bins - 1,), \
        f"Expected bin_edges shape: ({distogram_head.config.num_bins - 1},)"
    assert output['contact_probs'].shape == (seq_len, seq_len), \
        f"Expected contact_probs shape: ({seq_len}, {seq_len})"

    # Ensure contact probabilities are between 0 and 1
    assert torch.all(output['contact_probs'] >= 0), "Contact probabilities should be >= 0."
    assert torch.all(output['contact_probs'] <= 1), "Contact probabilities should be <= 1."
if __name__ == '__main__':
    test_distogram_head_forward(distogram_head())