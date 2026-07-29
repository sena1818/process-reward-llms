from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the local audit environment")
class NodeLossTests(unittest.TestCase):
    def test_node_balancing_is_independent_of_candidate_count(self) -> None:
        from prm_pref.training.losses import (
            nodewise_pairwise_loss,
            nodewise_pointwise_loss,
        )

        logits = torch.zeros(5)
        labels = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
        node_indices = torch.tensor([0, 0, 1, 1, 1])
        node_offsets = torch.tensor([0, 2, 5])
        point = nodewise_pointwise_loss(
            logits,
            labels,
            node_indices,
            num_nodes=2,
        )
        pair = nodewise_pairwise_loss(logits, labels, node_offsets)
        expected = torch.log(torch.tensor(2.0))
        self.assertTrue(torch.allclose(point, expected))
        self.assertTrue(torch.allclose(pair, expected))


if __name__ == "__main__":
    unittest.main()
