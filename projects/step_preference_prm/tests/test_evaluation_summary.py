from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prm_pref.eval.evaluator import build_evaluation_summary  # noqa: E402


def _result(*, run_name: str, seed: int) -> dict:
    return {
        "run_name": run_name,
        "mode": "hybrid",
        "backend": "causal_lora",
        "model_name": "test-model",
        "max_length": 2048,
        "lambda_pair": 0.3,
        "seed": seed,
        "train_nodes": 10,
        "checkpoint": "last",
        "checkpoint_epoch": 1,
        "validation": {
            "step": {"macro_f1": 0.7},
            "pairwise": {"accuracy": 0.8},
        },
        "test": {
            "step": {
                "macro_f1": 0.69,
                "auroc": 0.8,
                "brier": 0.2,
                "ece_10": 0.03,
            },
            "pairwise": {"accuracy": 0.81},
        },
    }


class EvaluationSummaryTests(unittest.TestCase):
    def test_report_all_supports_first_error_free_v1_metrics(self) -> None:
        summary = build_evaluation_summary(
            [_result(run_name="hybrid_seed42", seed=42)],
            selection_rule="report_all",
        )
        self.assertIsNone(summary["selected_hybrid_lambda"])
        self.assertEqual(summary["replicate_aggregates"][0]["n_runs"], 1)
        self.assertIsNone(
            summary["replicate_aggregates"][0]["metrics"]
            ["test_first_error_exact"]["mean"]
        )


if __name__ == "__main__":
    unittest.main()
