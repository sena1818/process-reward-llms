from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prm_pref.utils.metrics import (  # noqa: E402
    binary_classification_metrics,
    first_error_metrics,
    pairwise_metrics,
    problem_cluster_bootstrap,
    select_binary_threshold,
    select_first_error_threshold,
)


class MetricTests(unittest.TestCase):
    def test_binary_and_pairwise_metrics(self) -> None:
        binary = binary_classification_metrics([2.0, -2.0, 1.0, -1.0], [1, 0, 1, 0])
        self.assertEqual(binary["accuracy"], 1.0)
        self.assertEqual(binary["macro_f1"], 1.0)
        self.assertEqual(binary["auroc"], 1.0)
        pairwise = pairwise_metrics([2.0, 0.0], [1.0, 0.0])
        self.assertEqual(pairwise["accuracy"], 0.5)
        self.assertEqual(pairwise["ties"], 1)
        calibration = select_binary_threshold([2.0, -2.0, 1.0, -1.0], [1, 0, 1, 0])
        self.assertEqual(calibration["selection_split"], "validation")
        self.assertEqual(calibration["validation_step"]["macro_f1"], 1.0)

    def test_threshold_is_selected_from_validation_only(self) -> None:
        validation = [
            {
                "ratings": [1, 1, -1],
                "scores": [2.0, 1.0, -2.0],
                "first_error_index": 3,
            },
            {
                "ratings": [1, -1],
                "scores": [1.5, -1.0],
                "first_error_index": 2,
            },
            {
                "ratings": [0, 1],
                "scores": [-100.0, 2.0],
                "first_error_index": None,
            },
        ]
        calibration = select_first_error_threshold(validation)
        self.assertEqual(calibration["selection_split"], "validation")
        self.assertFalse(calibration["neutral_labels_used_for_candidates"])
        self.assertEqual(calibration["validation_first_error"]["within_1"], 1.0)

        test = [
            {
                "ratings": [1, -1],
                "scores": [1.0, -0.5],
                "first_error_index": 2,
            }
        ]
        result = first_error_metrics(test, calibration["threshold"])
        self.assertEqual(result["n_known_first_error"], 1)

    def test_problem_cluster_bootstrap_reports_problem_level_interval(self) -> None:
        records = [
            {"problem_id": "a", "value": 1.0},
            {"problem_id": "a", "value": 1.0},
            {"problem_id": "b", "value": 0.0},
        ]
        interval = problem_cluster_bootstrap(
            records,
            lambda sample: {
                "mean": sum(item["value"] for item in sample) / len(sample)
            },
            num_samples=50,
            seed=9,
        )
        self.assertEqual(interval["n_clusters"], 2)
        self.assertEqual(interval["n_bootstrap_samples"], 50)
        self.assertAlmostEqual(
            interval["metrics"]["mean"]["estimate"],
            2.0 / 3.0,
        )


if __name__ == "__main__":
    unittest.main()
