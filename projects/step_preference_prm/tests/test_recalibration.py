from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

script = Path(__file__).resolve().parents[1] / "scripts/recalibrate_scores.py"
spec = importlib.util.spec_from_file_location("recalibrate_scores", script)
recalibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recalibration)


class RecalibrationTests(unittest.TestCase):
    def test_recovers_known_probabilities(self):
        # At score +1, 8/10 labels are positive; at -1, 2/10 are positive.
        # The finite MLE therefore has slope log(4), bias 0.
        slope, bias = recalibration.fit_platt(
            [1.0] * 10 + [-1.0] * 10,
            [1] * 8 + [0] * 2 + [1] * 2 + [0] * 8,
        )
        self.assertAlmostEqual(slope, math.log(4), places=7)
        self.assertAlmostEqual(bias, 0, places=7)

    def test_rejects_missing_label_class(self):
        with self.assertRaises(ValueError):
            recalibration.fit_platt([0.0, 1.0], [1, 1])


if __name__ == "__main__":
    unittest.main()
