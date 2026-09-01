import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from ecg_classifier.evaluation import leakage_safe_cv, per_class_sensitivity_specificity


class EvaluationTest(unittest.TestCase):
    def test_class_metrics(self):
        result = per_class_sensitivity_specificity([0, 0, 1, 1], [0, 1, 1, 1], [0, 1])
        self.assertAlmostEqual(result[0]["sensitivity"], 0.5)
        self.assertAlmostEqual(result[1]["specificity"], 0.5)

    def test_cv(self):
        x = np.array([[i, i % 3] for i in range(30)], dtype=float)
        y = np.array([0] * 15 + [1] * 15)
        scores = leakage_safe_cv(LogisticRegression(), x, y, folds=3)
        self.assertEqual(scores.shape, (3,))


if __name__ == "__main__":
    unittest.main()
