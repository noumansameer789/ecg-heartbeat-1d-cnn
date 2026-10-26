import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from ecg_classifier.evaluation import leakage_safe_cv, per_class_sensitivity_specificity, preprocessing_safe_cv, validation_splits


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

    def test_patient_groups_never_cross_folds(self):
        groups = np.repeat(np.arange(12), 4)
        y = np.tile([0, 1, 0, 1], 12)
        x = np.column_stack([np.arange(48), y])
        seen = []
        for train, valid in validation_splits(x, y, folds=3, groups=groups):
            self.assertFalse(set(groups[train]) & set(groups[valid]))
            seen.extend(valid)
        self.assertEqual(sorted(seen), list(range(48)))
        scores = preprocessing_safe_cv(LogisticRegression(), x, y, folds=3, groups=groups)
        self.assertEqual(scores.shape, (3,))

    def test_invalid_group_mapping_is_rejected(self):
        x = np.arange(24).reshape(12, 2)
        y = np.tile([0, 1], 6)
        with self.assertRaises(ValueError):
            validation_splits(x, y, folds=3, groups=[1, 2])
        with self.assertRaises(ValueError):
            validation_splits(x, y, folds=3, groups=np.ones(12))

    def test_scaler_is_fitted_on_each_training_fold(self):
        from unittest.mock import patch
        from sklearn.preprocessing import StandardScaler
        fitted_means = []

        class RecordingScaler(StandardScaler):
            def fit(self, X, y=None, sample_weight=None):
                result = super().fit(X, y, sample_weight=sample_weight)
                fitted_means.append(self.mean_.copy())
                return result

        x = np.column_stack([np.arange(30, dtype=float) ** 3, np.arange(30)])
        y = np.tile([0, 1], 15)
        expected = [x[t].mean(axis=0) for t, _ in validation_splits(x, y, folds=3)]
        with patch("ecg_classifier.evaluation.StandardScaler", RecordingScaler):
            preprocessing_safe_cv(LogisticRegression(), x, y, folds=3)
        np.testing.assert_allclose(fitted_means, expected)


if __name__ == "__main__":
    unittest.main()

