"""Leakage-safe evaluation helpers for imbalanced heartbeat classification."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def per_class_sensitivity_specificity(y_true, y_pred, labels) -> dict[int, dict[str, float]]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    total = matrix.sum()
    result = {}
    for index, label in enumerate(labels):
        tp = matrix[index, index]
        fn = matrix[index, :].sum() - tp
        fp = matrix[:, index].sum() - tp
        tn = total - tp - fn - fp
        result[int(label)] = {
            "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        }
    return result


def leakage_safe_cv(estimator, x, y, folds: int = 5, seed: int = 42) -> np.ndarray:
    """Fit StandardScaler inside every fold instead of before splitting."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for train, valid in splitter.split(x, y):
        pipeline = Pipeline([("scale", StandardScaler()), ("model", clone(estimator))])
        pipeline.fit(x[train], y[train])
        scores.append(f1_score(y[valid], pipeline.predict(x[valid]), average="macro"))
    return np.asarray(scores)


def macro_roc_auc(y_true, probabilities) -> float:
    return float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))
