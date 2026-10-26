"""Fold-local preprocessing; optional patient/record-disjoint evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
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


def validation_splits(x, y, folds: int = 5, seed: int = 42, groups=None):
    """Return indices; group IDs must identify the independent sampling unit.

    Without groups this is beat-level stratification, not patient-disjoint
    validation. Supplying record IDs only proves patient separation when each
    patient has exactly one record; callers must establish that mapping.
    """
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x must be 2-D and y must contain one label per row")
    if groups is None:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        return list(splitter.split(x, y))
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != len(y):
        raise ValueError("groups must contain one patient/record ID per row")
    if len(np.unique(groups)) < folds:
        raise ValueError("At least one distinct group per fold is required")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(x, y, groups))


def preprocessing_safe_cv(estimator, x, y, folds: int = 5, seed: int = 42, groups=None) -> np.ndarray:
    """Fit preprocessing on each training fold, with optional group separation."""
    x, y = np.asarray(x), np.asarray(y)
    scores = []
    for train, valid in validation_splits(x, y, folds, seed, groups):
        pipeline = Pipeline([("scale", StandardScaler()), ("model", clone(estimator))])
        pipeline.fit(x[train], y[train])
        scores.append(f1_score(y[valid], pipeline.predict(x[valid]), average="macro"))
    return np.asarray(scores)


def leakage_safe_cv(estimator, x, y, folds: int = 5, seed: int = 42, groups=None) -> np.ndarray:
    """Compatibility alias. Does not establish patient separation without groups."""
    return preprocessing_safe_cv(estimator, x, y, folds, seed, groups)


def macro_roc_auc(y_true, probabilities) -> float:
    return float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))

