from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import f1_score


def tune_per_label_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    label_order: Sequence[str],
    start: float = 0.1,
    end: float = 0.9,
    step: float = 0.05,
) -> Dict[str, float]:
    thresholds = np.arange(start, end + 1e-8, step)
    best_thresholds: Dict[str, float] = {}

    for i, label in enumerate(label_order):
        y_true = labels[:, i].astype(np.int32)
        y_prob = probabilities[:, i]

        best_f1 = -1.0
        best_threshold = 0.5
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(np.int32)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)

        best_thresholds[label] = best_threshold

    return best_thresholds


def thresholds_to_list(
    best_thresholds: Dict[str, float], label_order: Sequence[str]
) -> list[float]:
    return [float(best_thresholds[label]) for label in label_order]
