"""Small, dependency-free binary-classification metrics (label 1 = fake = positive class)."""

from __future__ import annotations

import math
from typing import Dict, Sequence


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """AUC via the rank-sum formulation, with average ranks for ties. NaN if only one class present."""
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based average rank of the tie group
        rank_sum += avg_rank * sum(1 for k in range(i, j + 1) if pairs[k][1] == 1)
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def classification_report(labels: Sequence[int], scores: Sequence[float], threshold: float = 0.5) -> Dict[str, object]:
    """Binary metrics with class 1 as fake, plus macro scores, error rates, AUC, and confusion."""
    if len(labels) != len(scores) or len(labels) == 0:
        raise ValueError("labels and scores must be non-empty and the same length")
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    precision_real = tn / (tn + fn) if tn + fn else 0.0
    recall_real = tn / (tn + fp) if tn + fp else 0.0
    f1_real = (2 * precision_real * recall_real / (precision_real + recall_real)
               if precision_real + recall_real else 0.0)
    return {
        "n": len(labels),
        "threshold": threshold,
        "accuracy": (tp + tn) / len(labels),
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1,
        "precision_real": precision_real,
        "recall_real": recall_real,
        "f1_real": f1_real,
        "macro_precision": (precision + precision_real) / 2,
        "macro_recall": (recall + recall_real) / 2,
        "macro_f1": (f1 + f1_real) / 2,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "roc_auc": roc_auc(labels, scores),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def is_finite_number(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
