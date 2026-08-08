from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(metric_fn, *args, **kwargs):
    try:
        return float(metric_fn(*args, **kwargs))
    except Exception:
        return float("nan")


def compute_binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    labels = labels.astype(np.int64)
    probabilities = probabilities.astype(np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)

    return {
        "threshold": float(threshold),
        "num_residues": int(labels.size),
        "num_positive": int(labels.sum()),
        "positive_fraction": float(labels.mean()) if labels.size else float("nan"),
        "accuracy": _safe_metric(accuracy_score, labels, predictions),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, labels, predictions),
        "precision": _safe_metric(precision_score, labels, predictions, zero_division=0),
        "recall": _safe_metric(recall_score, labels, predictions, zero_division=0),
        "f1": _safe_metric(f1_score, labels, predictions, zero_division=0),
        "mcc": _safe_metric(matthews_corrcoef, labels, predictions),
        "auroc": _safe_metric(roc_auc_score, labels, probabilities),
        "auprc": _safe_metric(average_precision_score, labels, probabilities),
    }


def select_threshold(labels: np.ndarray, probabilities: np.ndarray, criterion: str = "mcc") -> tuple[float, dict[str, float]]:
    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.05, 0.95, 19, dtype=np.float64),
                probabilities.astype(np.float64),
            ]
        )
    )

    best_threshold = 0.5
    best_metrics = compute_binary_metrics(labels, probabilities, best_threshold)
    best_score = best_metrics.get(criterion, float("-inf"))
    if math.isnan(best_score):
        best_score = float("-inf")

    for threshold in candidate_thresholds:
        metrics = compute_binary_metrics(labels, probabilities, float(threshold))
        score = metrics.get(criterion, float("-inf"))
        if math.isnan(score):
            continue
        if score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score

    return best_threshold, best_metrics

