"""Evaluation metrics and plots.

For a heavily imbalanced problem like fraud, PR-AUC (average precision) is the
headline metric — ROC-AUC can look deceptively high when the negative class
dominates. We report both, plus precision / recall / F1 at a chosen operating
threshold.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def threshold_free_metrics(y_true, y_score) -> dict:
    """Metrics that do not depend on a decision threshold."""
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def metrics_at_threshold(y_true, y_score, threshold: float) -> dict:
    """Precision / recall / F1 / confusion counts at a fixed threshold."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def full_report(y_true, y_score, threshold: float) -> dict:
    out = threshold_free_metrics(y_true, y_score)
    out.update(metrics_at_threshold(y_true, y_score, threshold))
    return out


def pr_curve(y_true, y_score):
    precision, recall, thr = precision_recall_curve(y_true, y_score)
    return precision, recall, thr


def roc_curve_pts(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return fpr, tpr, thr
