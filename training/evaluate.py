"""Evaluation framework — metrics are defined BEFORE any training.

Expression: accuracy, balanced accuracy, macro precision/recall/F1,
            confusion matrix, per-class report, expected calibration error.
Retrieval:  precision@K / recall@K (same-label neighbours).
Morphology: PCA explained variance, silhouette score, cluster stability.
Robustness: performance stratified by pose / quality bins.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, silhouette_score)


def expression_metrics(y_true, y_pred, y_prob=None, labels=None) -> dict:
    labels = labels or sorted(set(y_true) | set(y_pred))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    out = {
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(p, 4), "macro_recall": round(r, 4), "macro_f1": round(f, 4),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }
    if y_prob is not None:
        out["expected_calibration_error"] = round(expected_calibration_error(np.asarray(y_prob), np.asarray(y_true) == np.asarray(y_pred)), 4)
    return out


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confidences > lo) & (confidences <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - confidences[m].mean())
    return float(ece)


def retrieval_metrics(embeddings: np.ndarray, labels: list, ks=(1, 5, 10)) -> dict:
    E = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    labels = np.asarray(labels)
    out = {}
    for k in ks:
        idx = np.argsort(-S, axis=1)[:, :k]
        hits = (labels[idx] == labels[:, None])
        out[f"precision@{k}"] = round(float(hits.mean()), 4)
        totals = np.array([(labels == l).sum() - 1 for l in labels])
        rec = np.where(totals > 0, hits.sum(1) / np.maximum(totals, 1), np.nan)
        out[f"recall@{k}"] = round(float(np.nanmean(rec)), 4)
    return out


def morphology_metrics(pca_scores: np.ndarray, cluster_labels: np.ndarray, explained: np.ndarray, group_labels=None) -> dict:
    out = {"pca_explained_variance": [round(float(v), 4) for v in explained],
           "pca_cumulative_2": round(float(explained[:2].sum()), 4) if len(explained) >= 2 else None}
    if len(set(cluster_labels)) > 1 and len(cluster_labels) > len(set(cluster_labels)):
        out["silhouette_clusters"] = round(float(silhouette_score(pca_scores, cluster_labels)), 4)
    if group_labels is not None and len(set(group_labels)) > 1:
        out["silhouette_groups"] = round(float(silhouette_score(pca_scores, group_labels)), 4)
    return out


def stratified_performance(records: list[dict], y_true, y_pred) -> dict:
    """Accuracy per pose / quality bin — reveals where the model degrades."""
    bins = defaultdict(list)
    for r, t, p in zip(records, y_true, y_pred):
        yaw = abs((r.get("pose") or {}).get("yaw_deg", 0))
        q = r.get("quality_score", 0)
        bins[f"yaw<{15 if yaw < 15 else 30 if yaw < 30 else 90}"].append(t == p)
        bins[f"quality>={40 if q < 70 else 70}"].append(t == p)
    return {k: {"n": len(v), "accuracy": round(float(np.mean(v)), 4)} for k, v in sorted(bins.items())}


def save(metrics: dict, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
