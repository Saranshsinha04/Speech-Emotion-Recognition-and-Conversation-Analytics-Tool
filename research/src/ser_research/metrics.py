"""Evaluation metrics, calibration, temperature scaling, and paired cluster bootstrap."""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    y_pred = np.argmax(probs, axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    uar = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    labels = list(range(len(class_names)))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(np.float64) / np.maximum(1, cm.sum(axis=1, keepdims=True))

    ece, brier, bin_acc, bin_conf = compute_calibration(y_true, probs, len(class_names))

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(sup[i]),
        }

    return {
        "accuracy": round(acc, 4),
        "uar": round(uar, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "ece": round(ece, 4),
        "brier": round(brier, 4),
        "reliability_bins": {
            "accuracy": [round(float(x), 4) for x in bin_acc],
            "confidence": [round(float(x), 4) for x in bin_conf],
        },
        "per_class": per_class,
        "confusion_matrix_counts": cm.tolist(),
        "confusion_matrix_norm": [[round(float(v), 4) for v in row] for row in cm_norm],
    }


def compute_calibration(
    y_true: np.ndarray, probs: np.ndarray, num_classes: int, n_bins: int = 15
) -> tuple[float, float, list[float], list[float]]:
    conf = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    correct = (preds == y_true).astype(np.float64)

    onehot = np.eye(num_classes, dtype=np.float64)[y_true]
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_acc, bin_conf = [], []
    n = max(1, len(y_true))
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            acc_i = float(np.mean(correct[mask]))
            conf_i = float(np.mean(conf[mask]))
            ece += (cnt / n) * abs(acc_i - conf_i)
            bin_acc.append(acc_i)
            bin_conf.append(conf_i)
        else:
            mid = 0.5 * (lo + hi)
            bin_acc.append(mid)
            bin_conf.append(mid)
    return ece, brier, bin_acc, bin_conf


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / max(1e-3, temperature)
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def fit_temperature_scaling(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    def nll(t: float) -> float:
        p = softmax_np(val_logits, temperature=t)
        p_true = np.clip(p[np.arange(len(val_labels)), val_labels], 1e-12, 1.0)
        return float(-np.mean(np.log(p_true)))

    res = minimize_scalar(nll, bounds=(0.25, 5.0), method="bounded")
    return float(res.x)


def cluster_bootstrap_diff(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    cluster_ids: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired cluster bootstrap CI for metric(B) - metric(A) resampling clusters (speakers/dialogs)."""
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(cluster_ids)
    cluster_to_idx = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}
    obs = metric_fn(y_true, preds_b) - metric_fn(y_true, preds_a)
    diffs = np.empty(n_boot, dtype=np.float64)
    n_c = len(unique_clusters)
    for b in range(n_boot):
        sampled = rng.choice(unique_clusters, size=n_c, replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in sampled])
        diffs[b] = metric_fn(y_true[idx], preds_b[idx]) - metric_fn(y_true[idx], preds_a[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff": round(float(obs), 4),
        "ci95_low": round(float(lo), 4),
        "ci95_high": round(float(hi), 4),
        "significant_gt_0": bool(lo > 0.0),
    }


def binary_ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {"auroc": 0.5, "auprc": float(np.mean(y_true))}
    return {
        "auroc": round(float(roc_auc_score(y_true, scores)), 4),
        "auprc": round(float(average_precision_score(y_true, scores)), 4),
    }


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(1e-12, p.sum())
    q = q / max(1e-12, q.sum())
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 1e-12
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
