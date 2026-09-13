from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all reported metrics for one split."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "auc_roc": float("nan"),
        "auc_pr": float("nan"),
        "f1": float("nan"),
        "mcc": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "specificity": float("nan"),
        "threshold": float(threshold),
    }

    if len(np.unique(y_true)) < 2:
        return metrics

    metrics["auc_roc"] = float(roc_auc_score(y_true, y_score))
    metrics["auc_pr"] = float(average_precision_score(y_true, y_score))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    return metrics


def select_threshold(y_true: np.ndarray, y_score: np.ndarray, grid: int = 81) -> float:
    """Pick the decision threshold that maximises F1 on the validation split."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    candidates = np.linspace(0.05, 0.95, grid)
    best_f1, best_t = -1.0, 0.5
    for t in candidates:
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


@torch.no_grad()
def predict(model: torch.nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and return concatenated (scores, labels)."""
    model.eval()
    scores, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(
            batch.x,
            batch.edge_index,
            batch.surface_edge_index,
            batch.batch,
        )
        scores.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels.append(batch.y.detach().cpu().numpy())
    if not scores:
        return np.array([]), np.array([])
    return np.concatenate(scores), np.concatenate(labels)


def mean_confidence_interval(values: Sequence[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Two-sided t-based confidence interval of the mean over repeated seeds."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, 0.0
    sem = float(stats.sem(values))
    if sem == 0 or np.isnan(sem):
        return mean, 0.0
    half = float(stats.t.ppf((1 + confidence) / 2.0, n - 1) * sem)
    return mean, half


def aggregate_seeds(records: List[Dict[str, float]], metrics: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Aggregate per-seed metric dictionaries into mean +/- 95% CI."""
    aggregated = {}
    for name in metrics:
        values = [r[name] for r in records if name in r and not np.isnan(r.get(name, float("nan")))]
        if not values:
            continue
        mean, half = mean_confidence_interval(values)
        aggregated[name] = {"mean": mean, "ci95": half, "n": len(values)}
    return aggregated


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Two-tailed paired t-test between two per-seed metric vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return {"t": float("nan"), "p": float("nan")}
    t_stat, p_value = stats.ttest_rel(a, b)
    return {"t": float(t_stat), "p": float(p_value)}


def format_metrics(metrics: Dict[str, float], digits: int = 4) -> str:
    return "  ".join(f"{k}={v:.{digits}f}" for k, v in metrics.items() if isinstance(v, float))
