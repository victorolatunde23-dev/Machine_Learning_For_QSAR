"""Classification metrics, fixed before any model is fitted.

Screening metrics (enrichment factor, precision@N) sit alongside the usual ones because a
library screen is a ranking problem: ROC-AUC describes the whole ordering, but what gets
acted on is the top of it, and the two can disagree. On the PSMA pKi set no model cleared
the linear floor on precision@50 even though several cleared it on ROC-AUC -- which is the
wrong way round for a shortlist, and only visible because both are reported.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = [
    "POSITIVE_CLASS",
    "safe_roc_auc",
    "enrichment_factor",
    "precision_at_n",
    "expected_calibration_error",
    "classification_metrics",
    "blend_equal",
    "DEFAULT_ENRICHMENT_FRACTIONS",
    "DEFAULT_PRECISION_AT_N",
]

POSITIVE_CLASS = 1
DEFAULT_ENRICHMENT_FRACTIONS: Sequence[float] = (0.01, 0.05, 0.10)
DEFAULT_PRECISION_AT_N: Sequence[int] = (50, 100, 200)


def safe_roc_auc(y_true, scores) -> float:
    """ROC-AUC, or NaN on a single-class input.

    Leave-one-cluster-out produces single-class folds routinely, and a metric that raises
    there turns a reportable gap in coverage into a crashed run.
    """
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def enrichment_factor(y_true, scores, fraction: float) -> float:
    """Hit rate in the top ``fraction`` divided by the overall hit rate.

    EF = 1 is a worthless ranking. Beware small folds: at n = 14 the top 1% is one
    compound, so the value can only be 0 or 1/base_rate.
    """
    y_true = np.asarray(y_true)
    if len(y_true) == 0:
        return float("nan")
    base = float((y_true == POSITIVE_CLASS).mean())
    if base == 0:
        return float("nan")
    k = max(1, int(round(fraction * len(y_true))))
    top = np.argsort(-np.asarray(scores, dtype=float))[:k]
    return float((y_true[top] == POSITIVE_CLASS).mean() / base)


def precision_at_n(y_true, scores, n: int) -> float:
    """Fraction of the top ``n`` that are actually active."""
    y_true = np.asarray(y_true)
    n = int(min(n, len(y_true)))
    if n == 0:
        return float("nan")
    top = np.argsort(-np.asarray(scores, dtype=float))[:n]
    return float((y_true[top] == POSITIVE_CLASS).mean())


def expected_calibration_error(y_true, probabilities, n_bins: int = 10) -> float:
    """Weighted mean gap between predicted and observed frequency over equal-width bins."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    n = len(p)
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(y_true[m].mean() - p[m].mean())
    return float(ece)


def classification_metrics(
    y_true,
    probabilities,
    threshold: float = 0.5,
    *,
    enrichment_fractions: Sequence[float] = DEFAULT_ENRICHMENT_FRACTIONS,
    precision_ns: Sequence[int] = DEFAULT_PRECISION_AT_N,
) -> Dict[str, float]:
    """One row of metrics for a (fold, model) pair.

    Returns NaN rather than raising on a single-class input, for the reason in
    :func:`safe_roc_auc`.
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(probabilities, dtype=float)
    y_pred = (p >= threshold).astype(int)
    multiclass = len(np.unique(y_true)) > 1
    n = len(y_true)

    out: Dict[str, float] = {
        "n": int(n),
        "n_active": int((y_true == POSITIVE_CLASS).sum()),
        "base_rate": float((y_true == POSITIVE_CLASS).mean()) if n else float("nan"),
        "roc_auc": safe_roc_auc(y_true, p),
        "pr_auc": float(average_precision_score(y_true, p)) if multiclass else float("nan"),
        "balanced_acc": (
            float(balanced_accuracy_score(y_true, y_pred)) if multiclass else float("nan")
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if multiclass else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": (
            float(brier_score_loss(y_true, np.clip(p, 0, 1))) if n else float("nan")
        ),
        "ECE": expected_calibration_error(y_true, np.clip(p, 0, 1)) if n else float("nan"),
    }
    for frac in enrichment_fractions:
        out[f"EF@{int(frac * 100)}%"] = enrichment_factor(y_true, p, frac)
    for k in precision_ns:
        out[f"precision@{k}"] = precision_at_n(y_true, p, k)
    return out


def blend_equal(probability_vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Equal-weight average of P(active) vectors.

    Equal weighting, not fitted weights, on evidence: simplex weights fitted on the test
    set lost to a plain average in all six scheme x endpoint comparisons tried, by up to
    0.08 R2 under grouped splits -- and they were fitted on the very data they were scored
    against. Note also that the blend failed to beat its best single member on seven
    consecutive endpoints, so always report it against the members
    (:func:`qsar_tl.evaluation.floors.blend_vs_best_member`).
    """
    if not probability_vectors:
        raise ValueError("nothing to blend")
    stack = np.stack([np.asarray(p, dtype=float) for p in probability_vectors], axis=0)
    return stack.mean(axis=0)
