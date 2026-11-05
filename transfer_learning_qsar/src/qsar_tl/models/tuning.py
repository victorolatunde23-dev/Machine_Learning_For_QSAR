"""Hyperparameter search, ranked by the AutoQSAR gap-aware score.

``score = (acc_test - 0.5) * (1 - |acc_train - acc_test|) + 0.5``, after Dixon et al.,
*Future Med. Chem.* 2016. It penalises the train/test gap, which is a reasonable thing to
want on small congeneric datasets.

It also **systematically prefers under-fit models**: on one endpoint it chose a 50-tree,
7-leaf booster over a configuration with a better held-out score. That is a defensible
trade, but it should be a visible one, so :func:`tune_model` reports how much held-out AUC
the gap penalty cost whenever the difference exceeds ``report_cost_above``.

The CV numbers this produces come from a *random* split of a congeneric dataset and carry
the usual analogue leakage. They are for choosing between configurations of the same
model, not for reporting performance -- that is what the grouped evaluation is for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .._optional import tqdm
from ..evaluation.metrics import safe_roc_auc
from .registry import fit_model, get_spec, predict_proba

__all__ = ["autoqsar_score", "TuningResult", "tune_model", "tune_all", "safe_cv_folds"]


def autoqsar_score(acc_train: float, acc_test: float) -> float:
    """Gap-aware score from a train and a test accuracy (or AUC)."""
    return (acc_test - 0.5) * (1.0 - abs(acc_train - acc_test)) + 0.5


def safe_cv_folds(y, requested: int) -> int:
    """Clamp fold count to the smaller class, since StratifiedKFold cannot exceed it.

    A run once died at ``n_splits=10 greater than the number of samples: n_samples=3``
    six steps after the actual problem (an upstream download failure). Clamping turns that
    class of late, opaque failure into a printable warning.
    """
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        # Nothing to stratify. The sanity gate should have stopped the run long before
        # here, so return the minimum rather than pretending a fold count is meaningful.
        return 2
    counts = np.bincount(y)
    smallest = int(counts[counts > 0].min())
    return max(2, min(int(requested), smallest))


@dataclass
class TuningResult:
    """Chosen parameters plus the evidence for the choice."""

    model: str
    params: Dict[str, Any]
    train_auc: float
    test_auc: float
    score: float
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    gap_penalty_cost: float = 0.0
    best_by_test_params: Optional[Dict[str, Any]] = None
    n_folds: int = 0

    def summary(self, report_cost_above: float = 0.005) -> str:
        lines = [
            f"{self.model:<16s} AutoQSAR={self.score:.3f} "
            f"(train {self.train_auc:.3f}, test {self.test_auc:.3f})  {self.params}"
        ]
        if self.gap_penalty_cost > report_cost_above:
            lines.append(
                f"  [gap penalty cost {self.gap_penalty_cost:+.3f} test AUC -- it chose "
                f"the flatter model; {self.best_by_test_params} scored "
                f"{self.test_auc + self.gap_penalty_cost:.3f}]"
            )
        return "\n".join(lines)


def tune_model(
    name: str,
    X,
    y,
    *,
    fps=None,
    grid: Optional[Sequence[Dict[str, Any]]] = None,
    cv_folds: int = 10,
    random_state: int = 42,
    balanced: bool = True,
    progress: bool = True,
) -> TuningResult:
    """Cross-validate every grid point for one model and rank by the gap-aware score."""
    spec = get_spec(name)
    grid = list(grid if grid is not None else spec.grid) or [{}]
    y = np.asarray(y, dtype=int)
    n_folds = safe_cv_folds(y, cv_folds)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    splits = list(skf.split(X, y))

    rows: List[Dict[str, Any]] = []
    it = tqdm(grid, desc=f"  {name}", leave=False) if progress else grid
    for params in it:
        tr_scores, te_scores = [], []
        for tr, te in splits:
            f_tr = None if fps is None else [fps[i] for i in tr]
            f_te = None if fps is None else [fps[i] for i in te]
            try:
                m = fit_model(
                    name, X[tr], y[tr], params, fps=f_tr, seed=random_state,
                    balanced=balanced,
                )
                p_tr = predict_proba(name, m, X[tr], fps=f_tr)[:, 1]
                p_te = predict_proba(name, m, X[te], fps=f_te)[:, 1]
            except Exception:
                continue
            tr_scores.append(safe_roc_auc(y[tr], p_tr))
            te_scores.append(safe_roc_auc(y[te], p_te))
        if not te_scores:
            continue
        a_tr = float(np.nanmean(tr_scores))
        a_te = float(np.nanmean(te_scores))
        rows.append(
            {
                "params": params,
                "train_auc": a_tr,
                "test_auc": a_te,
                "autoqsar": autoqsar_score(a_tr, a_te),
            }
        )

    if not rows:
        raise RuntimeError(
            f"every grid point failed for {name}. Check the feature matrix and that "
            f"{spec.missing or 'its dependencies'} are installed."
        )

    table = pd.DataFrame(rows).sort_values("autoqsar", ascending=False).reset_index(drop=True)
    best = table.iloc[0]
    best_by_test = table.sort_values("test_auc", ascending=False).iloc[0]
    return TuningResult(
        model=name,
        params=dict(best["params"]),
        train_auc=float(best["train_auc"]),
        test_auc=float(best["test_auc"]),
        score=float(best["autoqsar"]),
        table=table,
        gap_penalty_cost=float(best_by_test["test_auc"] - best["test_auc"]),
        best_by_test_params=dict(best_by_test["params"]),
        n_folds=n_folds,
    )


def tune_all(
    models: Sequence[str],
    X,
    y,
    *,
    fps=None,
    fixed: Optional[Dict[str, Dict[str, Any]]] = None,
    retune: bool = True,
    cv_folds: int = 10,
    random_state: int = 42,
    balanced: bool = True,
    skip_unavailable: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Tune each model, or take fixed parameters when ``retune`` is False.

    Returns ``{model: params}``. Unavailable models are skipped with a printed reason when
    ``skip_unavailable``, so a missing LightGBM wheel does not end the run.
    """
    fixed = dict(fixed or {})
    out: Dict[str, Dict[str, Any]] = {}
    results: Dict[str, TuningResult] = {}

    y = np.asarray(y, dtype=int)
    if verbose:
        n_folds = safe_cv_folds(y, cv_folds)
        if n_folds != cv_folds:
            print(
                f"!! tuning folds reduced from {cv_folds} to {n_folds}: the smaller class "
                f"has only {int(np.bincount(y).min())} compounds."
            )
        print(
            f"=== Tuning on {len(y)} labelled compounds "
            f"({int((y == 1).sum())} active), {n_folds}-fold stratified CV ==="
        )

    for name in models:
        spec = get_spec(name)
        if not spec.available:
            msg = f"  {name:<16s} SKIPPED: needs {spec.missing}"
            if skip_unavailable:
                if verbose:
                    print(msg)
                continue
            raise ImportError(msg.strip())
        if not retune:
            out[name] = dict(fixed.get(name, {}))
            if verbose:
                print(f"  {name:<16s} fixed params {out[name]}")
            continue
        res = tune_model(
            name, X, y, fps=fps, cv_folds=cv_folds, random_state=random_state,
            balanced=balanced, progress=verbose,
        )
        results[name] = res
        out[name] = res.params
        if verbose:
            print(res.summary())

    if verbose and results:
        print(
            "\nThese CV numbers come from random splits of a congeneric set and are "
            "optimistic.\nThe grouped (leave-one-cluster-out) estimate is the one to quote."
        )
    return out
