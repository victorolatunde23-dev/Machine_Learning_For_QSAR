"""Running the grouped evaluation and aggregating it three ways.

Read **pooled** first: every compound scored exactly once, by a model that never saw its
chemical family. That is the number to quote.

**Per-fold** says which families are reachable, not how well the model works -- fold
variance on this data was larger than the differences between models, and one family
(59 compounds at mean Tanimoto distance 0.341) defeated most of the zoo while another
scored 0.92.

**Macro** is the mean and spread across folds. It is a description of the spread, not a
better estimate than pooled.

The **random-split reference** is run alongside on purpose: the gap between it and pooled
is the analogue leakage, and quantifying it is the single most useful output of the whole
evaluation. On this dataset it ran from -0.061 (LightGBM) to -0.133 (naive Bayes), and the
largest gap belonged to nearest-neighbour lookup, which is the mechanism made visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .._optional import tqdm
from ..models.registry import fit_model, predict_proba
from .metrics import classification_metrics, safe_roc_auc
from .splits import FoldSet

__all__ = ["LocoResult", "run_loco", "random_split_reference", "leakage_table"]


@dataclass
class LocoResult:
    """Out-of-fold predictions and the three aggregations of them."""

    out_of_fold: Dict[str, np.ndarray] = field(default_factory=dict)
    per_fold: pd.DataFrame = field(default_factory=pd.DataFrame)
    covered: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    skipped_folds: List[str] = field(default_factory=list)
    unreported_folds: List[str] = field(default_factory=list)

    @property
    def models(self) -> List[str]:
        return list(self.out_of_fold)

    def pooled(
        self,
        *,
        threshold: float = 0.5,
        enrichment_fractions: Sequence[float] = (0.01, 0.05, 0.10),
        precision_ns: Sequence[int] = (50, 100, 200),
    ) -> pd.DataFrame:
        """Every compound scored once, concatenated and scored in one go. Quote this."""
        rows = []
        for name, scores in self.out_of_fold.items():
            keep = self.covered & np.isfinite(scores)
            row = classification_metrics(
                self.y[keep], scores[keep], threshold,
                enrichment_fractions=enrichment_fractions, precision_ns=precision_ns,
            )
            row["model"] = name
            row["coverage"] = int(keep.sum())
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("model")

    def macro(self) -> pd.DataFrame:
        """Mean and standard deviation across folds. A spread, not an estimate."""
        if self.per_fold.empty:
            return pd.DataFrame()
        numeric = self.per_fold.select_dtypes(include=[np.number])
        grouped = numeric.groupby(level="model")
        return grouped.agg(["mean", "std"]).round(4)


def run_loco(
    folds: FoldSet,
    X,
    y,
    *,
    models: Sequence[str],
    params: Dict[str, Dict[str, Any]],
    fps=None,
    sample_weight=None,
    min_fold_n: int = 10,
    random_state: int = 42,
    balanced: bool = True,
    enrichment_fractions: Sequence[float] = (0.01, 0.05, 0.10),
    precision_ns: Sequence[int] = (50, 100, 200),
    verbose: bool = True,
) -> LocoResult:
    """Fit each model on each fold's complement and predict the held-out fold.

    Folds smaller than ``min_fold_n`` still contribute to the pooled estimate; they just
    do not get a per-fold row. An earlier version reported a 5-compound fold whose metrics
    ranged 0.500-0.917 across models and meant nothing, which is noise presented as a
    result.
    """
    y = np.asarray(y, dtype=float)
    labelled = np.isfinite(y)
    y_int = np.where(labelled, y, 0).astype(int)

    oof: Dict[str, np.ndarray] = {m: np.full(len(y), np.nan) for m in models}
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    unreported: List[str] = []

    it = tqdm(list(folds), desc="Leave-one-cluster-out", unit="fold") if verbose else folds
    for fold in it:
        test = fold.mask & labelled
        train = (~fold.mask) & labelled
        if test.sum() == 0:
            continue
        if len(np.unique(y_int[train])) < 2:
            skipped.append(fold.name)
            if verbose:
                print(f"  {fold.name}: training side is single-class, skipped")
            continue

        tr_idx = np.flatnonzero(train)
        te_idx = np.flatnonzero(test)
        f_tr = None if fps is None else [fps[i] for i in tr_idx]
        f_te = None if fps is None else [fps[i] for i in te_idx]
        sw = None if sample_weight is None else np.asarray(sample_weight)[tr_idx]

        for name in models:
            try:
                model = fit_model(
                    name, X[tr_idx], y_int[tr_idx], params.get(name, {}), fps=f_tr,
                    sample_weight=sw, seed=random_state, balanced=balanced,
                )
                p = predict_proba(name, model, X[te_idx], fps=f_te)[:, 1]
            except Exception as exc:  # a member that cannot be fitted here is not fatal
                if verbose:
                    print(f"  {fold.name} / {name}: {type(exc).__name__}: {exc}")
                continue
            oof[name][te_idx] = p

            if int(test.sum()) >= min_fold_n:
                row = classification_metrics(
                    y_int[te_idx], p,
                    enrichment_fractions=enrichment_fractions, precision_ns=precision_ns,
                )
                row.update(
                    {
                        "fold": fold.name,
                        "model": name,
                        "mean_distance_to_train": fold.mean_distance_to_train,
                    }
                )
                rows.append(row)
            elif fold.name not in unreported:
                unreported.append(fold.name)

    per_fold = (
        pd.DataFrame(rows).set_index(["fold", "model"]).sort_index()
        if rows
        else pd.DataFrame()
    )
    first = models[0] if models else None
    covered = (
        np.isfinite(oof[first]) & labelled if first else np.zeros(len(y), dtype=bool)
    )

    if verbose and unreported:
        print(
            f"\n{len(unreported)} fold(s) below min_fold_n={min_fold_n} contribute to the "
            f"pooled estimate but get no per-fold row: {unreported}"
        )
    return LocoResult(
        out_of_fold=oof,
        per_fold=per_fold,
        covered=covered,
        y=y_int,
        skipped_folds=skipped,
        unreported_folds=unreported,
    )


def random_split_reference(
    X,
    y,
    *,
    models: Sequence[str],
    params: Dict[str, Dict[str, Any]],
    fps=None,
    repeats: int = 5,
    random_state: int = 42,
    balanced: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Stratified random-split ROC-AUC, as the leakage comparator.

    This is deliberately *not* presented as a performance estimate. It exists so the gap
    to the pooled grouped estimate can be quantified.
    """
    y = np.asarray(y, dtype=float)
    labelled = np.isfinite(y)
    yl = y[labelled].astype(int)
    Xl = X[labelled]
    fpsl = None if fps is None else [fps[i] for i in np.flatnonzero(labelled)]

    n_splits = max(2, min(repeats, int(np.bincount(yl).min())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores: Dict[str, List[float]] = {m: [] for m in models}

    splits = list(skf.split(Xl, yl))
    it = tqdm(splits, desc="Random-split reference", unit="fold") if verbose else splits
    for tr, te in it:
        f_tr = None if fpsl is None else [fpsl[i] for i in tr]
        f_te = None if fpsl is None else [fpsl[i] for i in te]
        for name in models:
            try:
                m = fit_model(
                    name, Xl[tr], yl[tr], params.get(name, {}), fps=f_tr,
                    seed=random_state, balanced=balanced,
                )
                p = predict_proba(name, m, Xl[te], fps=f_te)[:, 1]
            except Exception:
                continue
            scores[name].append(safe_roc_auc(yl[te], p))

    rows = []
    for name, vals in scores.items():
        clean = [v for v in vals if np.isfinite(v)]
        rows.append(
            {
                "model": name,
                "roc_auc_mean": float(np.mean(clean)) if clean else float("nan"),
                "roc_auc_std": float(np.std(clean)) if clean else float("nan"),
                "n_splits": len(clean),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def leakage_table(pooled: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Random-split AUC against pooled grouped AUC, with the gap.

    The gap is the analogue leakage: how much of a random-split score was the model
    recognising relatives rather than generalising.
    """
    rows = []
    for name in pooled.index:
        if name not in reference.index:
            continue
        rnd = float(reference.loc[name, "roc_auc_mean"])
        pld = float(pooled.loc[name, "roc_auc"])
        rows.append(
            {
                "model": name,
                "random_split": round(rnd, 3),
                "pooled_grouped": round(pld, 3),
                "leakage_gap": round(pld - rnd, 3),
            }
        )
    return (
        pd.DataFrame(rows).set_index("model").sort_values("pooled_grouped", ascending=False)
        if rows
        else pd.DataFrame()
    )
