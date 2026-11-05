"""Choosing the active/inactive cut.

A classifier emits a probability; calling something "active" needs a threshold. 0.5 is
near-optimal only when the base rate is 0.5 *and* the costs are symmetric, so it is an
assumption rather than a choice.

Two rules here, both from failures:

1. **Choose on out-of-fold predictions.** Picking the threshold on the training fit puts
   it wherever the model overfits to, which on this data was ~0.99 train AUC.
2. **Cap the called fraction.** MCC and F1 will both happily maximise at a very permissive
   cut when the model is weak. One run chose p >= 0.11, which called 86% of compounds
   active at recall 1.000 -- over a 290,000-compound library, a 250,000-compound
   "shortlist". The point of a screen is to decide what *not* to look at, so a cut that
   does not narrow anything is not a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_curve,
)

__all__ = ["ThresholdChoice", "threshold_sweep", "choose_threshold", "list_size_table"]

_OBJECTIVES = ("mcc", "f1", "balanced_accuracy", "youden")


def threshold_sweep(
    y_true,
    probabilities,
    grid: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Precision, recall, MCC, F1, balanced accuracy and called fraction per threshold."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probabilities, dtype=float)
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.01), 2)

    fpr, tpr, roc_thr = roc_curve(y, p) if len(np.unique(y)) > 1 else (None, None, None)
    youden_lookup: Dict[float, float] = {}
    if roc_thr is not None:
        for t, tp, fp in zip(roc_thr, tpr, fpr):
            youden_lookup[float(np.round(t, 2))] = float(tp - fp)

    rows: List[Dict[str, float]] = []
    for t in grid:
        t = float(t)
        yp = (p >= t).astype(int)
        two_classes = len(np.unique(yp)) > 1
        rows.append(
            {
                "threshold": t,
                "precision": float(precision_score(y, yp, zero_division=0)),
                "recall": float(recall_score(y, yp, zero_division=0)),
                "f1": float(f1_score(y, yp, zero_division=0)),
                "mcc": float(matthews_corrcoef(y, yp)) if two_classes else 0.0,
                "balanced_accuracy": float(balanced_accuracy_score(y, yp)),
                "youden": youden_lookup.get(round(t, 2), float("nan")),
                "called_frac": float(yp.mean()),
                "n_called": int(yp.sum()),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class ThresholdChoice:
    """A chosen cut, with everything needed to argue about it."""

    threshold: float
    objective: str
    objective_value: float
    precision: float
    recall: float
    called_frac: float
    capped: bool  # the cap bound, i.e. the unconstrained optimum was rejected
    unconstrained_threshold: float
    unconstrained_called_frac: float
    degenerate: bool  # recall ~ 1.0 at the chosen cut
    sweep: pd.DataFrame
    notes: List[str]

    def as_dict(self) -> Dict[str, object]:
        return {
            "threshold": self.threshold,
            "objective": self.objective,
            "objective_value": self.objective_value,
            "precision": self.precision,
            "recall": self.recall,
            "called_frac": self.called_frac,
            "capped": self.capped,
            "degenerate": self.degenerate,
        }

    def precision_targets(
        self,
        targets: Sequence[float] = (0.80, 0.90, 0.95),
        library_size: Optional[int] = None,
        min_called: int = 10,
    ) -> pd.DataFrame:
        """Highest-recall threshold reaching each precision target, if reachable.

        ``min_called`` guards against the tail artefact: far out on the threshold grid a
        handful of compounds survive and may all happen to be active, giving a precision
        of 1.0 that describes three molecules. A target met only by a selection smaller
        than ``min_called`` is reported as unreachable, because it is not a target anyone
        can act on.
        """
        rows = []
        for target in targets:
            ok = self.sweep[
                (self.sweep["precision"] >= target)
                & (self.sweep["n_called"] >= min_called)
            ]
            if len(ok):
                r = ok.loc[ok["recall"].idxmax()]
                row = {
                    "precision_target": target,
                    "threshold": float(r["threshold"]),
                    "recall": float(r["recall"]),
                    "called_frac": float(r["called_frac"]),
                    "reachable": True,
                }
                if library_size:
                    row["est_n_called"] = int(round(r["called_frac"] * library_size))
            else:
                row = {
                    "precision_target": target,
                    "threshold": float("nan"),
                    "recall": float("nan"),
                    "called_frac": float("nan"),
                    "reachable": False,
                }
                if library_size:
                    row["est_n_called"] = 0
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = [
            f"max {self.objective} at p >= {self.threshold:.2f}  "
            f"precision {self.precision:.3f}  recall {self.recall:.3f}  "
            f"calls {100 * self.called_frac:.1f}%"
        ]
        lines.extend(self.notes)
        return "\n".join(lines)


def choose_threshold(
    y_true,
    probabilities,
    *,
    objective: str = "mcc",
    max_called_frac: float = 0.25,
    override: Optional[float] = None,
    grid: Optional[Sequence[float]] = None,
) -> ThresholdChoice:
    """Pick a decision threshold from out-of-fold predictions.

    Restricts to thresholds calling at most ``max_called_frac`` active, then maximises
    ``objective``. If no threshold satisfies the cap, the most stringent one in the grid is
    used and a note says so -- that situation means the model is not separating the
    classes, which is worth reporting rather than silently accepting.
    """
    if objective not in _OBJECTIVES:
        raise ValueError(f"objective must be one of {_OBJECTIVES}, got {objective!r}")
    if not 0.0 < max_called_frac <= 1.0:
        raise ValueError("max_called_frac must be in (0, 1]")

    y_arr = np.asarray(y_true).astype(int)
    if len(np.unique(y_arr)) < 2:
        raise ValueError(
            "cannot choose a threshold from single-class labels: every objective is "
            "degenerate. Check that the active cut falls inside the data."
        )

    sweep = threshold_sweep(y_true, probabilities, grid)
    if sweep[objective].isna().all():
        raise ValueError(
            f"objective {objective!r} is undefined everywhere on this input"
        )

    unconstrained = sweep.loc[sweep[objective].idxmax()]
    # A cut must call *something* as well as not calling too much. Without the lower
    # bound the cap is trivially satisfiable by a threshold that selects nothing, and the
    # "no threshold is stringent enough" case below could never be reached.
    allowed = sweep[(sweep["called_frac"] <= max_called_frac) & (sweep["n_called"] > 0)]
    capped_ok = len(allowed) > 0
    if capped_ok:
        best = allowed.loc[allowed[objective].idxmax()]
    else:
        # Fall back to the most stringent cut that still selects *something*. Falling
        # back to the very end of the grid can land on a threshold that calls nothing,
        # which is not an answer.
        non_empty = sweep[sweep["n_called"] > 0]
        best = non_empty.iloc[-1] if len(non_empty) else sweep.iloc[-1]

    chosen = float(override) if override is not None else float(best["threshold"])
    notes: List[str] = []

    if not capped_ok:
        notes.append(
            f"!! NO threshold produces a non-empty selection calling "
            f"<= {100 * max_called_frac:.0f}% of compounds active. Using the most "
            f"stringent cut in the sweep (p >= {best['threshold']:.2f}), which calls "
            f"{100 * best['called_frac']:.1f}%. The scores are too concentrated to "
            f"separate the classes at this cap; treat the output as a weak ranking only."
        )
    elif float(unconstrained["threshold"]) != float(best["threshold"]):
        notes.append(
            f"(unconstrained max {objective} was at p >= "
            f"{unconstrained['threshold']:.2f}, but it calls "
            f"{100 * unconstrained['called_frac']:.1f}% active -- above the "
            f"{100 * max_called_frac:.0f}% cap -- so it was rejected. That cut maximises "
            f"{objective} by being permissive, which is not a shortlist.)"
        )

    degenerate = bool(best["recall"] >= 0.99)
    if degenerate:
        notes.append(
            "!! recall is ~1.0 at this cut: it is calling essentially every active AND "
            "most inactives. Lower max_called_frac."
        )
    if override is not None:
        notes.append(f"-> override is set: using p >= {chosen:.2f}")

    # Report the metrics at the threshold actually used, which may be the override.
    row = sweep.iloc[(sweep["threshold"] - chosen).abs().idxmin()]
    return ThresholdChoice(
        threshold=chosen,
        objective=objective,
        objective_value=float(row[objective]),
        precision=float(row["precision"]),
        recall=float(row["recall"]),
        called_frac=float(row["called_frac"]),
        capped=bool(capped_ok and float(unconstrained["threshold"]) != float(best["threshold"])),
        unconstrained_threshold=float(unconstrained["threshold"]),
        unconstrained_called_frac=float(unconstrained["called_frac"]),
        degenerate=degenerate,
        sweep=sweep,
        notes=notes,
    )


def list_size_table(
    library_scores,
    sweep: pd.DataFrame,
    sizes: Sequence[int] = (100, 500, 1000, 5000, 10000, 25000),
) -> pd.DataFrame:
    """What each shortlist size implies.

    The threshold column is read off the *library's* score distribution; the precision and
    recall columns are what that same threshold achieved out-of-fold on held-out training
    families. Those are the best estimates available, and they are optimistic whenever the
    library is further from the training set than any held-out fold -- which for a
    diversity catalogue screened with a congeneric training set it certainly is.
    """
    scores = pd.Series(np.asarray(library_scores, dtype=float)).dropna()
    rows: List[Dict[str, object]] = []
    for n in sizes:
        if n > len(scores):
            continue
        t = float(scores.nlargest(n).iloc[-1])
        nearest = sweep.iloc[(sweep["threshold"] - t).abs().idxmin()]
        rows.append(
            {
                "list_size": int(n),
                "implied_threshold": round(t, 4),
                "oof_precision": round(float(nearest["precision"]), 3),
                "oof_recall": round(float(nearest["recall"]), 3),
                "oof_mcc": round(float(nearest["mcc"]), 3),
                "in_sweep_range": bool(
                    sweep["threshold"].min() <= t <= sweep["threshold"].max()
                ),
            }
        )
    return pd.DataFrame(rows)
