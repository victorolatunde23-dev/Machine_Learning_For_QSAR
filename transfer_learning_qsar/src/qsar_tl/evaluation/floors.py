"""Floor comparisons: what are the extra models actually buying?

Two baselines, both cheap, both of which a QSAR model has to clear to justify existing:

* **kNN (Tanimoto)** is nearest-neighbour lookup -- "this looks like a known active". It
  is the similarity principle unadorned.
* **LogReg (L2)** is a linear model on fingerprint bits. If an ensemble cannot beat a
  linear model, the complexity is not paying for itself.

Measured on **both** ROC-AUC and precision@N, because they disagree and the disagreement
is the finding. On the PSMA pKi set LightGBM cleared lookup by +0.092 AUC -- real
structure-activity signal, not neighbour recall -- but *nothing* cleared the linear floor
on precision@50. A shortlist only uses the top of the ranking, and there the linear model
was as good as the best ensemble. Reporting AUC alone would have hidden that; reporting
precision alone would have hidden the first half.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["FloorComparison", "compare_to_floors", "blend_vs_best_member"]


@dataclass
class FloorComparison:
    """One floor, and which models cleared it on which metric."""

    floor: str
    metric_values: Dict[str, float] = field(default_factory=dict)
    clears: Dict[str, List[tuple]] = field(default_factory=dict)  # metric -> [(model, gain)]
    margins: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        head = "    ".join(
            f"{m} {v:.3f}" for m, v in self.metric_values.items() if np.isfinite(v)
        )
        lines = [f"{self.floor:<16s} {head}"]
        for metric, winners in self.clears.items():
            margin = self.margins.get(metric, 0.0)
            if winners:
                detail = ", ".join(f"{n} ({g:+.3f})" for n, g in winners)
                lines.append(f"  on {metric} (>{margin:.2f}) : {detail}")
            else:
                lines.append(f"  on {metric} (>{margin:.2f}) : NOTHING clears it")
        lines.extend(self._verdict())
        return "\n".join(lines)

    def _verdict(self) -> List[str]:
        """Say what a split result means, per floor, rather than printing one message."""
        metrics = list(self.clears)
        if len(metrics) < 2:
            return []
        disc, top = metrics[0], metrics[-1]
        won_disc = bool(self.clears.get(disc))
        won_top = bool(self.clears.get(top))
        if won_disc and not won_top:
            return [
                "  -> ahead on overall discrimination but NOT at the top of the",
                "     ranking. For a shortlist that is the wrong way round.",
            ]
        if won_top and not won_disc:
            return [
                "  -> tie on whole-ranking discrimination, but the models ARE ahead at",
                "     the top of the ranking, which is the part a screen uses.",
            ]
        if not won_disc and not won_top:
            if self.floor.startswith("kNN"):
                return [
                    "  -> nothing beats nearest-neighbour lookup. On this data the models",
                    "     are recognising neighbours rather than learning structure-activity",
                    "     relationships, which for ranking unfamiliar chemistry is the",
                    "     single most important thing to know.",
                ]
            return [
                "  -> nothing beats a linear model on fingerprints. The ensembles are not",
                "     paying for their complexity here.",
            ]
        return ["  -> ahead on both. The extra model capacity is earning its place."]


def compare_to_floors(
    pooled: pd.DataFrame,
    *,
    floors: Sequence[str] = ("kNN (Tanimoto)", "LogReg (L2)"),
    metrics: Sequence[str] = ("roc_auc", "precision@50"),
    margins: Optional[Dict[str, float]] = None,
) -> List[FloorComparison]:
    """Compare every model in ``pooled`` against each floor on each metric.

    ``pooled`` is indexed by model name, one row per model, as produced by the pooled
    aggregation in :mod:`qsar_tl.evaluation.loco`.

    ``margins`` is the gain a model must exceed to count as clearing a floor; it defaults
    to 0.02 for AUC-like metrics and 0.05 for precision@N, because precision at a fixed N
    moves in steps of 1/N and a one-compound difference is not a result.
    """
    margins = dict(margins or {})
    out: List[FloorComparison] = []
    for floor in floors:
        if floor not in pooled.index:
            continue
        comp = FloorComparison(floor=floor)
        for metric in metrics:
            if metric not in pooled.columns:
                continue
            base = float(pooled.loc[floor, metric])
            comp.metric_values[metric] = base
            default_margin = 0.05 if metric.startswith("precision@") else 0.02
            margin = float(margins.get(metric, default_margin))
            comp.margins[metric] = margin
            winners = []
            for name, row in pooled.iterrows():
                if name == floor:
                    continue
                value = float(row[metric])
                if np.isfinite(value) and value - base > margin:
                    winners.append((str(name), value - base))
            winners.sort(key=lambda t: -t[1])
            comp.clears[metric] = winners
        out.append(comp)
    return out


def blend_vs_best_member(
    pooled: pd.DataFrame,
    *,
    blend_name: str = "Blend",
    metric: str = "roc_auc",
    members: Optional[Sequence[str]] = None,
) -> str:
    """Does the committee beat its best single member?

    Asked every time because the answer was no on seven consecutive endpoints: the
    equal-weight blend trailed its best member by 0.021 to 0.033 AUC. A blend that cannot
    beat a member is costing compute and interpretability for nothing.
    """
    blend_rows = [i for i in pooled.index if str(i).startswith(blend_name)]
    if not blend_rows or metric not in pooled.columns:
        return "No blend row to compare."
    blend = blend_rows[0]
    pool = pooled.drop(index=blend_rows)
    if members:
        pool = pool.loc[[m for m in members if m in pool.index]]
    if pool.empty:
        return "No members to compare the blend against."
    best = pool[metric].idxmax()
    delta = float(pooled.loc[blend, metric]) - float(pool.loc[best, metric])
    verdict = (
        f"'{best}' alone is as good or better -- the blend is not earning it"
        if delta <= 0
        else "the blend is ahead of every member"
    )
    return (
        f"{blend} {float(pooled.loc[blend, metric]):.3f} vs best single {best} "
        f"{float(pool.loc[best, metric]):.3f} ({delta:+.3f}) -- {verdict}"
    )
