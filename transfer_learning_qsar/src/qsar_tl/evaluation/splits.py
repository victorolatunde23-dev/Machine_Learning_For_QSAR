"""Grouped splits: Butina clustering, leave-one-cluster-out folds, scaffold splits.

A random split of a congeneric dataset leaves close analogues on both sides of it, so the
model is asked about molecules it has effectively already seen. On this PSMA set that
inflated ROC-AUC by 0.06 to 0.13 depending on the model, and the inflation was largest for
nearest-neighbour lookup (-0.129) -- exactly the method that can only work by recognising
neighbours, which is the mechanism made visible.

Leave-one-cluster-out holds out a whole chemical family at a time, which is the situation
a library screen actually puts a model in.

The clustering itself needs fingerprints, so :func:`butina_clusters` takes a precomputed
distance callable and :func:`butina_clusters_from_fingerprints` is the RDKit-backed
convenience wrapper. That split keeps the fold-construction logic -- the part with the
judgement in it -- testable without a chemistry toolkit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .._optional import require, tqdm

__all__ = [
    "Fold",
    "FoldSet",
    "butina_clusters",
    "butina_clusters_from_fingerprints",
    "condensed_tanimoto_distances",
    "build_loco_folds",
    "scaffold_groups",
]


# ----------------------------------------------------------------------------- clustering


def butina_clusters(
    condensed_distances: Sequence[float], n_items: int, cutoff: float = 0.4
) -> np.ndarray:
    """Butina cluster id per item, from a condensed lower-triangle distance list.

    ``condensed_distances`` is ordered as RDKit expects: for i in 1..n-1, for j in 0..i-1.
    Returns an array of length ``n_items``.
    """
    Butina = require("rdkit").ML.Cluster.Butina  # type: ignore[attr-defined]
    clusters = Butina.ClusterData(
        list(condensed_distances), int(n_items), float(cutoff), isDistData=True
    )
    groups = np.full(int(n_items), -1, dtype=int)
    for cid, members in enumerate(clusters):
        for idx in members:
            groups[idx] = cid
    return groups


def condensed_tanimoto_distances(fingerprints, *, progress: bool = True) -> List[float]:
    """Condensed 1 - Tanimoto distances, in the order Butina expects."""
    DataStructs = require("rdkit").DataStructs  # type: ignore[attr-defined]
    n = len(fingerprints)
    out: List[float] = []
    rows = range(1, n)
    it = tqdm(rows, desc="Butina distances", leave=False) if progress else rows
    for i in it:
        sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        out.extend(1.0 - s for s in sims)
    return out


def butina_clusters_from_fingerprints(
    fingerprints, cutoff: float = 0.4, *, progress: bool = True
) -> np.ndarray:
    """Cluster fingerprints with Butina at ``cutoff`` Tanimoto distance."""
    d = condensed_tanimoto_distances(fingerprints, progress=progress)
    return butina_clusters(d, len(fingerprints), cutoff)


def scaffold_groups(smiles: Sequence[str]) -> np.ndarray:
    """Bemis-Murcko scaffold group ids. A cheaper, coarser alternative to Butina."""
    rdkit = require("rdkit")
    from rdkit import Chem  # type: ignore
    from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore

    keys: Dict[str, int] = {}
    out = np.empty(len(smiles), dtype=int)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            key = f"__unparsed_{i}"
        else:
            try:
                key = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            except Exception:
                key = f"__failed_{i}"
        out[i] = keys.setdefault(key, len(keys))
    return out


# --------------------------------------------------------------------------------- folds


@dataclass
class Fold:
    """One held-out chemical family."""

    name: str
    mask: np.ndarray  # boolean, over the training set
    n: int = 0
    n_active: int = 0
    base_rate: float = float("nan")
    dominant_source: str = "?"
    source_share: float = float("nan")
    mean_distance_to_train: float = float("nan")
    scorable: bool = True

    def as_row(self) -> Dict[str, object]:
        return {
            "fold": self.name,
            "n": self.n,
            "n_active": self.n_active,
            "base_rate": round(self.base_rate, 3) if np.isfinite(self.base_rate) else np.nan,
            "dominant_source": self.dominant_source,
            "source_share": (
                round(self.source_share, 2) if np.isfinite(self.source_share) else np.nan
            ),
            "mean_distance_to_train": (
                round(self.mean_distance_to_train, 3)
                if np.isfinite(self.mean_distance_to_train)
                else np.nan
            ),
        }


@dataclass
class FoldSet:
    """The folds, plus what the clustering looked like."""

    folds: List[Fold] = field(default_factory=list)
    n_clusters: int = 0
    largest_cluster: int = 0
    n_singletons: int = 0
    rejected: List[tuple] = field(default_factory=list)  # (cluster id, size)
    pooled_cluster_count: int = 0

    def __len__(self) -> int:
        return len(self.folds)

    def __iter__(self):
        return iter(self.folds)

    def to_frame(self) -> pd.DataFrame:
        if not self.folds:
            return pd.DataFrame(columns=["fold", "n", "n_active", "base_rate"])
        return pd.DataFrame([f.as_row() for f in self.folds]).set_index("fold")

    @property
    def coverage(self) -> int:
        return int(sum(f.n for f in self.folds))

    def single_source_folds(
        self, min_share: float = 0.6, generic: Optional[frozenset] = None
    ) -> List[Fold]:
        """Folds dominated by one real source -- the hardest case, worth reading apart.

        Holding one of these out removes a large, chemically narrow block of training data
        at the same moment it poses an unfamiliar test set, so a low score there is two
        effects at once rather than one.
        """
        from ..data.labels import GENERIC_SOURCE_TAGS

        generic = generic or GENERIC_SOURCE_TAGS
        return [
            f
            for f in self.folds
            if np.isfinite(f.source_share)
            and f.source_share >= min_share
            and f.dominant_source not in generic
        ]


def build_loco_folds(
    groups: np.ndarray,
    y: np.ndarray,
    *,
    min_cluster_size: int = 25,
    min_fold_actives: int = 5,
    max_own_folds: int = 12,
    n_remainder_folds: int = 3,
    source_tags: Optional[Sequence[str]] = None,
    distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> FoldSet:
    """Turn cluster ids into leave-one-cluster-out folds.

    A cluster gets its own fold when it is both large enough (``min_cluster_size``) and
    balanced enough to score (``min_fold_actives`` of *each* class). Everything else is
    pooled into ``n_remainder_folds`` size-balanced "small families" folds by greedy
    assignment, so the remainder still produces usable estimates instead of a long tail of
    one-compound folds.

    The scorability requirement is why this is not a plain GroupKFold: an earlier version
    of this evaluation reported per-fold metrics for folds of 5 compounds whose ROC-AUC
    ranged 0.500-0.917 across models and meant nothing. Clusters that are large but too
    unbalanced to score are pooled too, and reported in ``rejected`` so the reason is
    visible rather than silent.

    Parameters
    ----------
    groups
        Cluster id per compound.
    y
        Binary labels; NaN is allowed for carried-but-unlabelled compounds.
    distance_fn
        Optional ``(held_out_idx, train_idx) -> mean distance``, used only to annotate
        folds. Omitted when no chemistry toolkit is available.
    """
    groups = np.asarray(groups)
    y = np.asarray(y, dtype=float)
    n = len(groups)
    if len(y) != n:
        raise ValueError(f"groups ({n}) and y ({len(y)}) differ in length")

    sizes = pd.Series(groups).value_counts()
    labelled = np.isfinite(y)

    def scorable(mask: np.ndarray) -> bool:
        lab = y[mask & labelled]
        return bool(
            (lab == 1).sum() >= min_fold_actives and (lab == 0).sum() >= min_fold_actives
        )

    own: List[int] = []
    pooled: List[int] = []
    rejected: List[tuple] = []
    for cid in sizes.index:
        cid = int(cid)
        mask = groups == cid
        if sizes[cid] < min_cluster_size:
            pooled.append(cid)
        elif not scorable(mask):
            pooled.append(cid)
            rejected.append((cid, int(sizes[cid])))
        elif len(own) < max_own_folds:
            own.append(cid)
        else:
            pooled.append(cid)

    # greedy size-balanced assignment of the remainder
    bins: List[List[int]] = [[] for _ in range(max(1, n_remainder_folds))]
    bin_sizes = [0] * len(bins)
    for cid in sorted(pooled, key=lambda c: -int(sizes[c])):
        j = int(np.argmin(bin_sizes))
        bins[j].append(cid)
        bin_sizes[j] += int(sizes[cid])

    folds: List[Fold] = []
    for cid in own:
        folds.append(Fold(name=f"cluster {cid}", mask=groups == cid))
    for j, cids in enumerate(bins):
        if cids:
            folds.append(
                Fold(name=f"small families #{j + 1}", mask=np.isin(groups, cids))
            )

    for f in folds:
        m = f.mask
        lab = y[m & labelled]
        f.n = int(m.sum())
        f.n_active = int((lab == 1).sum())
        f.base_rate = float(lab.mean()) if len(lab) else float("nan")
        f.scorable = scorable(m)
        if source_tags is not None:
            counts = pd.Series(np.asarray(source_tags)[m]).value_counts()
            if len(counts):
                f.dominant_source = str(counts.index[0])
                f.source_share = float(counts.iloc[0] / f.n)
        if distance_fn is not None:
            f.mean_distance_to_train = float(
                distance_fn(np.flatnonzero(m), np.flatnonzero(~m))
            )

    return FoldSet(
        folds=folds,
        n_clusters=int(len(sizes)),
        largest_cluster=int(sizes.iloc[0]) if len(sizes) else 0,
        n_singletons=int((sizes == 1).sum()),
        rejected=rejected,
        pooled_cluster_count=len(pooled),
    )
