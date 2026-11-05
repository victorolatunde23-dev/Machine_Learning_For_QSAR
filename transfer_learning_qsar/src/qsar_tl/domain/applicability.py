"""Applicability domain: four distance metrics, with Tanimoto as a veto.

Each metric is thresholded at a percentile of the *training set's own internal
nearest-neighbour distances*, so "out of domain" means "further from the training set than
training compounds are from each other", which is a statement the training data itself
supports.

The veto is the important design decision. Treating Tanimoto as one vote in four let
compounds pass while failing the structural test, and for a congeneric target that is the
flag with teeth: the other three metrics compare bulk physicochemical properties, so a
molecule of roughly the right size and lipophilicity passes them while being structurally
unrelated to anything in training. In one run every single top hit was flagged
``tanimoto_OOD`` (dissimilarity 0.73-0.83 against a 0.457 threshold) and every one was
nevertheless reported as in-domain -- while the grouped evaluation showed that is precisely
the regime where the models fail. With the veto on, failing Tanimoto alone puts a compound
out of domain regardless of the other three.

A caution worth keeping in view: on a diversity catalogue screened with a congeneric
training set, this will legitimately exclude almost everything. That is a real chemistry
mismatch rather than a fixable modelling defect, and the honest response is either to
accept that predictions are extrapolations or to change the library -- not to loosen the
domain until the answer looks better.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .._optional import require, tqdm

__all__ = [
    "ApplicabilityDomain",
    "min_tanimoto_dissimilarity",
    "internal_tanimoto_dissimilarity",
    "FLAGS",
]

FLAGS = ("tanimoto_OOD", "descriptor_OOD", "latent_OOD", "unfamiliarity_OOD")


def min_tanimoto_dissimilarity(query_fps, reference_fps, *, progress: bool = False,
                               desc: str = "Tanimoto to nearest reference") -> np.ndarray:
    """``1 - max similarity`` to anything in ``reference_fps``, per query."""
    DataStructs = require("rdkit").DataStructs  # type: ignore[attr-defined]
    ref = list(reference_fps)
    out = np.empty(len(query_fps))
    it = tqdm(query_fps, desc=desc, unit="mol", leave=False) if progress else query_fps
    for i, fp in enumerate(it):
        out[i] = 1.0 - max(DataStructs.BulkTanimotoSimilarity(fp, ref))
    return out


def internal_tanimoto_dissimilarity(fps, *, progress: bool = False) -> np.ndarray:
    """Each compound's distance to its nearest *other* compound in the same set."""
    DataStructs = require("rdkit").DataStructs  # type: ignore[attr-defined]
    fps = list(fps)
    out = np.empty(len(fps))
    it = (
        tqdm(fps, desc="AD calibration: internal NN Tanimoto", unit="mol", leave=False)
        if progress
        else fps
    )
    for i, fp in enumerate(it):
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
        sims[i] = -1.0  # exclude self
        out[i] = 1.0 - max(sims)
    return out


@dataclass
class ApplicabilityDomain:
    """Four-metric domain, calibrated on a training set.

    Parameters mirror :class:`qsar_tl.config.DomainConfig`.
    """

    percentile: float = 95.0
    consensus_min_flags: int = 3
    tanimoto_veto: bool = True
    unfamiliarity_k: int = 5
    latent_pca_components: int = 10
    random_state: int = 42

    thresholds: Dict[str, float] = field(default_factory=dict)
    _scaler: Optional[StandardScaler] = None
    _pca: Optional[PCA] = None
    _nn_descriptor: Optional[NearestNeighbors] = None
    _nn_latent: Optional[NearestNeighbors] = None
    _nn_unfamiliarity: Optional[NearestNeighbors] = None
    _train_fps: Optional[List] = None
    _n_train: int = 0

    # ------------------------------------------------------------------------- fitting

    def fit(self, train_fps, train_descriptors: np.ndarray,
            *, progress: bool = False) -> "ApplicabilityDomain":
        """Calibrate every threshold on the training set's internal distances.

        Fit on *every* compound carried through, including those with no label for the
        modelled endpoint: the domain is a statement about chemical space coverage, not
        about which label happens to be present, and a compound measured on another
        endpoint is still evidence that its region of chemistry is familiar ground.
        """
        desc = np.nan_to_num(np.asarray(train_descriptors, dtype=float))
        self._n_train = len(desc)
        self._scaler = StandardScaler().fit(desc)
        scaled = self._scaler.transform(desc)

        n_comp = max(1, min(self.latent_pca_components, scaled.shape[1], len(scaled) - 1))
        self._pca = PCA(n_components=n_comp, random_state=self.random_state).fit(scaled)
        latent = self._pca.transform(scaled)

        internal = internal_tanimoto_dissimilarity(train_fps, progress=progress)
        self.thresholds["tanimoto"] = float(np.percentile(internal, self.percentile))

        for key, data in (("descriptor_manhattan", scaled), ("latent_manhattan", latent)):
            nn = NearestNeighbors(n_neighbors=min(2, len(data)), p=1).fit(data)
            d, _ = nn.kneighbors(data)
            col = d[:, 1] if d.shape[1] > 1 else d[:, 0]
            self.thresholds[key] = float(np.percentile(col, self.percentile))

        k = min(self.unfamiliarity_k + 1, len(scaled))
        nn_unfam = NearestNeighbors(n_neighbors=max(2, k)).fit(scaled)
        d_unfam, _ = nn_unfam.kneighbors(scaled)
        self.thresholds["unfamiliarity"] = float(
            np.percentile(d_unfam[:, 1:].mean(axis=1), self.percentile)
        )

        self._nn_descriptor = NearestNeighbors(n_neighbors=1, p=1).fit(scaled)
        self._nn_latent = NearestNeighbors(n_neighbors=1, p=1).fit(latent)
        self._nn_unfamiliarity = NearestNeighbors(
            n_neighbors=min(self.unfamiliarity_k, len(scaled))
        ).fit(scaled)
        self._train_fps = list(train_fps)
        return self

    # ------------------------------------------------------------------------- scoring

    def score(self, fps, descriptors: np.ndarray, *, progress: bool = False) -> pd.DataFrame:
        """Distances, per-metric flags, flag count and the consensus verdict."""
        if self._scaler is None:
            raise RuntimeError("ApplicabilityDomain is not fitted")
        scaled = self._scaler.transform(
            np.nan_to_num(np.asarray(descriptors, dtype=float))
        )
        latent = self._pca.transform(scaled)

        tan = min_tanimoto_dissimilarity(
            fps, self._train_fps, progress=progress,
            desc="AD: Tanimoto to nearest training compound",
        )
        d_desc, _ = self._nn_descriptor.kneighbors(scaled)
        d_lat, _ = self._nn_latent.kneighbors(latent)
        d_unfam, _ = self._nn_unfamiliarity.kneighbors(scaled)

        out = pd.DataFrame(
            {
                "tanimoto_dissimilarity": tan,
                "descriptor_manhattan": d_desc[:, 0],
                "latent_manhattan": d_lat[:, 0],
                "unfamiliarity": d_unfam.mean(axis=1),
            }
        )
        out["tanimoto_OOD"] = out["tanimoto_dissimilarity"] > self.thresholds["tanimoto"]
        out["descriptor_OOD"] = (
            out["descriptor_manhattan"] > self.thresholds["descriptor_manhattan"]
        )
        out["latent_OOD"] = out["latent_manhattan"] > self.thresholds["latent_manhattan"]
        out["unfamiliarity_OOD"] = out["unfamiliarity"] > self.thresholds["unfamiliarity"]
        out["n_flags_OOD"] = out[list(FLAGS)].sum(axis=1)

        out["consensus_OOD"] = out["n_flags_OOD"] >= self.consensus_min_flags
        if self.tanimoto_veto:
            out["consensus_OOD"] = out["consensus_OOD"] | out["tanimoto_OOD"]
        return out

    # ------------------------------------------------------------------------ reporting

    def report(self, scores: pd.DataFrame) -> str:
        """Human-readable summary, including what the veto changed."""
        n = len(scores)
        lines = [
            f"AD thresholds ({self.percentile:g}th percentile of the training set's own "
            f"internal NN distances, n={self._n_train:,}):"
        ]
        for key, value in self.thresholds.items():
            lines.append(f"  {key}: {value:.4f}")
        lines.append(f"\nScored library: {n:,} compounds")
        for flag in FLAGS:
            lines.append(
                f"  {flag:>18s} fires on {int(scores[flag].sum()):>8,} "
                f"({100 * scores[flag].mean():5.1f}%)"
            )
        in_domain = int((~scores["consensus_OOD"]).sum())
        lines.append(
            f"\n  Tanimoto veto: {'ON' if self.tanimoto_veto else 'OFF'}   "
            f"(consensus_min_flags = {self.consensus_min_flags})"
        )
        lines.append(
            f"  -> in domain: {in_domain:,} / {n:,} "
            f"({100 * in_domain / max(n, 1):.2f}%)"
        )
        if self.tanimoto_veto:
            without = int((scores["n_flags_OOD"] < self.consensus_min_flags).sum())
            lines.append(
                f"     without the veto it would have been {without:,} "
                f"({100 * without / max(n, 1):.2f}%) -- the veto removes "
                f"{without - in_domain:,} compounds that passed on bulk-property grounds"
            )
            lines.append(
                "     while being structurally unlike anything in the training set."
            )
        if in_domain == 0:
            lines.append(
                "\n  !! NOTHING is in domain. That is a chemistry mismatch between this "
                "library\n     and the training set, not a threshold to loosen. Either "
                "accept that every\n     prediction is an extrapolation and use the "
                "ranking only, or screen a\n     library that overlaps the training "
                "chemistry."
            )
        return "\n".join(lines)
