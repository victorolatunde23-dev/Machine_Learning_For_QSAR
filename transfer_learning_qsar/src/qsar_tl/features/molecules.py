"""Standardisation, fingerprints, descriptors and design-matrix assembly.

All of this needs RDKit, which is an optional dependency: see :mod:`qsar_tl._optional`.

On feature choice: the bake-off on the PSMA pKi set gave ``fp_only`` 0.865,
``fp_plus_descriptors`` 0.850 and ``descriptors_only`` 0.844, so the 70 curated
descriptors were actively costing accuracy. They are also the most expensive thing in the
pipeline -- 217 RDKit descriptors per molecule, several of which need an eigendecomposition
-- which over a few hundred thousand compounds dominates the runtime. ``fp_only`` is
therefore the default. The descriptor machinery is kept because the applicability domain
needs it and because the bake-off should be re-runnable on a new target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._optional import require, tqdm

__all__ = [
    "Standardiser",
    "morgan_fingerprint",
    "fingerprints_to_array",
    "descriptor_names",
    "compute_descriptors",
    "AutoQSARDescriptorSelector",
    "FeaturisedSet",
    "featurise",
    "build_design_matrix",
]


# -------------------------------------------------------------------------- standardising


class Standardiser:
    """Normalise -> largest fragment -> uncharge, then sanitise.

    Applied to the training set and the library alike, so the two are described on the
    same footing. It also makes InChIKey deduplication meaningful: in one catalogue 31% of
    the entries turned out to be protonation and tautomer variants of structures already
    present, which inflates both the library count and the apparent hit count.
    """

    def __init__(self) -> None:
        rdMolStandardize = require("rdkit").Chem.MolStandardize.rdMolStandardize  # type: ignore
        self._normalizer = rdMolStandardize.Normalizer()
        self._fragment = rdMolStandardize.LargestFragmentChooser()
        self._uncharger = rdMolStandardize.Uncharger()

    def __call__(self, mol):
        """Return a standardised molecule, or None if any step fails."""
        if mol is None:
            return None
        Chem = require("rdkit").Chem  # type: ignore[attr-defined]
        try:
            m = self._normalizer.normalize(mol)
            m = self._fragment.choose(m)
            m = self._uncharger.uncharge(m)
            Chem.SanitizeMol(m)
            return m
        except Exception:
            return None

    def from_smiles(self, smiles: object):
        Chem = require("rdkit").Chem  # type: ignore[attr-defined]
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        return self(Chem.MolFromSmiles(smiles))


def inchikey(mol) -> Optional[str]:
    """InChIKey, falling back to canonical SMILES when InChI generation fails."""
    Chem = require("rdkit").Chem  # type: ignore[attr-defined]
    try:
        return Chem.inchi.MolToInchiKey(mol)
    except Exception:
        try:
            return Chem.MolToSmiles(mol)
        except Exception:
            return None


def skeleton_key(key: Optional[str]) -> Optional[str]:
    """The connectivity block of an InChIKey -- the first 14 characters.

    Stereochemistry, tautomer and protonation state live in the later blocks, so this
    matches structures that are the same constitution. Useful as a second tier in a
    positive control: a full-key comparison once reported "no library compound matches a
    training compound" while two of them had Tanimoto distance exactly 0.000 to a training
    structure, because the full key differed in a layer an achiral fingerprint ignores.
    """
    if not key or len(key) < 14:
        return key
    return key[:14]


# --------------------------------------------------------------------------- fingerprints


def morgan_fingerprint(mol, radius: int = 2, n_bits: int = 2048):
    """Morgan (ECFP-like) bit vector."""
    AllChem = require("rdkit").Chem.AllChem  # type: ignore[attr-defined]
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def fingerprints_to_array(fingerprints, dtype=np.uint8) -> np.ndarray:
    """Stack bit vectors into an ``(n, n_bits)`` array."""
    DataStructs = require("rdkit").DataStructs  # type: ignore[attr-defined]
    if not len(fingerprints):
        return np.zeros((0, 0), dtype=dtype)
    n_bits = fingerprints[0].GetNumBits()
    out = np.zeros((len(fingerprints), n_bits), dtype=dtype)
    for i, fp in enumerate(fingerprints):
        DataStructs.ConvertToNumpyArray(fp, out[i])
    return out


# ----------------------------------------------------------------------------- descriptors


def descriptor_names() -> List[str]:
    """Every RDKit 2D descriptor name available in this install."""
    Descriptors = require("rdkit").Chem.Descriptors  # type: ignore[attr-defined]
    return [name for name, _fn in Descriptors._descList]


def compute_descriptors(mols, names: Optional[Sequence[str]] = None,
                        *, progress: bool = False) -> Tuple[np.ndarray, List[str]]:
    """``(n, n_descriptors)`` array plus the names, NaN where a descriptor fails."""
    Descriptors = require("rdkit").Chem.Descriptors  # type: ignore[attr-defined]
    table = dict(Descriptors._descList)
    use = list(names) if names is not None else list(table)
    fns = [table[n] for n in use]
    rows = np.full((len(mols), len(use)), np.nan)
    it = tqdm(mols, desc="Descriptors", unit="mol", leave=False) if progress else mols
    for i, mol in enumerate(it):
        if mol is None:
            continue
        for j, fn in enumerate(fns):
            try:
                rows[i, j] = float(fn(mol))
            except Exception:
                pass
    return rows, use


class AutoQSARDescriptorSelector:
    """Near-constant filter, then correlation-cluster redundancy reduction.

    As described in the AutoQSAR paper (Dixon et al., *Future Med. Chem.* 2016). Stage 1
    drops descriptors whose most common value covers more than ``near_constant_frac`` of
    the rows; stage 2 clusters the survivors by ``1 - |r|`` with complete linkage and keeps
    the member closest to each cluster centroid.

    Fitted on the training set only -- fitting it on a broader corpus was tried and made no
    difference, while adding a live web dependency.
    """

    def __init__(self, near_constant_frac: float = 0.90, corr_threshold: float = 0.8):
        self.near_constant_frac = float(near_constant_frac)
        self.corr_threshold = float(corr_threshold)
        self.selected_idx: Optional[np.ndarray] = None
        self.selected_names: Optional[List[str]] = None
        self.stage1_kept: int = 0

    def fit(self, X: np.ndarray, names: Sequence[str]) -> "AutoQSARDescriptorSelector":
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        X = np.nan_to_num(np.asarray(X, dtype=float))
        n = max(1, len(X))

        keep1: List[int] = []
        for j in range(X.shape[1]):
            col = X[:, j]
            _vals, counts = np.unique(col, return_counts=True)
            if counts.max() / n < self.near_constant_frac and np.var(col) > 1e-8:
                keep1.append(j)
        if not keep1:
            keep1 = list(range(X.shape[1]))
        X1 = X[:, keep1]
        self.stage1_kept = len(keep1)

        if X1.shape[1] > 1:
            corr = np.nan_to_num(np.corrcoef(X1, rowvar=False), nan=0.0)
            np.fill_diagonal(corr, 1.0)
            dist = 1.0 - np.abs(corr)
            dist = (dist + dist.T) / 2.0  # enforce exact symmetry against float noise
            np.fill_diagonal(dist, 0.0)
            Z = linkage(squareform(dist, checks=False), method="complete")
            labels = fcluster(Z, t=1.0 - self.corr_threshold, criterion="distance")
            keep2: List[int] = []
            for cl in np.unique(labels):
                members = np.flatnonzero(labels == cl)
                if len(members) == 1:
                    keep2.append(int(members[0]))
                    continue
                sub = np.abs(corr[np.ix_(members, members)])
                keep2.append(int(members[np.argmax(sub.mean(axis=1))]))
            keep2 = sorted(keep2)
        else:
            keep2 = list(range(X1.shape[1]))

        self.selected_idx = np.array([keep1[j] for j in keep2], dtype=int)
        self.selected_names = [names[j] for j in self.selected_idx]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_idx is None:
            raise RuntimeError("selector is not fitted")
        return np.nan_to_num(np.asarray(X, dtype=float))[:, self.selected_idx]

    def summary(self, n_input: int) -> str:
        return (
            f"  near-constant filter: {n_input} -> {self.stage1_kept} descriptors\n"
            f"  correlation clustering (r<={self.corr_threshold}): "
            f"{self.stage1_kept} -> {len(self.selected_idx or [])} descriptors"
        )


# ------------------------------------------------------------------------------ featurise


@dataclass
class FeaturisedSet:
    """Molecules, fingerprints, descriptors and keys for one set of compounds."""

    frame: pd.DataFrame
    fingerprints: List = field(default_factory=list)
    descriptors: Optional[np.ndarray] = None
    descriptor_names: List[str] = field(default_factory=list)
    keys: List[str] = field(default_factory=list)
    n_failed: int = 0
    n_duplicates: int = 0

    def __len__(self) -> int:
        return len(self.frame)


def featurise(
    frame: pd.DataFrame,
    *,
    smiles_col: str = "smiles",
    radius: int = 2,
    n_bits: int = 2048,
    standardise: bool = True,
    with_descriptors: bool = False,
    dedupe: bool = True,
    progress: bool = True,
    label: str = "compounds",
) -> FeaturisedSet:
    """Standardise, fingerprint and optionally describe every row of ``frame``.

    Rows that fail to parse are dropped and counted. Deduplication is on the standardised
    InChIKey, keeping the first occurrence.
    """
    Chem = require("rdkit").Chem  # type: ignore[attr-defined]
    std = Standardiser() if standardise else None

    mols, fps, keys, keep = [], [], [], []
    rows = frame[smiles_col].tolist()
    it = tqdm(rows, desc=f"Featurising {label}", unit="mol") if progress else rows
    for smi in it:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi.strip() else None
        if std is not None:
            mol = std(mol)
        if mol is None:
            keep.append(False)
            continue
        try:
            fps.append(morgan_fingerprint(mol, radius, n_bits))
        except Exception:
            keep.append(False)
            continue
        mols.append(mol)
        keys.append(inchikey(mol) or Chem.MolToSmiles(mol))
        keep.append(True)

    mask = np.asarray(keep, dtype=bool)
    out = frame[mask].reset_index(drop=True).copy()
    n_failed = int((~mask).sum())

    n_dup = 0
    if dedupe and len(out):
        out["_key"] = keys
        first = ~out["_key"].duplicated()
        n_dup = int((~first).sum())
        if n_dup:
            fps = [f for f, k in zip(fps, first) if k]
            mols = [m for m, k in zip(mols, first) if k]
            keys = [k for k, f in zip(keys, first) if f]
            out = out[first].reset_index(drop=True)
    elif len(out):
        out["_key"] = keys

    desc, names = None, []
    if with_descriptors and mols:
        desc, names = compute_descriptors(mols, progress=progress)

    return FeaturisedSet(
        frame=out,
        fingerprints=fps,
        descriptors=desc,
        descriptor_names=names,
        keys=keys,
        n_failed=n_failed,
        n_duplicates=n_dup,
    )


def build_design_matrix(
    fingerprints,
    descriptors: Optional[np.ndarray],
    *,
    feature_set: str = "fp_only",
    scaler=None,
) -> np.ndarray:
    """Assemble the model input according to ``feature_set``.

    ``scaler`` is applied to the descriptor block only; fingerprint bits are already on a
    common scale and scaling them destroys the sparsity the tree models exploit.
    """
    if feature_set not in ("fp_only", "descriptors_only", "fp_plus_descriptors"):
        raise ValueError(f"unknown feature_set {feature_set!r}")

    fp_block = (
        fingerprints_to_array(fingerprints).astype(np.float32)
        if feature_set != "descriptors_only"
        else None
    )
    desc_block = None
    if feature_set != "fp_only":
        if descriptors is None:
            raise ValueError(f"feature_set={feature_set!r} needs descriptors")
        desc_block = np.nan_to_num(np.asarray(descriptors, dtype=np.float32))
        if scaler is not None:
            desc_block = scaler.transform(desc_block).astype(np.float32)

    if feature_set == "fp_only":
        return fp_block
    if feature_set == "descriptors_only":
        return desc_block
    return np.hstack([fp_block, desc_block])
