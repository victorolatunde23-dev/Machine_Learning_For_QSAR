"""Reading a compound library, and streaming it past a model in chunks.

Two failure modes shaped this module.

**Silent substitution.** An earlier loader rewrote any path starting with ``/`` into a
relative one unconditionally (a Colab-ism, where ``/file.sdf`` meant the Drive root), then
wrote and loaded an 8-compound demo file when the path missed -- while printing
``Loading: ...`` as though nothing were wrong. A whole run screened 8 molecules. So:
the fallback rewrite only happens when the absolute path is absent *and* the relative one
exists, and a missing library raises rather than substituting anything.

**Memory.** A 290,000 x 2048 float32 feature matrix is 2.4 GB before anything else, so
:func:`stream_and_score` never holds more than ``chunk_size`` molecules and appends results
to disk as it goes.
"""

from __future__ import annotations

import gzip
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._optional import require, tqdm

__all__ = [
    "resolve_library_path",
    "open_text",
    "sniff_columns",
    "load_library",
    "iter_sdf",
    "mol_identifier",
    "library_health_report",
    "LibraryHealth",
]

SMILES_COLUMN_NAMES = (
    "smiles", "smi", "canonical_smiles", "structure", "mol", "smiles_string",
)
ID_COLUMN_NAMES = (
    "compound_id", "id", "idnumber", "catalog_id", "name", "title", "zinc_id",
    "molecule_name", "reg_id",
)
ID_SD_PROPERTIES = (
    "_Name", "idnumber", "ID", "id", "Catalog ID", "catalog_id", "ZINC_ID", "Name",
)


def resolve_library_path(path_str: str) -> Path:
    """Resolve a library path, raising with a useful message when it is missing.

    A leading ``/`` meant "the Drive root" in the notebook versions of this pipeline; on a
    real filesystem it means an absolute path. The relative reading is tried **only** when
    the absolute path does not exist and the relative one does.
    """
    p = Path(path_str)
    if not p.exists() and str(path_str).startswith(("/", "\\")):
        alt = Path("." + str(path_str))
        if alt.exists():
            return alt
    if not p.exists():
        raise FileNotFoundError(
            f"screening library not found: {path_str}\n"
            f"This loader deliberately does not substitute a demo file: a silent fallback "
            f"produces a plausible-looking ranking of the wrong molecules, which is worse "
            f"than stopping."
        )
    return p


def open_text(path: Path):
    """Text handle for a plain or gzipped file."""
    if path.name.lower().endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"))
    return open(path, "r")


def sniff_columns(
    frame: pd.DataFrame,
    *,
    smiles_col: Optional[str] = None,
    id_col: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Find the SMILES and identifier columns, by name then by content."""
    lower = {str(c).strip().lower(): c for c in frame.columns}

    if smiles_col and smiles_col in frame.columns:
        smi = smiles_col
    else:
        smi = next((lower[n] for n in SMILES_COLUMN_NAMES if n in lower), None)
    if smi is None:
        # fall back to the first column that mostly parses as SMILES
        Chem = require("rdkit").Chem  # type: ignore[attr-defined]
        best, best_rate = None, 0.0
        for col in frame.columns:
            sample = frame[col].dropna().astype(str).head(25)
            if sample.empty:
                continue
            rate = float(
                np.mean([Chem.MolFromSmiles(s) is not None for s in sample])
            )
            if rate > best_rate:
                best, best_rate = col, rate
        if best is None or best_rate < 0.5:
            raise ValueError(
                f"could not identify a SMILES column in {list(frame.columns)}; "
                f"pass screen.smiles_col explicitly"
            )
        smi = best

    if id_col and id_col in frame.columns:
        ident = id_col
    else:
        ident = next(
            (lower[n] for n in ID_COLUMN_NAMES if n in lower and lower[n] != smi), None
        )
    return smi, ident


def mol_identifier(mol, index: int, stem: str) -> str:
    """An identifier for an SD record, from ``_Name`` or a known property, else positional."""
    for key in ID_SD_PROPERTIES:
        try:
            if mol.HasProp(key):
                value = mol.GetProp(key).strip()
                if value:
                    return value
        except Exception:
            continue
    return f"{stem}:{index + 1}"


def iter_sdf(path: Path) -> Iterator:
    """Stream an SDF (optionally gzipped) without loading it all into memory."""
    Chem = require("rdkit").Chem  # type: ignore[attr-defined]
    handle = gzip.open(path, "rb") if path.name.lower().endswith(".gz") else open(path, "rb")
    return Chem.ForwardSDMolSupplier(handle)


def load_library(
    path_str: str,
    *,
    smiles_col: Optional[str] = None,
    id_col: Optional[str] = None,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load a whole library into a frame with ``smiles`` and ``compound_id`` columns.

    Accepts ``.sdf``, ``.smi``, ``.csv``, ``.tsv`` and ``.gz`` variants. For a very large
    SDF prefer :func:`stream_and_score`, which never materialises the whole thing.
    """
    path = resolve_library_path(path_str)
    name = path.name.lower()
    if verbose:
        print(f"Loading: {path}")

    if name.endswith((".sdf", ".sd", ".sdf.gz", ".sd.gz")):
        Chem = require("rdkit").Chem  # type: ignore[attr-defined]
        rows, n_fail = [], 0
        it = iter_sdf(path)
        bar = tqdm(it, desc="  reading SDF", unit="mol") if verbose else it
        for i, mol in enumerate(bar):
            if limit and len(rows) >= limit:
                break
            if mol is None:
                n_fail += 1
                continue
            rows.append(
                {
                    "smiles": Chem.MolToSmiles(mol),
                    "compound_id": mol_identifier(mol, i, path.name),
                }
            )
        frame = pd.DataFrame(rows)
        fmt = "SDF"
    elif name.endswith((".smi", ".ism", ".smi.gz", ".ism.gz")):
        rows = []
        with open_text(path) as fh:
            for i, line in enumerate(fh):
                if limit and len(rows) >= limit:
                    break
                parts = line.split()
                if not parts:
                    continue
                rows.append(
                    {
                        "smiles": parts[0],
                        "compound_id": parts[1] if len(parts) > 1 else f"row_{i}",
                    }
                )
        frame, n_fail, fmt = pd.DataFrame(rows), 0, "SMILES list"
    else:
        sep = "\t" if name.endswith((".tsv", ".tsv.gz", ".tab")) else None
        raw = pd.read_csv(
            path, sep=sep, engine="python" if sep is None else "c", nrows=limit
        )
        smi, ident = sniff_columns(raw, smiles_col=smiles_col, id_col=id_col)
        if verbose:
            print(f"  SMILES column: {smi!r}   id column: {ident!r}")
        frame = pd.DataFrame(
            {
                "smiles": raw[smi].astype(str),
                "compound_id": (
                    raw[ident].astype(str)
                    if ident
                    else [f"{path.stem}:{i + 1}" for i in range(len(raw))]
                ),
            }
        )
        n_fail, fmt = 0, "delimited text"

    frame = frame[frame["smiles"].astype(str).str.strip() != ""].reset_index(drop=True)
    if verbose:
        print(
            f"  format: {fmt} | rows loaded: {len(frame):,}"
            + (f" | unparseable records skipped: {n_fail:,}" if n_fail else "")
        )
    return frame


# -------------------------------------------------------------------------- health report


@dataclass
class LibraryHealth:
    """What a library actually contains, before it is trusted."""

    n: int = 0
    n_sampled: int = 0
    n_unparseable: int = 0
    heavy_atom_median: float = float("nan")
    heavy_atom_p10: float = float("nan")
    heavy_atom_p90: float = float("nan")
    mw_median: float = float("nan")
    mw_p10: float = float("nan")
    mw_p90: float = float("nan")
    fragment_like_frac: float = float("nan")
    very_small_frac: float = float("nan")
    very_large_frac: float = float("nan")

    @property
    def looks_druglike(self) -> bool:
        return not (self.fragment_like_frac > 0.10 or self.very_small_frac > 0.05)

    def summary(self) -> str:
        lines = [
            f"LIBRARY HEALTH  (sample of {self.n_sampled:,} of {self.n:,})",
            f"  unparseable in sample : {self.n_unparseable:,}",
            f"  heavy atoms           : median {self.heavy_atom_median:.0f}, "
            f"10th pct {self.heavy_atom_p10:.0f}, 90th pct {self.heavy_atom_p90:.0f}",
            f"  molecular weight      : median {self.mw_median:.0f}, "
            f"10th pct {self.mw_p10:.0f}, 90th pct {self.mw_p90:.0f}",
            f"  fragment-like (<12 heavy atoms) : {100 * self.fragment_like_frac:5.1f}%",
            f"  very small (MW < 150)           : {100 * self.very_small_frac:5.1f}%",
            f"  very large (MW > 700)           : {100 * self.very_large_frac:5.1f}%",
        ]
        if not self.looks_druglike:
            lines += [
                "",
                "  !! This does not look like a drug-like screening collection. A large",
                "     fragment fraction is the signature of a building-block or reagent",
                "     catalogue. Note that ordering a file small-to-large also produces",
                "     this pattern in an unshuffled sample -- check a random sample before",
                "     concluding the file is wrong.",
            ]
        else:
            lines.append("\n  Size profile looks drug-like.")
        return "\n".join(lines)


def library_health_report(
    frame: pd.DataFrame,
    *,
    smiles_col: str = "smiles",
    n_sample: int = 20000,
    random_state: int = 42,
    progress: bool = False,
) -> LibraryHealth:
    """Profile a library's size distribution on a **random** sample.

    The sample is random rather than the head on purpose: a catalogue ordered
    small-to-large will show solvents and reagents in its first rows and look like a
    building-block collection when it is nothing of the kind. That exact mistake was made
    on this project and repeated four times before a random sample disproved it.
    """
    Chem = require("rdkit").Chem  # type: ignore[attr-defined]
    Descriptors = require("rdkit").Chem.Descriptors  # type: ignore[attr-defined]

    n = len(frame)
    sample = (
        frame.sample(n=min(n_sample, n), random_state=random_state)
        if n > n_sample
        else frame
    )
    heavy, mw, n_fail = [], [], 0
    rows = sample[smiles_col].astype(str).tolist()
    it = tqdm(rows, desc="  profiling", unit="mol", leave=False) if progress else rows
    for smi in it:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_fail += 1
            continue
        heavy.append(mol.GetNumHeavyAtoms())
        mw.append(Descriptors.MolWt(mol))

    if not heavy:
        return LibraryHealth(n=n, n_sampled=len(sample), n_unparseable=n_fail)

    heavy_a, mw_a = np.asarray(heavy, dtype=float), np.asarray(mw, dtype=float)
    return LibraryHealth(
        n=n,
        n_sampled=len(sample),
        n_unparseable=n_fail,
        heavy_atom_median=float(np.median(heavy_a)),
        heavy_atom_p10=float(np.percentile(heavy_a, 10)),
        heavy_atom_p90=float(np.percentile(heavy_a, 90)),
        mw_median=float(np.median(mw_a)),
        mw_p10=float(np.percentile(mw_a, 10)),
        mw_p90=float(np.percentile(mw_a, 90)),
        fragment_like_frac=float((heavy_a < 12).mean()),
        very_small_frac=float((mw_a < 150).mean()),
        very_large_frac=float((mw_a > 700).mean()),
    )
