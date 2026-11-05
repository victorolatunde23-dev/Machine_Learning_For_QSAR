"""Streaming a large library past fitted models in bounded memory.

The featuriser is injected rather than imported, which is what makes this testable without
RDKit: a test can pass a trivial ``smiles -> bit vector`` function and exercise the
chunking, deduplication, threshold application and consensus logic for real.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._optional import tqdm

__all__ = ["ScreenResult", "stream_and_score", "iter_smiles_records", "consensus_tiers"]


@dataclass
class ScreenResult:
    """Counters from a streaming pass, plus where the predictions went."""

    path: Optional[Path] = None
    n_read: int = 0
    n_failed: int = 0
    n_duplicates: int = 0
    n_scored: int = 0
    frame: Optional[pd.DataFrame] = None

    def summary(self) -> str:
        pct = 100 * self.n_duplicates / max(self.n_read, 1)
        lines = [
            f"  records read            : {self.n_read:,}",
            f"  unparseable / failed    : {self.n_failed:,}",
            f"  duplicate structures    : {self.n_duplicates:,} ({pct:.1f}% of the file)",
            f"  unique compounds scored : {self.n_scored:,}",
        ]
        if self.path:
            lines.append(f"  written to              : {self.path}")
        return "\n".join(lines)


def iter_smiles_records(
    frame: pd.DataFrame, *, smiles_col: str = "smiles", id_col: str = "compound_id"
) -> Iterator[Tuple[str, str]]:
    """``(identifier, smiles)`` pairs from an in-memory frame."""
    ids = (
        frame[id_col].astype(str)
        if id_col in frame.columns
        else pd.Series([f"row_{i}" for i in range(len(frame))])
    )
    for ident, smi in zip(ids, frame[smiles_col].astype(str)):
        yield ident, smi


def stream_and_score(
    records: Iterable[Tuple[str, str]],
    *,
    featurise_chunk: Callable[[Sequence[str]], Tuple[np.ndarray, List[int], List[object]]],
    score_chunk: Callable[[np.ndarray, Sequence[object]], Dict[str, np.ndarray]],
    thresholds: Dict[str, float],
    chunk_size: int = 25000,
    limit: Optional[int] = None,
    output_csv: Optional[Path] = None,
    key_fn: Optional[Callable[[object], Optional[str]]] = None,
    training_keys: Optional[set] = None,
    extra_columns: Optional[Callable[[Sequence[object]], Dict[str, Sequence]]] = None,
    progress: bool = True,
    return_frame: bool = True,
) -> ScreenResult:
    """Score an arbitrarily large stream of ``(id, smiles)`` in bounded memory.

    Parameters
    ----------
    featurise_chunk
        ``smiles_list -> (X, kept_indices, carriers)``. ``carriers`` is whatever the
        scorer needs per surviving molecule (fingerprints, say) and is passed straight
        through; ``kept_indices`` indexes into the input list so identifiers stay aligned.
    score_chunk
        ``(X, carriers) -> {model_name: P(active)}``. Already calibrated.
    thresholds
        Per-model decision cut. Each model contributes a ``call_<model>`` column and the
        count of models calling active becomes ``n_models_active``.
    key_fn
        ``carrier -> deduplication key`` (an InChIKey, usually). Omit to skip dedup.
    training_keys
        Keys present in the training set. Matching compounds are *flagged*
        ``in_training``, not dropped: they are a free positive control, and where they rank
        is the first thing to check in any screen.
    """
    out_path = Path(output_csv) if output_csv else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()

    seen: set = set()
    result = ScreenResult(path=out_path)
    model_names = list(thresholds)
    buf_ids: List[str] = []
    buf_smiles: List[str] = []
    collected: List[pd.DataFrame] = []
    first_write = True

    def flush() -> None:
        nonlocal first_write
        if not buf_smiles:
            return
        X, kept, carriers = featurise_chunk(buf_smiles)
        result.n_failed += len(buf_smiles) - len(kept)
        if len(kept) == 0:
            buf_ids.clear()
            buf_smiles.clear()
            return

        ids = [buf_ids[i] for i in kept]
        smiles = [buf_smiles[i] for i in kept]

        keys: Optional[List[Optional[str]]] = None
        if key_fn is not None:
            keys = [key_fn(c) for c in carriers]
            fresh = []
            for i, k in enumerate(keys):
                if k is not None and k in seen:
                    result.n_duplicates += 1
                    continue
                if k is not None:
                    seen.add(k)
                fresh.append(i)
            if len(fresh) != len(kept):
                X = X[fresh]
                ids = [ids[i] for i in fresh]
                smiles = [smiles[i] for i in fresh]
                carriers = [carriers[i] for i in fresh]
                keys = [keys[i] for i in fresh]
            if not ids:
                buf_ids.clear()
                buf_smiles.clear()
                return

        frame = pd.DataFrame({"compound_id": ids, "smiles": smiles})
        if keys is not None:
            frame["inchikey"] = keys
            frame["in_training"] = [
                bool(training_keys and k in training_keys) for k in keys
            ]
        else:
            frame["in_training"] = False

        scores = score_chunk(X, carriers)
        call_cols: List[str] = []
        for name in model_names:
            p = np.asarray(scores[name], dtype=float)
            col = _safe_column(name)
            frame[f"p_{col}"] = np.round(p, 5)
            frame[f"call_{col}"] = (p >= thresholds[name]).astype(int)
            call_cols.append(f"call_{col}")
        frame["n_models_active"] = frame[call_cols].sum(axis=1)
        frame["p_mean"] = frame[[f"p_{_safe_column(n)}" for n in model_names]].mean(
            axis=1
        ).round(5)

        if extra_columns is not None:
            for key, values in extra_columns(carriers).items():
                frame[key] = values

        if out_path is not None:
            frame.to_csv(out_path, mode="a", header=first_write, index=False)
            first_write = False
        if return_frame:
            collected.append(frame)
        result.n_scored += len(frame)
        buf_ids.clear()
        buf_smiles.clear()

    bar = tqdm(desc="Screening", unit="mol") if progress else None
    for ident, smi in records:
        if limit is not None and result.n_read >= limit:
            break
        result.n_read += 1
        if bar is not None:
            bar.update(1)
        buf_ids.append(str(ident))
        buf_smiles.append(str(smi))
        if len(buf_smiles) >= chunk_size:
            flush()
            if bar is not None:
                bar.set_postfix(scored=f"{result.n_scored:,}")
    flush()
    if bar is not None:
        bar.close()

    if return_frame:
        result.frame = (
            pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()
        )
    return result


def _safe_column(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum() or ch == "_")


def consensus_tiers(frame: pd.DataFrame, n_models: int) -> pd.Series:
    """Counts of compounds called by all models, by some, and by none.

    The all-models tier is the defensible list. The some-models tier is where two
    different inductive biases disagree, and the size of that tier is itself a measure of
    how far outside its training distribution the library sits.
    """
    counts = frame["n_models_active"].value_counts()
    index = list(range(n_models, -1, -1))
    return counts.reindex(index).fillna(0).astype(int)
