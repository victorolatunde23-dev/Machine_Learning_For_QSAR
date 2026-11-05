"""Wiring a fitted :class:`~qsar_tl.pipeline.Pipeline` to the streaming scorer.

Separate from :mod:`qsar_tl.screen.stream` so that the streaming logic stays
dependency-free and testable, while this module is allowed to need RDKit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._optional import require
from ..models.registry import predict_proba
from .library import iter_sdf, load_library, mol_identifier, resolve_library_path
from .stream import ScreenResult, consensus_tiers, iter_smiles_records, stream_and_score

__all__ = ["screen_library", "shortlist", "positive_control_report", "agreement_report"]


def _sdf_records(path: Path):
    """``(identifier, smiles)`` from an SDF without loading it all."""
    Chem = require("rdkit").Chem  # type: ignore[attr-defined]
    for i, mol in enumerate(iter_sdf(path)):
        if mol is None:
            # Counted as a failure downstream; yielding a sentinel keeps the read count
            # honest, which matters because the duplicate percentage is reported over it.
            yield f"{path.name}:{i + 1}", ""
            continue
        yield mol_identifier(mol, i, path.name), Chem.MolToSmiles(mol)


def screen_library(pipeline, *, library_path: Optional[str] = None,
                   output_csv: Optional[Path] = None) -> ScreenResult:
    """Score a library with the pipeline's final models, streamed in chunks.

    Uses the thresholds and calibrators chosen during evaluation, so the ``call_*`` columns
    mean the same thing they meant in the out-of-fold estimate.
    """
    from ..features.molecules import (
        Standardiser,
        build_design_matrix,
        inchikey,
        morgan_fingerprint,
    )

    cfg = pipeline.config
    state = pipeline.state
    if not state.final_models:
        raise RuntimeError("fit the models first (Pipeline.fit_final or .run)")

    path = resolve_library_path(library_path or cfg.screen.library_path)
    standardiser = Standardiser() if cfg.features.standardise else None
    Chem = require("rdkit").Chem  # type: ignore[attr-defined]
    needs_descriptors = cfg.features.feature_set != "fp_only"

    # Each surviving molecule is carried as (fingerprint, inchikey): the scorer needs the
    # first and the deduplicator the second, and keeping them paired is what stops the two
    # lists drifting out of alignment when a chunk loses rows.
    def featurise_chunk(smiles_list: Sequence[str]):
        kept: List[int] = []
        carriers: List[Tuple[object, Optional[str]]] = []
        mols: List[object] = []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi.strip() else None
            if standardiser is not None:
                mol = standardiser(mol)
            if mol is None:
                continue
            try:
                fp = morgan_fingerprint(
                    mol, cfg.features.morgan_radius, cfg.features.morgan_n_bits
                )
            except Exception:
                continue
            key = inchikey(mol) if cfg.labels.dedupe_on_inchikey else None
            carriers.append((fp, key))
            mols.append(mol)
            kept.append(i)
        if not kept:
            return np.zeros((0, 0)), [], []
        descriptors = None
        if needs_descriptors:
            from ..features.molecules import compute_descriptors

            raw, _names = compute_descriptors(mols)
            descriptors = state.descriptor_selector.transform(raw)
        X = build_design_matrix(
            [c[0] for c in carriers], descriptors,
            feature_set=cfg.features.feature_set,
            scaler=state.descriptor_scaler if needs_descriptors else None,
        )
        return X, kept, carriers

    def score_chunk(X, carriers) -> Dict[str, np.ndarray]:
        fps = [c[0] for c in carriers]
        out: Dict[str, np.ndarray] = {}
        for name, model in state.final_models.items():
            p = predict_proba(
                name, model, X, fps=fps, chunk=cfg.screen.chunk_size
            )[:, 1]
            calibrator = state.calibrators.get(name)
            if calibrator is not None:
                p = calibrator.transform(p)
            out[name] = p
        return out

    thresholds = {
        name: state.thresholds.get(name, 0.5) for name in state.final_models
    }
    records = (
        _sdf_records(path)
        if path.name.lower().endswith((".sdf", ".sd", ".sdf.gz", ".sd.gz"))
        else iter_smiles_records(
            load_library(
                str(path), smiles_col=cfg.screen.smiles_col, id_col=cfg.screen.id_col,
                limit=cfg.screen.limit, verbose=pipeline.verbose,
            )
        )
    )

    out_csv = output_csv or (Path(cfg.output_dir) / "library_predictions.csv")
    result = stream_and_score(
        records,
        featurise_chunk=featurise_chunk,
        score_chunk=score_chunk,
        thresholds=thresholds,
        chunk_size=cfg.screen.chunk_size,
        limit=cfg.screen.limit,
        output_csv=out_csv,
        key_fn=(lambda carrier: carrier[1]) if cfg.labels.dedupe_on_inchikey else None,
        training_keys=state.training_keys,
        progress=pipeline.verbose,
    )
    return result


def positive_control_report(frame: pd.DataFrame, *, score_col: str = "p_mean") -> str:
    """Where compounds that are also in the training set rank.

    The first thing to read in any screen. If known answers do not come out near the top,
    nothing further down is worth reading, and that check costs nothing because those
    compounds came through the same pipeline as everything else.
    """
    if "in_training" not in frame.columns:
        return "No in_training column: the positive control was not computed."
    known = frame[frame["in_training"]]
    if known.empty:
        return (
            "No library compound matches a training structure, so there is no internal\n"
            "control on this ranking. Note that a full-InChIKey comparison can miss a\n"
            "match that differs only in stereochemistry or protonation; compare the\n"
            "14-character connectivity block too before concluding there is no overlap."
        )
    ranks = frame[score_col].rank(ascending=False)
    pct = 100 * (1 - ranks[frame["in_training"]] / len(frame))
    median = float(pct.median())
    lines = [
        f"{len(known)} library compound(s) are also in the training set.",
        f"  median percentile: {median:.1f}",
    ]
    if median < 90:
        lines.append(
            "  !! The known compounds do NOT rank near the top. Treat the rest of this "
            "ranking as unsupported until that is understood."
        )
    else:
        lines.append(
            "  They rank near the top, which is the minimum the ranking has to achieve "
            "to be worth reading."
        )
    return "\n".join(lines)


def agreement_report(frame: pd.DataFrame, model_names: Sequence[str]) -> str:
    """How often the models agree, and what the disagreement means."""
    cols = [
        c for c in frame.columns if c.startswith("call_")
    ]
    if len(cols) < 2:
        return "Fewer than two models: no agreement to report."
    tiers = consensus_tiers(frame, len(cols))
    lines = [
        f"  all models active : {tiers.get(len(cols), 0):,}   <- the defensible list",
        f"  some but not all  : {sum(tiers.get(k, 0) for k in range(1, len(cols))):,}",
        f"  none              : {tiers.get(0, 0):,}",
    ]
    if len(cols) == 2:
        a, b = cols
        both = int(((frame[a] == 1) & (frame[b] == 1)).sum())
        either = int(((frame[a] == 1) | (frame[b] == 1)).sum())
        lines.append(f"  Jaccard overlap of the two active sets: {both / max(either, 1):.3f}")
        p_cols = [c for c in frame.columns if c.startswith("p_") and c != "p_mean"]
        if len(p_cols) >= 2:
            rho = frame[p_cols[0]].corr(frame[p_cols[1]], method="spearman")
            lines.append(f"  Spearman correlation of the score columns: {rho:.3f}")
        lines.append(
            "  Two models with different inductive biases agreeing is a weak but real\n"
            "  signal. A low overlap means most single-model calls are an artefact of one\n"
            "  model's bias rather than a property of the molecule."
        )
    return "\n".join(lines)


def shortlist(frame: pd.DataFrame, n: int = 500, *, model_names: Optional[Sequence[str]] = None
              ) -> pd.DataFrame:
    """Top ``n`` compounds, consensus first, then by mean percentile rank.

    Deliberately **not** ranked by mean probability. The models have different thresholds
    and different probability scales, so averaging the raw numbers is the same blending
    that lost to its best single member on seven consecutive endpoints -- and it buries the
    compounds both models agree on among single-model calls. Percentile ranks are
    scale-free, so averaging them is an ordering rule over independent opinions rather than
    a combined predictor.
    """
    out = frame.copy()
    p_cols = [c for c in out.columns if c.startswith("p_") and c != "p_mean"]
    if not p_cols:
        raise ValueError("no per-model probability columns found")
    for c in p_cols:
        out[f"pct_{c[2:]}"] = out[c].rank(pct=True)
    pct_cols = [f"pct_{c[2:]}" for c in p_cols]
    out["rank_pct_mean"] = out[pct_cols].mean(axis=1).round(5)
    sort_cols = (
        ["n_models_active", "rank_pct_mean"]
        if "n_models_active" in out.columns
        else ["rank_pct_mean"]
    )
    return out.sort_values(sort_cols, ascending=False).head(n).reset_index(drop=True)
