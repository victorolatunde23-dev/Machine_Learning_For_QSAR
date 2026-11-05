"""End-to-end run on synthetic measurements, with no network and no BindingDB.

    python examples/quickstart.py

Needs the `chem` extra for the featurisation step; everything before it runs anywhere.
The point is to show the stage-by-stage shape of a run and what each stage prints.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from qsar_tl import Config, Pipeline, have


def synthetic_measurements(n: int = 400, seed: int = 0) -> tuple:
    """Measurements with the structure the pipeline cares about.

    Includes a fixed-offset block, so the derived-value detector has something to find,
    and censored bounds, so the recovery path runs.
    """
    rng = np.random.RandomState(seed)
    # A congeneric series with genuinely distinct members. Getting this wrong is
    # instructive: an earlier version of this generator cycled 16 structures, the
    # replicate aggregation collapsed 344 rows onto 18 compounds, and the pipeline's
    # sanity gate stopped the run and said so.
    substituents = ["", "F", "Cl", "Br", "I", "OC", "C(F)(F)F", "C#N", "N(C)C",
                    "S(N)(=O)=O", "C(=O)O", "O"]
    tails = ["N", "NC", "NCC", "O"]

    def build(index: int) -> str:
        sub = substituents[index % len(substituents)]
        chain = 1 + (index // len(substituents)) % 9
        tail = tails[(index // (len(substituents) * 9)) % len(tails)]
        ring = f"c1ccc({sub})cc1" if sub else "c1ccccc1"
        return f"{ring}C(=O)N{'C' * chain}{tail}"

    long_rows, wide_rows = [], []
    for i in range(n):
        smiles = build(i)
        name = f"US20250213705, Compound {i}" if i < n // 4 else f"CHEMBL{900000 + i}"

        p_ic50 = float(np.clip(rng.normal(7.3, 1.3), 3.0, 10.5))
        long_rows.append(
            {"smiles": smiles, "ligand_name": name,
             "affinity_raw": f"{10 ** (9 - p_ic50):.6g}", "endpoint_name": "pIC50"}
        )
        row = {"smiles": smiles, "ligand_name": name, "pIC50": p_ic50, "pKi": np.nan}

        if i < n // 4:
            p_ki = p_ic50 + np.log10(140.0)       # the derived block
        elif rng.rand() < 0.8:
            p_ki = p_ic50 + rng.normal(0.1, 0.6)  # independently measured
        else:
            p_ki = None
        if p_ki is not None:
            long_rows.append(
                {"smiles": smiles, "ligand_name": name,
                 "affinity_raw": f"{10 ** (9 - p_ki):.6g}", "endpoint_name": "pKi"}
            )
            row["pKi"] = p_ki
        wide_rows.append(row)

    for j in range(60):                            # censored negatives, all distinct
        long_rows.append(
            {"smiles": f"CC(=O)Nc1ccc({'C' * (1 + j % 5)}{'O' if j % 2 else 'N'})cc1"
                       f"{'C' * (j // 10)}",
             "ligand_name": f"cens_{j}", "affinity_raw": ">100000",
             "endpoint_name": "pKi"}
        )
    return pd.DataFrame(long_rows), pd.DataFrame(wide_rows)


def main() -> int:
    if not have("rdkit"):
        print(
            "This example needs rdkit for the featurisation step:\n"
            "  pip install 'qsar-tl[chem]'\n\n"
            "The stages before it (label assembly, derived-value detection, censored\n"
            "recovery) run without it -- see tests/test_pipeline.py."
        )
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="qsar-tl-quickstart-"))
    cfg = Config.from_dict(
        {
            "run_name": "quickstart",
            "output_dir": str(tmp / "out"),
            "data_dir": str(tmp / "data"),
            "labels": {
                "modelled_endpoints": ["pKi"],
                "primary_endpoint": "pKi",
                "derived_endpoint": "pKi",
                "keep_unlabelled_for_domain": True,
            },
            # sklearn-only, so the example runs without LightGBM
            "models": {
                "enabled": ["RandomForest", "LogReg (L2)"],
                "floors": ["LogReg (L2)"],
                "retune": False,
                "params": {"RandomForest": {"n_estimators": 200}},
            },
            "evaluation": {
                "loco_min_cluster_size": 15,
                "loco_min_fold_actives": 3,
                "loco_n_remainder_folds": 2,
                "loco_min_fold_n": 5,
                "cv_folds": 3,
            },
            "domain": {"enabled": False},
            "screen": {"source": "none"},
        }
    )

    long, wide = synthetic_measurements()
    pipe = Pipeline(cfg)

    print("\n" + "=" * 78 + "\nSTAGE 1  labels\n" + "=" * 78)
    pipe.load_labels(long, wide)

    print("\n" + "=" * 78 + "\nSTAGE 2  features\n" + "=" * 78)
    pipe.featurise()

    print("\n" + "=" * 78 + "\nSTAGE 3  folds\n" + "=" * 78)
    pipe.build_folds()

    print("\n" + "=" * 78 + "\nSTAGE 4  tune + evaluate\n" + "=" * 78)
    pipe.tune()
    pipe.evaluate()

    print("\n" + "=" * 78 + "\nSTAGE 5  deploy\n" + "=" * 78)
    pipe.fit_final()
    pipe.save()

    print(
        "\nThese numbers are about randomly generated affinities, so they mean nothing.\n"
        "What the example shows is the shape of a run and what each stage reports.\n"
        f"Artefacts: {cfg.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
