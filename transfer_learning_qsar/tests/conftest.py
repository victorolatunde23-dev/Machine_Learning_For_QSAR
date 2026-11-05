"""Shared fixtures.

Deliberately no RDKit: the fixtures below build a synthetic dataset with the structure the
pipeline cares about (fixed-offset derived values, censored bounds, a dominant source,
chemical families) out of numpy and pandas alone, so the data-cleaning and evaluation
logic is exercised everywhere.
"""

import numpy as np
import pandas as pd
import pytest

pytest_plugins = []


@pytest.fixture
def rng():
    return np.random.RandomState(13)


@pytest.fixture
def long_frame(rng):
    """A measurement frame with every feature the label assembly has to handle.

    - 60 compounds from one patent whose pKi is pIC50 + log10(140): a derived block
    - 20 from a second patent at a different fixed offset: a second block
    - 60 with genuinely independent pKi
    - 60 measured only by IC50
    - 60 measured only by Ki
    - censored '>' and '<' bounds, some informative and some straddling the cut
    """
    rows = []
    for i in range(260):
        smi = "C" * (3 + i % 25) + "N" + f"x{i}"
        if i < 60:
            kind, name = "derived140", f"US20250213705, Compound {i}"
        elif i < 80:
            kind, name = "derived5", f"US10736974, Example {i}"
        elif i < 140:
            kind, name = "independent", f"CHEMBL{100000 + i}"
        elif i < 200:
            kind, name = "ic50_only", f"CHEMBL{200000 + i}"
        else:
            kind, name = "ki_only", f"ligand_{i}"

        pic = float(np.clip(rng.normal(7.4, 1.4), 3.0, 10.5))
        if kind != "ki_only":
            rows.append(
                {"smiles": smi, "ligand_name": name,
                 "affinity_raw": f"{10 ** (9 - pic):.6g}", "endpoint_name": "pIC50"}
            )
        if kind == "derived140":
            pki = pic + np.log10(140.0)
        elif kind == "derived5":
            pki = pic + np.log10(5.0)
        elif kind == "independent":
            pki = pic + rng.normal(0.0, 0.6)
        elif kind == "ki_only":
            pki = float(np.clip(rng.normal(7.9, 1.3), 3.0, 11.0))
        else:
            pki = None
        if pki is not None:
            rows.append(
                {"smiles": smi, "ligand_name": name,
                 "affinity_raw": f"{10 ** (9 - pki):.6g}", "endpoint_name": "pKi"}
            )

    # censored: '>' far above the cut (provably inactive), '<' far below (provably active),
    # and one of each that straddles it (uninformative)
    for j in range(20):
        rows.append(
            {"smiles": f"N{'C' * (2 + j)}Oc{j}", "ligand_name": f"censored_hi_{j}",
             "affinity_raw": ">100000", "endpoint_name": "pKi"}
        )
    for j in range(6):
        rows.append(
            {"smiles": f"O{'C' * (2 + j)}Nd{j}", "ligand_name": f"censored_lo_{j}",
             "affinity_raw": "<0.5", "endpoint_name": "pKi"}
        )
    rows.append(
        {"smiles": "CCCNOstraddle", "ligand_name": "straddle",
         "affinity_raw": ">0.5", "endpoint_name": "pKi"}
    )
    return pd.DataFrame(rows)


@pytest.fixture
def wide_frame(long_frame):
    """Pivot of the unqualified measurements, as the derived-value detector expects."""
    from qsar_tl.data.labels import source_tag

    frame = long_frame.copy()
    frame["qualifier"] = frame["affinity_raw"].astype(str).str.extract(r"^\s*([<>]=?)")[0]
    frame = frame[frame["qualifier"].isna()]
    frame["value_nm"] = pd.to_numeric(frame["affinity_raw"], errors="coerce")
    frame["p"] = 9 - np.log10(frame["value_nm"])
    wide = (
        frame.pivot_table(index="smiles", columns="endpoint_name", values="p",
                          aggfunc="median")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    names = frame.groupby("smiles")["ligand_name"].first()
    wide["ligand_name"] = wide["smiles"].map(names)
    wide["source_tag"] = wide["ligand_name"].map(source_tag)
    return wide


@pytest.fixture
def binary_problem(rng):
    """A separable-ish binary problem with cluster structure, as numeric features.

    Six clusters of unequal size, with the label partly determined by cluster, so grouped
    and random splits give measurably different answers.
    """
    sizes = [90, 60, 40, 25, 15, 10]
    X, y, groups = [], [], []
    for cid, n in enumerate(sizes):
        centre = rng.normal(0, 2.5, size=12)
        X.append(centre + rng.normal(0, 0.8, size=(n, 12)))
        base = 0.75 if cid % 2 == 0 else 0.3
        y.append((rng.rand(n) < base).astype(int))
        groups.append(np.full(n, cid))
    return (
        np.vstack(X).astype(np.float32),
        np.concatenate(y),
        np.concatenate(groups),
    )


@pytest.fixture
def scores_and_labels(rng):
    """Probabilities with real but imperfect signal, for threshold and calibration tests."""
    n = 400
    y = (rng.rand(n) < 0.45).astype(int)
    p = np.clip(0.5 + 0.22 * (y * 2 - 1) + rng.normal(0, 0.18, n), 0.01, 0.99)
    return p, y


@pytest.fixture
def have_rdkit():
    try:
        import rdkit  # noqa: F401
        return True
    except ImportError:
        return False
