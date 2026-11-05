"""Pipeline wiring up to the point where a chemistry toolkit becomes necessary.

Label loading and the sanity gate need no RDKit, and the gate is the thing most worth
testing: a run was once lost because an upstream failure surfaced six stages later as an
unrelated sklearn error about fold counts.
"""

import numpy as np
import pandas as pd
import pytest

from qsar_tl import Config, InsufficientDataError, Pipeline


def _config(tmp_path, **patch):
    base = {
        "run_name": "test",
        "output_dir": str(tmp_path / "out"),
        "data_dir": str(tmp_path / "data"),
        "labels": {
            "modelled_endpoints": ["pKi"],
            "primary_endpoint": "pKi",
            "derived_endpoint": "pKi",
        },
        "models": {"enabled": ["RandomForest", "LogReg (L2)"],
                   "floors": ["LogReg (L2)"]},
        "domain": {"enabled": False},
    }
    base.update(patch)
    return Config.from_dict(base)


def test_load_labels_runs_the_detector_and_assembles(tmp_path, long_frame, wide_frame):
    pipe = Pipeline(_config(tmp_path), verbose=False)
    table = pipe.load_labels(long_frame, wide_frame)
    assert "y_pKi" in table.columns
    assert pipe.state.derived_report is not None
    assert pipe.state.derived_report.n_flagged > 0
    # the derived compounds must not carry a pKi label
    derived = set(wide_frame.loc[pipe.state.derived_report.mask, "smiles"])
    labelled = set(table.loc[table["y_pKi"].notna(), "smiles"])
    assert not (labelled & derived)


def test_sanity_gate_passes_on_adequate_data(tmp_path, long_frame, wide_frame):
    pipe = Pipeline(_config(tmp_path), verbose=False)
    table = pipe.load_labels(long_frame, wide_frame)
    y = table["y_pKi"].dropna()
    assert len(y) >= 50
    assert min((y == 1).sum(), (y == 0).sum()) >= 10


def test_sanity_gate_stops_with_a_diagnosis_on_a_tiny_set(tmp_path):
    """The failure mode: three compounds, and an error six stages later about folds."""
    long = pd.DataFrame(
        [
            {"smiles": f"CCO{i}", "ligand_name": f"s{i}", "affinity_raw": "50",
             "endpoint_name": "pKi"}
            for i in range(3)
        ]
    )
    wide = pd.DataFrame(
        [{"smiles": f"CCO{i}", "ligand_name": f"s{i}", "pIC50": np.nan, "pKi": 7.3}
         for i in range(3)]
    )
    pipe = Pipeline(_config(tmp_path), verbose=False)
    with pytest.raises(InsufficientDataError) as exc:
        pipe.load_labels(long, wide)
    message = str(exc.value)
    assert "3 labelled compounds" in message
    assert "SYNTHETIC" in message or "synthetic" in message
    assert "active_thresholds" in message


def test_sanity_gate_catches_a_threshold_outside_the_data(tmp_path, long_frame,
                                                          wide_frame):
    cfg = _config(tmp_path)
    cfg.labels.active_thresholds["pKi"] = 20.0   # nothing can be active
    pipe = Pipeline(cfg, verbose=False)
    with pytest.raises(InsufficientDataError):
        pipe.load_labels(long_frame, wide_frame)


def test_derived_removal_can_be_switched_off(tmp_path, long_frame, wide_frame):
    cfg = _config(tmp_path)
    cfg.labels.drop_derived_endpoint_values = False
    pipe = Pipeline(cfg, verbose=False)
    table = pipe.load_labels(long_frame, wide_frame)
    with_derived = int(table["y_pKi"].notna().sum())

    pipe2 = Pipeline(_config(tmp_path), verbose=False)
    without = int(pipe2.load_labels(long_frame, wide_frame)["y_pKi"].notna().sum())
    assert with_derived > without


def test_keep_unlabelled_carries_extra_compounds(tmp_path, long_frame, wide_frame):
    cfg = _config(tmp_path)
    cfg.labels.keep_unlabelled_for_domain = True
    pipe = Pipeline(cfg, verbose=False)
    table = pipe.load_labels(long_frame, wide_frame)
    assert len(table) > int(table["y_pKi"].notna().sum())


def test_output_directories_are_created(tmp_path):
    cfg = _config(tmp_path)
    Pipeline(cfg, verbose=False)
    assert (tmp_path / "out").is_dir()
    assert (tmp_path / "data").is_dir()


def test_pipeline_rejects_an_invalid_config(tmp_path):
    cfg = _config(tmp_path)
    cfg.features.feature_set = "nonsense"
    from qsar_tl.config import ConfigError

    with pytest.raises(ConfigError):
        Pipeline(cfg, verbose=False)


def test_featurise_without_rdkit_names_the_extra(tmp_path, long_frame, wide_frame,
                                                  have_rdkit):
    """The promise of the optional-dependency layer: a useful error, not a bare import."""
    if have_rdkit:
        pytest.skip("rdkit is installed")
    pipe = Pipeline(_config(tmp_path), verbose=False)
    pipe.load_labels(long_frame, wide_frame)
    from qsar_tl import OptionalDependencyError

    with pytest.raises(OptionalDependencyError, match="chem"):
        pipe.featurise()


def test_csv_source_round_trip(tmp_path, long_frame):
    """A local CSV needs neither duckdb nor the network."""
    from qsar_tl.data.bindingdb import fetch_bindingdb

    path = tmp_path / "m.csv"
    long_frame.to_csv(path, index=False)
    cfg = _config(tmp_path)
    cfg.data.source = "csv"
    cfg.data.csv_path = str(path)
    long, wide = fetch_bindingdb(cfg, verbose=False)
    assert len(long) == len(long_frame)
    assert "pKi" in wide.columns


def test_csv_source_requires_a_path(tmp_path):
    from qsar_tl.data.bindingdb import fetch_bindingdb

    cfg = _config(tmp_path)
    cfg.data.source = "csv"
    with pytest.raises(ValueError, match="csv_path"):
        fetch_bindingdb(cfg, verbose=False)
