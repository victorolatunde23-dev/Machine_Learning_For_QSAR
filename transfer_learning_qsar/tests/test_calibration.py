"""Calibration: the maps, the skip rule, and the pooling caveat."""

import numpy as np
import pytest

from qsar_tl.evaluation.calibration import (
    Calibrator,
    calibration_caveat,
    cross_fitted_calibration,
    fit_calibrator,
)
from qsar_tl.evaluation.metrics import expected_calibration_error


def test_identity_calibrator_passes_scores_through():
    cal = Calibrator("identity", None)
    p = np.array([0.1, 0.5, 0.9])
    assert cal.transform(p) == pytest.approx(p)
    assert cal.slope is None


def test_identity_clips_out_of_range_input():
    cal = Calibrator("identity", None)
    assert cal.transform(np.array([-0.2, 1.4])) == pytest.approx([0.0, 1.0])


def test_platt_improves_a_badly_calibrated_predictor():
    rng = np.random.RandomState(0)
    n = 2000
    y = (rng.rand(n) < 0.3).astype(int)
    # systematically overconfident: squash true probabilities toward the extremes
    p = np.clip(0.5 + 0.45 * (y * 2 - 1) + rng.normal(0, 0.15, n), 0.01, 0.99)
    before = expected_calibration_error(y, p)
    cal = fit_calibrator(p, y, "sigmoid")
    after = expected_calibration_error(y, cal.transform(p))
    assert after < before


def test_platt_preserves_ranking():
    rng = np.random.RandomState(1)
    p = rng.uniform(0.05, 0.95, 500)
    y = (rng.rand(500) < p).astype(int)
    cal = fit_calibrator(p, y, "sigmoid")
    out = cal.transform(p)
    assert np.corrcoef(np.argsort(p), np.argsort(out))[0, 1] > 0.99
    assert cal.slope > 0


def test_isotonic_runs_and_stays_in_range():
    rng = np.random.RandomState(2)
    p = rng.uniform(0, 1, 400)
    y = (rng.rand(400) < p).astype(int)
    out = fit_calibrator(p, y, "isotonic").transform(p)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_single_class_input_yields_an_identity_map():
    """Nothing to calibrate against, so the map must not invent a transformation."""
    cal = fit_calibrator(np.linspace(0, 1, 20), np.zeros(20, dtype=int), "sigmoid")
    assert cal.method == "identity"


def test_none_method_is_an_identity_map():
    assert fit_calibrator(np.array([0.5]), np.array([1]), "none").method == "identity"


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown calibration"):
        fit_calibrator(np.array([0.5, 0.6]), np.array([0, 1]), "beta")


def test_platt_export_is_plain_data_with_a_formula():
    """A pickled calibrator only reloads where its class exists; this is the portable copy."""
    rng = np.random.RandomState(3)
    p = rng.uniform(0, 1, 300)
    y = (rng.rand(300) < p).astype(int)
    d = fit_calibrator(p, y, "sigmoid").to_dict()
    assert d["method"] == "sigmoid"
    assert "coef" in d and "intercept" in d
    assert "exp" in d["apply"]


def test_isotonic_export_carries_the_knots():
    rng = np.random.RandomState(4)
    p = rng.uniform(0, 1, 300)
    y = (rng.rand(300) < p).astype(int)
    d = fit_calibrator(p, y, "isotonic").to_dict()
    assert len(d["x"]) == len(d["y"])
    assert "interp" in d["apply"]


def test_exported_platt_parameters_reproduce_the_transform():
    rng = np.random.RandomState(5)
    p = rng.uniform(0, 1, 300)
    y = (rng.rand(300) < p).astype(int)
    cal = fit_calibrator(p, y, "sigmoid")
    d = cal.to_dict()
    manual = 1.0 / (1.0 + np.exp(-(d["coef"] * p + d["intercept"])))
    assert cal.transform(p) == pytest.approx(manual, abs=1e-9)


def test_well_calibrated_models_are_left_alone():
    """Recalibrating an already-calibrated model once made a blend worse, 0.084 -> 0.167."""
    rng = np.random.RandomState(6)
    n = 4000
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.rand(n) < p).astype(int)
    cals, reports = cross_fitted_calibration({"good": p}, y, min_ece=0.10)
    assert cals["good"].method == "identity"
    assert reports[0].calibrated is False
    assert "already calibrated" in reports[0].reason


def test_badly_calibrated_models_are_recalibrated():
    rng = np.random.RandomState(7)
    n = 2000
    y = (rng.rand(n) < 0.3).astype(int)
    p = np.clip(0.5 + 0.45 * (y * 2 - 1) + rng.normal(0, 0.15, n), 0.01, 0.99)
    cals, reports = cross_fitted_calibration({"bad": p}, y, min_ece=0.05)
    assert reports[0].calibrated is True
    assert reports[0].ece_calibrated < reports[0].ece_raw
    assert cals["bad"].method == "sigmoid"


def test_calibration_disabled_leaves_everything_untouched():
    rng = np.random.RandomState(8)
    p = rng.uniform(0, 1, 200)
    y = (rng.rand(200) < 0.5).astype(int)
    cals, reports = cross_fitted_calibration({"m": p}, y, method="none")
    assert cals["m"].method == "identity"
    assert "disabled" in reports[0].reason


def test_nan_scores_are_excluded(binary_problem):
    """Out-of-fold columns have NaN wherever a fold could not be scored."""
    rng = np.random.RandomState(9)
    n = 500
    y = (rng.rand(n) < 0.4).astype(int)
    p = np.clip(0.5 + 0.4 * (y * 2 - 1) + rng.normal(0, 0.2, n), 0.01, 0.99)
    p[:50] = np.nan
    _cals, reports = cross_fitted_calibration({"m": p}, y, min_ece=0.0)
    assert reports[0].n == n - 50


def test_caveat_explains_an_auc_shift_without_claiming_inversion():
    rng = np.random.RandomState(10)
    n = 1500
    y = (rng.rand(n) < 0.35).astype(int)
    p = np.clip(0.5 + 0.4 * (y * 2 - 1) + rng.normal(0, 0.2, n), 0.01, 0.99)
    _cals, reports = cross_fitted_calibration({"m": p}, y, min_ece=0.0)
    text = calibration_caveat(reports)
    assert "one map PER FOLD" in text
    assert "RAW pooled AUC" in text


def test_caveat_when_nothing_was_touched():
    rng = np.random.RandomState(11)
    n = 3000
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.rand(n) < p).astype(int)
    _cals, reports = cross_fitted_calibration({"m": p}, y, min_ece=0.10)
    assert "No model needed recalibration" in calibration_caveat(reports)
