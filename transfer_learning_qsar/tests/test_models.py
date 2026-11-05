"""The registry, tuning, and the full grouped-evaluation loop.

Runs on the sklearn-only models, so it works without RDKit or LightGBM. The Tanimoto
models and LightGBM are skipped when absent, which is the behaviour the package promises.
"""

import numpy as np
import pandas as pd
import pytest

from qsar_tl.evaluation.floors import blend_vs_best_member, compare_to_floors
from qsar_tl.evaluation.loco import leakage_table, random_split_reference, run_loco
from qsar_tl.evaluation.splits import build_loco_folds
from qsar_tl.models.registry import (
    MODEL_ZOO,
    available_models,
    bootstrap_fit,
    fit_model,
    get_spec,
    known_models,
    predict_proba,
    unavailable_models,
)
from qsar_tl.models.tuning import autoqsar_score, safe_cv_folds, tune_all, tune_model

SKLEARN_ONLY = ["RandomForest", "LogReg (L2)", "BernoulliNB", "KPLS-DA"]


# ------------------------------------------------------------------------------- registry


def test_known_models_does_not_need_dependencies():
    """config.validate() calls this, so it must never import a heavy package."""
    assert "LightGBM" in known_models()
    assert "SVM (Tanimoto)" in known_models()


def test_available_is_a_subset_of_known():
    assert set(available_models()) <= set(known_models())


def test_unavailable_models_names_the_missing_module():
    for name, missing in unavailable_models().items():
        assert missing, f"{name} is listed as unavailable with nothing missing"


def test_unknown_model_raises_with_the_known_list():
    with pytest.raises(KeyError, match="unknown model"):
        get_spec("Gradient Boosting Machine")


def test_every_spec_declares_what_it_needs():
    for name, spec in MODEL_ZOO.items():
        assert spec.name == name
        assert callable(spec.fit) and callable(spec.proba)
        assert isinstance(spec.grid, list)


@pytest.mark.parametrize("name", SKLEARN_ONLY)
def test_sklearn_models_fit_and_predict(binary_problem, name):
    X, y, _groups = binary_problem
    model = fit_model(name, X, y, {}, seed=0)
    p = predict_proba(name, model, X)
    assert p.shape == (len(y), 2)
    assert np.all(p >= 0) and np.all(p <= 1)
    assert p.sum(axis=1) == pytest.approx(np.ones(len(y)), abs=1e-6)


@pytest.mark.parametrize("name", SKLEARN_ONLY)
def test_single_class_fold_still_returns_two_columns(binary_problem, name):
    """LOCO produces single-class training halves; a blend needs a full (n, 2) matrix."""
    X, y, _groups = binary_problem
    y_one = np.zeros(len(y), dtype=int)
    model = fit_model(name, X, y_one, {}, seed=0)
    p = predict_proba(name, model, X)
    assert p.shape == (len(y), 2)
    assert np.all(np.isfinite(p))


def test_fingerprint_model_without_fingerprints_raises():
    spec_name = "kNN (Tanimoto)"
    if spec_name not in available_models():
        pytest.skip("needs rdkit")
    with pytest.raises(ValueError, match="fingerprints"):
        fit_model(spec_name, np.zeros((4, 4)), np.array([0, 1, 0, 1]), {})


def test_unavailable_model_raises_a_helpful_import_error():
    missing = [n for n in known_models() if n not in available_models()]
    if not missing:
        pytest.skip("everything is installed")
    with pytest.raises(ImportError, match="not installed|needs"):
        fit_model(missing[0], np.zeros((4, 2)), np.array([0, 1, 0, 1]), {})


def test_bootstrap_is_stratified_and_returns_models(binary_problem):
    X, y, _groups = binary_problem
    models = bootstrap_fit("LogReg (L2)", X, y, {}, n_bootstrap=5, progress=False)
    assert len(models) == 5
    for m in models:
        assert predict_proba("LogReg (L2)", m, X[:10]).shape == (10, 2)


def test_predict_proba_chunking_matches_unchunked(binary_problem):
    X, y, _groups = binary_problem
    model = fit_model("RandomForest", X, y, {}, seed=0)
    whole = predict_proba("RandomForest", model, X)
    chunked = predict_proba("RandomForest", model, X, chunk=17)
    assert whole == pytest.approx(chunked)


# --------------------------------------------------------------------------------- tuning


def test_autoqsar_score_penalises_the_train_test_gap():
    tight = autoqsar_score(0.80, 0.78)
    wide = autoqsar_score(0.99, 0.78)
    assert tight > wide


def test_autoqsar_score_of_chance_is_half():
    assert autoqsar_score(0.5, 0.5) == pytest.approx(0.5)


def test_safe_cv_folds_clamps_to_the_smaller_class():
    y = np.array([0] * 100 + [1] * 3)
    assert safe_cv_folds(y, 10) == 3


def test_safe_cv_folds_never_below_two():
    assert safe_cv_folds(np.array([0, 1]), 10) == 2
    assert safe_cv_folds(np.array([0] * 5), 10) == 2


def test_tune_model_returns_params_and_evidence(binary_problem):
    X, y, _groups = binary_problem
    result = tune_model("LogReg (L2)", X, y, cv_folds=3, progress=False)
    assert result.params in [g for g in get_spec("LogReg (L2)").grid]
    assert 0 <= result.test_auc <= 1
    assert not result.table.empty
    assert "AutoQSAR" in result.summary()


def test_tune_model_reports_the_gap_penalty_cost(binary_problem):
    X, y, _groups = binary_problem
    result = tune_model("RandomForest", X, y, cv_folds=3, progress=False)
    assert result.gap_penalty_cost >= 0


def test_tune_all_skips_unavailable_models(binary_problem, capsys):
    X, y, _groups = binary_problem
    params = tune_all(
        ["LogReg (L2)", "LightGBM"], X, y, retune=True, cv_folds=3,
        skip_unavailable=True, verbose=True,
    )
    assert "LogReg (L2)" in params
    if "LightGBM" not in available_models():
        assert "LightGBM" not in params
        assert "SKIPPED" in capsys.readouterr().out


def test_tune_all_can_refuse_to_skip(binary_problem):
    X, y, _groups = binary_problem
    if "LightGBM" in available_models():
        pytest.skip("lightgbm is installed")
    with pytest.raises(ImportError):
        tune_all(["LightGBM"], X, y, retune=True, skip_unavailable=False, verbose=False)


def test_tune_all_honours_fixed_params(binary_problem):
    X, y, _groups = binary_problem
    fixed = {"LogReg (L2)": {"C": 0.123}}
    params = tune_all(["LogReg (L2)"], X, y, fixed=fixed, retune=False, verbose=False)
    assert params["LogReg (L2)"]["C"] == 0.123


# ---------------------------------------------------------------------------- evaluation


@pytest.fixture
def loco_setup(binary_problem):
    X, y, groups = binary_problem
    folds = build_loco_folds(groups, y, min_cluster_size=20, min_fold_actives=4,
                             n_remainder_folds=2)
    params = {n: {} for n in ("RandomForest", "LogReg (L2)")}
    return X, y, folds, params


def test_run_loco_covers_every_compound_once(loco_setup):
    X, y, folds, params = loco_setup
    result = run_loco(folds, X, y, models=list(params), params=params,
                      min_fold_n=5, verbose=False)
    assert result.covered.sum() == len(y)
    for name in params:
        assert np.all(np.isfinite(result.out_of_fold[name]))


def test_pooled_metrics_have_one_row_per_model(loco_setup):
    X, y, folds, params = loco_setup
    pooled = run_loco(folds, X, y, models=list(params), params=params,
                      min_fold_n=5, verbose=False).pooled()
    assert set(pooled.index) == set(params)
    assert "roc_auc" in pooled.columns
    assert (pooled["coverage"] == len(y)).all()


def test_small_folds_contribute_to_pooled_but_get_no_row(loco_setup):
    """An earlier evaluation reported a 5-compound fold whose metrics meant nothing."""
    X, y, folds, params = loco_setup
    result = run_loco(folds, X, y, models=list(params), params=params,
                      min_fold_n=1000, verbose=False)
    assert result.per_fold.empty            # every fold is below the floor
    assert result.covered.sum() == len(y)   # but all of them still scored
    assert result.unreported_folds


def test_macro_reports_mean_and_spread(loco_setup):
    X, y, folds, params = loco_setup
    result = run_loco(folds, X, y, models=list(params), params=params,
                      min_fold_n=5, verbose=False)
    macro = result.macro()
    assert ("roc_auc", "mean") in macro.columns
    assert ("roc_auc", "std") in macro.columns


def test_random_split_reference_and_leakage_gap(loco_setup):
    X, y, folds, params = loco_setup
    pooled = run_loco(folds, X, y, models=list(params), params=params,
                      min_fold_n=5, verbose=False).pooled()
    reference = random_split_reference(X, y, models=list(params), params=params,
                                       repeats=3, verbose=False)
    table = leakage_table(pooled, reference)
    assert set(table.columns) == {"random_split", "pooled_grouped", "leakage_gap"}
    # the whole point: a grouped split should not flatter the model
    assert (table["leakage_gap"] <= 0.2).all()


# -------------------------------------------------------------------------------- floors


def test_floor_comparison_identifies_models_that_clear_it():
    pooled = pd.DataFrame(
        {
            "roc_auc": [0.90, 0.70, 0.72],
            "precision@50": [0.88, 0.70, 0.72],
        },
        index=["LightGBM", "kNN (Tanimoto)", "LogReg (L2)"],
    )
    comps = compare_to_floors(pooled, floors=["kNN (Tanimoto)"],
                              metrics=("roc_auc", "precision@50"))
    winners = dict(comps[0].clears["roc_auc"])
    assert "LightGBM" in winners
    assert winners["LightGBM"] == pytest.approx(0.20)


def test_floor_comparison_reports_a_split_verdict():
    """Ahead on AUC but not at the top of the ranking is the wrong way round."""
    pooled = pd.DataFrame(
        {"roc_auc": [0.80, 0.70], "precision@50": [0.70, 0.72]},
        index=["LightGBM", "LogReg (L2)"],
    )
    text = compare_to_floors(pooled, floors=["LogReg (L2)"],
                             metrics=("roc_auc", "precision@50"))[0].summary()
    assert "NOTHING clears it" in text
    assert "wrong way round" in text


def test_floor_comparison_margins_differ_by_metric():
    """precision@N moves in steps of 1/N, so a one-compound difference is not a result."""
    pooled = pd.DataFrame(
        {"roc_auc": [0.73, 0.70], "precision@50": [0.72, 0.70]},
        index=["A", "kNN (Tanimoto)"],
    )
    comp = compare_to_floors(pooled, floors=["kNN (Tanimoto)"],
                             metrics=("roc_auc", "precision@50"))[0]
    assert comp.clears["roc_auc"]          # +0.03 clears the 0.02 AUC margin
    assert not comp.clears["precision@50"]  # +0.02 does not clear the 0.05 margin


def test_missing_floor_is_skipped():
    pooled = pd.DataFrame({"roc_auc": [0.8]}, index=["A"])
    assert compare_to_floors(pooled, floors=["kNN (Tanimoto)"]) == []


def test_blend_vs_best_member_detects_a_losing_blend():
    pooled = pd.DataFrame(
        {"roc_auc": [0.783, 0.804, 0.777]},
        index=["Blend (5)", "LightGBM", "RandomForest"],
    )
    text = blend_vs_best_member(pooled)
    assert "LightGBM" in text
    assert "not earning it" in text


def test_blend_vs_best_member_detects_a_winning_blend():
    pooled = pd.DataFrame({"roc_auc": [0.9, 0.8]}, index=["Blend", "LightGBM"])
    assert "ahead of every member" in blend_vs_best_member(pooled)


def test_blend_vs_best_member_without_a_blend_row():
    pooled = pd.DataFrame({"roc_auc": [0.8]}, index=["LightGBM"])
    assert "No blend row" in blend_vs_best_member(pooled)
