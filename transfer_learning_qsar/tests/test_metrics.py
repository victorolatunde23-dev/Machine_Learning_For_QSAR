"""Metrics, including the screening metrics and their small-sample edge cases."""

import numpy as np
import pytest

from qsar_tl.evaluation.metrics import (
    blend_equal,
    classification_metrics,
    enrichment_factor,
    expected_calibration_error,
    precision_at_n,
    safe_roc_auc,
)


def test_safe_roc_auc_is_nan_on_a_single_class():
    """LOCO produces single-class folds routinely; raising there would crash a run."""
    assert np.isnan(safe_roc_auc(np.zeros(10), np.linspace(0, 1, 10)))
    assert np.isnan(safe_roc_auc(np.ones(10), np.linspace(0, 1, 10)))


def test_safe_roc_auc_perfect_and_inverted():
    y = np.array([0, 0, 1, 1])
    assert safe_roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert safe_roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_enrichment_factor_perfect_ranking():
    y = np.array([1] * 10 + [0] * 90)
    scores = np.linspace(1, 0, 100)
    # top 10% is all 10 actives; base rate 0.1 -> EF = 1.0/0.1 = 10
    assert enrichment_factor(y, scores, 0.10) == pytest.approx(10.0)


def test_enrichment_factor_of_one_is_a_worthless_ranking():
    rng = np.random.RandomState(0)
    y = (rng.rand(2000) < 0.3).astype(int)
    ef = enrichment_factor(y, rng.rand(2000), 0.10)
    assert 0.6 < ef < 1.4


def test_enrichment_factor_needs_at_least_one_compound():
    y = np.array([1] * 2 + [0] * 12)
    # the top 1% of 14 compounds rounds to one, not zero
    assert np.isfinite(enrichment_factor(y, np.linspace(1, 0, 14), 0.01))


def test_enrichment_factor_is_nan_without_actives():
    assert np.isnan(enrichment_factor(np.zeros(10), np.linspace(0, 1, 10), 0.1))


def test_precision_at_n_clamps_to_the_available_rows():
    y = np.array([1, 1, 0])
    assert precision_at_n(y, np.array([0.9, 0.8, 0.1]), 50) == pytest.approx(2 / 3)


def test_precision_at_n_picks_the_top_scores():
    y = np.array([0, 0, 1, 1])
    assert precision_at_n(y, np.array([0.1, 0.2, 0.8, 0.9]), 2) == 1.0


def test_ece_is_zero_for_a_perfectly_calibrated_predictor():
    rng = np.random.RandomState(0)
    p = rng.uniform(0.05, 0.95, 40000)
    y = (rng.rand(40000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02


def test_ece_is_large_for_an_overconfident_predictor():
    y = np.array([0] * 500 + [1] * 500)
    p = np.array([0.99] * 500 + [0.01] * 500)
    assert expected_calibration_error(y, p) > 0.9


def test_ece_handles_the_closed_upper_bin():
    """Probabilities of exactly 1.0 must land in the last bin, not be dropped."""
    y = np.array([1, 1, 0, 0])
    assert np.isfinite(expected_calibration_error(y, np.array([1.0, 1.0, 0.0, 0.0])))


def test_classification_metrics_keys_and_ranges(scores_and_labels):
    p, y = scores_and_labels
    m = classification_metrics(y, p, 0.5, precision_ns=(50,), enrichment_fractions=(0.1,))
    for key in ("n", "n_active", "base_rate", "roc_auc", "pr_auc", "balanced_acc",
                "mcc", "precision", "recall", "f1", "brier", "ECE",
                "EF@10%", "precision@50"):
        assert key in m
    assert m["n"] == len(y)
    assert 0 <= m["roc_auc"] <= 1
    assert -1 <= m["mcc"] <= 1


def test_classification_metrics_single_class_returns_nan_not_an_error():
    m = classification_metrics(np.zeros(20, dtype=int), np.linspace(0, 1, 20))
    assert np.isnan(m["roc_auc"])
    assert np.isnan(m["mcc"])
    assert m["n_active"] == 0


def test_threshold_changes_the_hard_metrics_but_not_auc(scores_and_labels):
    p, y = scores_and_labels
    low = classification_metrics(y, p, 0.2)
    high = classification_metrics(y, p, 0.8)
    assert low["roc_auc"] == pytest.approx(high["roc_auc"])
    assert low["recall"] > high["recall"]


def test_blend_equal_averages():
    a = np.array([0.0, 0.5, 1.0])
    b = np.array([1.0, 0.5, 0.0])
    assert blend_equal([a, b]) == pytest.approx([0.5, 0.5, 0.5])


def test_blend_equal_rejects_an_empty_list():
    with pytest.raises(ValueError):
        blend_equal([])
