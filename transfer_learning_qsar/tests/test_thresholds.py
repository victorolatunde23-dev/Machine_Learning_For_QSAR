"""Threshold selection, and the called-fraction cap that makes it usable.

The cap is the point of this module. A run once chose p >= 0.11 by maximising MCC, which
called 86% of compounds active at recall 1.000 -- over a 290,000-compound library, a
250,000-compound "shortlist".
"""

import numpy as np
import pandas as pd
import pytest

from qsar_tl.evaluation.thresholds import (
    choose_threshold,
    list_size_table,
    threshold_sweep,
)


def test_sweep_shape_and_monotonicity(scores_and_labels):
    p, y = scores_and_labels
    sweep = threshold_sweep(y, p)
    assert {"threshold", "precision", "recall", "mcc", "f1", "called_frac",
            "n_called"} <= set(sweep.columns)
    # recall and the called fraction can only fall as the threshold rises
    assert sweep["recall"].is_monotonic_decreasing
    assert sweep["called_frac"].is_monotonic_decreasing


def test_sweep_respects_a_custom_grid(scores_and_labels):
    p, y = scores_and_labels
    sweep = threshold_sweep(y, p, grid=[0.25, 0.5, 0.75])
    assert list(sweep["threshold"]) == [0.25, 0.5, 0.75]


def test_cap_rejects_a_permissive_optimum():
    """A weak model whose MCC peaks at a permissive cut must not set the threshold there."""
    rng = np.random.RandomState(0)
    n = 1000
    y = (rng.rand(n) < 0.5).astype(int)
    # almost no signal, and the scores are squashed low so a low cut maximises MCC
    p = np.clip(0.12 + 0.03 * y + rng.normal(0, 0.02, n), 0.01, 0.99)

    loose = choose_threshold(y, p, max_called_frac=1.0)
    tight = choose_threshold(y, p, max_called_frac=0.25)

    assert loose.called_frac > 0.5
    assert tight.called_frac <= 0.25 + 1e-9
    assert tight.capped is True
    assert any("cap" in note for note in tight.notes)


def test_cap_is_reported_when_it_binds(scores_and_labels):
    p, y = scores_and_labels
    choice = choose_threshold(y, p, max_called_frac=0.05)
    assert choice.called_frac <= 0.05 + 1e-9


def test_unreachable_cap_falls_back_and_says_so():
    """If no cut is stringent enough, the model is not separating the classes."""
    y = np.array([0, 1] * 100)
    p = np.full(200, 0.9)  # every compound scores identically and high
    choice = choose_threshold(y, p, max_called_frac=0.01)
    assert any("NO threshold" in n for n in choice.notes)
    # a cut that selects nothing is not a usable answer either, so the cap is only
    # satisfied by a non-empty selection
    assert choice.called_frac > 0


def test_degenerate_recall_is_flagged():
    rng = np.random.RandomState(1)
    n = 400
    y = (rng.rand(n) < 0.5).astype(int)
    p = np.full(n, 0.6)
    choice = choose_threshold(y, p, max_called_frac=1.0)
    assert choice.degenerate is True
    assert any("recall is ~1.0" in n for n in choice.notes)


def test_override_wins_and_is_reported(scores_and_labels):
    p, y = scores_and_labels
    choice = choose_threshold(y, p, override=0.73)
    assert choice.threshold == pytest.approx(0.73)
    assert any("override" in n for n in choice.notes)


@pytest.mark.parametrize("objective", ["mcc", "f1", "balanced_accuracy", "youden"])
def test_each_objective_runs(scores_and_labels, objective):
    p, y = scores_and_labels
    choice = choose_threshold(y, p, objective=objective, max_called_frac=1.0)
    assert 0.0 <= choice.threshold <= 1.0
    assert choice.objective == objective


def test_unknown_objective_is_rejected(scores_and_labels):
    p, y = scores_and_labels
    with pytest.raises(ValueError, match="objective"):
        choose_threshold(y, p, objective="roc_auc")


def test_bad_cap_is_rejected(scores_and_labels):
    p, y = scores_and_labels
    with pytest.raises(ValueError, match="max_called_frac"):
        choose_threshold(y, p, max_called_frac=0.0)


def test_single_class_labels_raise_a_clear_error():
    with pytest.raises(ValueError, match="single-class|undefined"):
        choose_threshold(np.zeros(50, dtype=int), np.linspace(0, 1, 50))


def test_precision_targets_table(scores_and_labels):
    p, y = scores_and_labels
    choice = choose_threshold(y, p)
    table = choice.precision_targets((0.6, 0.8), library_size=290000)
    assert set(table["precision_target"]) == {0.6, 0.8}
    assert "est_n_called" in table.columns
    # a reachable target gives a usable threshold and a non-zero estimated list size
    reachable = table[table["reachable"]]
    assert len(reachable)
    assert (reachable["threshold"].between(0, 1)).all()
    assert (reachable["est_n_called"] > 0).all()


def test_precision_targets_reports_unreachable_targets():
    """A target the model cannot hit must be marked, not silently given a threshold."""
    rng = np.random.RandomState(5)
    n = 600
    y = (rng.rand(n) < 0.5).astype(int)
    p = np.clip(0.5 + rng.normal(0, 0.1, n), 0.01, 0.99)   # no signal at all
    choice = choose_threshold(y, p, max_called_frac=1.0)
    table = choice.precision_targets((0.95,))
    row = table.iloc[0]
    assert bool(row["reachable"]) is False
    assert np.isnan(row["threshold"])


def test_list_size_table_maps_sizes_to_thresholds(scores_and_labels):
    p, y = scores_and_labels
    choice = choose_threshold(y, p)
    library = np.random.RandomState(0).uniform(0, 1, 5000)
    table = list_size_table(library, choice.sweep, sizes=(100, 1000))
    assert list(table["list_size"]) == [100, 1000]
    # a bigger list needs a lower threshold
    assert table.iloc[0]["implied_threshold"] >= table.iloc[1]["implied_threshold"]
    assert "oof_precision" in table.columns


def test_list_size_table_skips_sizes_bigger_than_the_library(scores_and_labels):
    p, y = scores_and_labels
    choice = choose_threshold(y, p)
    table = list_size_table(np.random.rand(50), choice.sweep, sizes=(10, 100000))
    assert list(table["list_size"]) == [10]


def test_choice_summary_and_dict(scores_and_labels):
    p, y = scores_and_labels
    choice = choose_threshold(y, p)
    assert "max mcc" in choice.summary()
    d = choice.as_dict()
    assert d["objective"] == "mcc"
    assert 0 <= d["threshold"] <= 1
