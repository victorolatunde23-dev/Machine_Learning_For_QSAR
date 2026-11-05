"""Fold construction: the judgement about which families can be scored.

Clustering itself needs a chemistry toolkit, so these tests drive
:func:`build_loco_folds` with synthetic cluster ids -- which is where the decisions live.
"""

import numpy as np
import pytest

from qsar_tl.evaluation.splits import Fold, FoldSet, build_loco_folds


def test_large_balanced_clusters_get_their_own_fold(binary_problem):
    _X, y, groups = binary_problem
    folds = build_loco_folds(groups, y, min_cluster_size=20, min_fold_actives=4,
                             max_own_folds=12, n_remainder_folds=2)
    names = [f.name for f in folds]
    assert any(n.startswith("cluster") for n in names)
    assert any(n.startswith("small families") for n in names)


def test_every_compound_lands_in_exactly_one_fold(binary_problem):
    _X, y, groups = binary_problem
    folds = build_loco_folds(groups, y, min_cluster_size=20, min_fold_actives=4)
    stacked = np.sum([f.mask for f in folds], axis=0)
    assert set(np.unique(stacked)) == {1}
    assert folds.coverage == len(y)


def test_small_clusters_are_pooled_not_dropped():
    groups = np.array([0] * 40 + list(range(1, 21)))  # one big, twenty singletons
    y = np.array([0, 1] * 30)
    folds = build_loco_folds(groups, y, min_cluster_size=20, min_fold_actives=4,
                             n_remainder_folds=2)
    assert folds.coverage == len(y)
    assert folds.pooled_cluster_count == 20


def test_unbalanced_large_clusters_are_rejected_with_a_reason():
    """A fold of 30 compounds that are all active cannot be scored.

    An earlier evaluation reported per-fold metrics for folds like this; they ranged
    0.500-0.917 across models and meant nothing.
    """
    groups = np.array([0] * 30 + [1] * 30)
    y = np.array([1] * 30 + [0, 1] * 15)  # cluster 0 is single-class
    folds = build_loco_folds(groups, y, min_cluster_size=20, min_fold_actives=5,
                             n_remainder_folds=1)
    assert (0, 30) in folds.rejected
    assert all(not f.name.startswith("cluster 0") for f in folds)


def test_max_own_folds_is_respected():
    groups = np.repeat(np.arange(8), 30)
    y = np.tile([0, 1], 120)
    folds = build_loco_folds(groups, y, min_cluster_size=20, min_fold_actives=5,
                             max_own_folds=3, n_remainder_folds=2)
    own = [f for f in folds if f.name.startswith("cluster")]
    assert len(own) == 3


def test_remainder_folds_are_size_balanced():
    groups = np.concatenate([np.full(10, i) for i in range(12)])
    y = np.tile([0, 1], 60)
    folds = build_loco_folds(groups, y, min_cluster_size=50, min_fold_actives=5,
                             n_remainder_folds=3)
    sizes = sorted(f.n for f in folds)
    assert len(folds) == 3
    assert max(sizes) - min(sizes) <= 10  # greedy assignment keeps them close


def test_nan_labels_are_carried_but_not_counted(binary_problem):
    """Compounds with no label for the modelled endpoint still belong to a family."""
    _X, y, groups = binary_problem
    y = y.astype(float)
    y[:20] = np.nan
    folds = build_loco_folds(groups, y, min_cluster_size=20, min_fold_actives=4)
    assert folds.coverage == len(y)
    counted = sum(f.n_active for f in folds)
    assert counted == int(np.nansum(y))


def test_source_tags_annotate_the_folds():
    groups = np.array([0] * 30 + [1] * 30)
    y = np.tile([0, 1], 30)
    tags = ["US1234567"] * 30 + ["CHEMBL"] * 30
    folds = build_loco_folds(groups, y, min_cluster_size=20, min_fold_actives=5,
                             source_tags=tags, n_remainder_folds=1)
    by_name = {f.name: f for f in folds}
    assert by_name["cluster 0"].dominant_source == "US1234567"
    assert by_name["cluster 0"].source_share == pytest.approx(1.0)


def test_single_source_folds_excludes_generic_buckets():
    """A fold that is 100% CHEMBL is not 'dominated by one source': CHEMBL is a bucket."""
    groups = np.array([0] * 30 + [1] * 30)
    y = np.tile([0, 1], 30)
    tags = ["US1234567"] * 30 + ["CHEMBL"] * 30
    folds = build_loco_folds(groups, y, min_cluster_size=20, min_fold_actives=5,
                             source_tags=tags, n_remainder_folds=1)
    flagged = [f.name for f in folds.single_source_folds()]
    assert "cluster 0" in flagged
    assert "cluster 1" not in flagged


def test_distance_fn_annotates_folds():
    groups = np.array([0] * 30 + [1] * 30)
    y = np.tile([0, 1], 30)
    folds = build_loco_folds(
        groups, y, min_cluster_size=20, min_fold_actives=5, n_remainder_folds=1,
        distance_fn=lambda inside, outside: 0.42,
    )
    assert all(f.mean_distance_to_train == pytest.approx(0.42) for f in folds)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="differ in length"):
        build_loco_folds(np.zeros(10), np.zeros(5))


def test_fold_set_frame_and_iteration(binary_problem):
    _X, y, groups = binary_problem
    folds = build_loco_folds(groups, y, min_cluster_size=20, min_fold_actives=4)
    frame = folds.to_frame()
    assert "n" in frame.columns and "n_active" in frame.columns
    assert len(list(folds)) == len(folds)


def test_empty_foldset_frame_has_columns():
    assert "n" in FoldSet().to_frame().columns
