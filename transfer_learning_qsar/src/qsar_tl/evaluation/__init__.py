"""Grouped-split evaluation, calibration, thresholds and floor comparisons."""

from .calibration import (
    CalibrationReport,
    Calibrator,
    calibration_caveat,
    cross_fitted_calibration,
    fit_calibrator,
)
from .floors import FloorComparison, blend_vs_best_member, compare_to_floors
from .loco import LocoResult, leakage_table, random_split_reference, run_loco
from .metrics import (
    blend_equal,
    classification_metrics,
    enrichment_factor,
    expected_calibration_error,
    precision_at_n,
    safe_roc_auc,
)
from .splits import (
    Fold,
    FoldSet,
    build_loco_folds,
    butina_clusters,
    butina_clusters_from_fingerprints,
    scaffold_groups,
)
from .thresholds import ThresholdChoice, choose_threshold, list_size_table, threshold_sweep

__all__ = [
    "CalibrationReport",
    "Calibrator",
    "Fold",
    "FoldSet",
    "FloorComparison",
    "LocoResult",
    "ThresholdChoice",
    "blend_equal",
    "blend_vs_best_member",
    "build_loco_folds",
    "butina_clusters",
    "butina_clusters_from_fingerprints",
    "calibration_caveat",
    "choose_threshold",
    "classification_metrics",
    "compare_to_floors",
    "cross_fitted_calibration",
    "enrichment_factor",
    "expected_calibration_error",
    "fit_calibrator",
    "leakage_table",
    "list_size_table",
    "precision_at_n",
    "random_split_reference",
    "run_loco",
    "safe_roc_auc",
    "scaffold_groups",
    "threshold_sweep",
]
