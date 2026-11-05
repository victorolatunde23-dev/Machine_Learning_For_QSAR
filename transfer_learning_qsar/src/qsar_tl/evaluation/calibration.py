"""Probability calibration, cross-fitted, with the caveats attached.

Three things learned the hard way:

* **Platt beats isotonic at this sample size.** Isotonic collapses ties on a few hundred
  compounds and moved pooled AUC by 0.10 in one run; Platt improved ECE in 10 of 10
  model x endpoint pairs against isotonic's 7 of 10. ``sigmoid`` is the default.
* **Do not recalibrate what is already calibrated.** One endpoint arrived at raw ECE
  0.072-0.084 (its base rate is near 0.5 and balanced class weighting naturally targets
  that), and recalibrating anyway made the blend *worse*, 0.084 -> 0.167. Pairs below
  ``min_ece`` are left alone.
* **A pooled out-of-fold column is not a single calibration map.** It is assembled from
  one map per fold, so while each map preserves order within its own fold, pooling can
  reorder compounds that sat in *different* folds. The pooled calibrated AUC can therefore
  shift without any map being inverted. Quote the raw AUC for discrimination and the
  calibrated numbers for probability quality. A single deployment calibrator applied to a
  library does not have this problem and leaves the ranking untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .metrics import expected_calibration_error, safe_roc_auc

__all__ = ["Calibrator", "fit_calibrator", "cross_fitted_calibration", "CalibrationReport"]


@dataclass
class Calibrator:
    """A fitted one-dimensional probability map, exportable as plain data.

    The plain-data export matters: a pickled calibrator only reloads where the defining
    class exists and the library versions match, and losing a calibrator means losing the
    ability to reproduce a call. :meth:`to_dict` carries the parameters and the formula.
    """

    method: str  # "sigmoid" | "isotonic" | "identity"
    model: object = None

    def transform(self, scores) -> np.ndarray:
        p = np.asarray(scores, dtype=float)
        if self.model is None or self.method == "identity":
            return np.clip(p, 0.0, 1.0)
        if self.method == "sigmoid":
            return self.model.predict_proba(p.reshape(-1, 1))[:, 1]
        return np.clip(self.model.predict(p), 0.0, 1.0)

    __call__ = transform

    @property
    def slope(self) -> Optional[float]:
        """Platt slope, or None. A negative slope would invert the ranking."""
        if self.method == "sigmoid" and self.model is not None:
            return float(np.ravel(self.model.coef_)[0])
        return None

    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {"method": self.method, "fitted": self.model is not None}
        if self.model is None:
            return out
        if self.method == "sigmoid":
            out["coef"] = float(np.ravel(self.model.coef_)[0])
            out["intercept"] = float(np.ravel(self.model.intercept_)[0])
            out["apply"] = "1 / (1 + exp(-(coef * p_raw + intercept)))"
        else:
            out["x"] = [float(v) for v in np.asarray(self.model.X_thresholds_).ravel()]
            out["y"] = [float(v) for v in np.asarray(self.model.y_thresholds_).ravel()]
            out["apply"] = "numpy.interp(p_raw, x, y), clipped to [0, 1]"
        return out


def fit_calibrator(scores, y_true, method: str = "sigmoid") -> Calibrator:
    """Fit a calibration map. ``method="none"`` returns an identity map."""
    p = np.asarray(scores, dtype=float).reshape(-1, 1)
    y = np.asarray(y_true).astype(int)
    if method in ("none", "identity"):
        return Calibrator("identity", None)
    if len(np.unique(y)) < 2:
        return Calibrator("identity", None)
    if method == "sigmoid":
        # C large => effectively unregularised, which is what Platt scaling is.
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(p, y)
        return Calibrator("sigmoid", lr)
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p.ravel(), y)
        return Calibrator("isotonic", iso)
    raise ValueError(f"unknown calibration method {method!r}")


@dataclass
class CalibrationReport:
    """Before/after for one model, plus whether it was touched at all."""

    name: str
    calibrated: bool
    n: int
    observed_rate: float
    ece_raw: float
    ece_calibrated: float
    auc_raw: float
    auc_calibrated: float
    slope: Optional[float]
    reason: str = ""

    @property
    def auc_shift(self) -> float:
        return abs(self.auc_calibrated - self.auc_raw)

    def as_row(self) -> Dict[str, object]:
        return {
            "model": self.name,
            "calibrated": self.calibrated,
            "n": self.n,
            "observed_rate": round(self.observed_rate, 4),
            "ECE_raw": round(self.ece_raw, 4),
            "ECE_cal": round(self.ece_calibrated, 4),
            "auc_raw": round(self.auc_raw, 4),
            "auc_cal": round(self.auc_calibrated, 4),
            "slope": None if self.slope is None else round(self.slope, 4),
            "reason": self.reason,
        }


def cross_fitted_calibration(
    out_of_fold_scores: Dict[str, np.ndarray],
    y_true,
    *,
    method: str = "sigmoid",
    min_ece: float = 0.10,
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[Dict[str, Calibrator], List[CalibrationReport]]:
    """Calibrate each model's out-of-fold scores, skipping those already calibrated.

    Returns the *deployment* calibrators -- one map per model, fitted on all the
    out-of-fold predictions -- and a report whose before/after columns come from a
    cross-fitted estimate, so the reported improvement is not measured on the data the map
    was fitted to.

    ``min_ece`` is the skip rule: a model whose raw ECE is already below it is left alone.
    """
    y = np.asarray(y_true).astype(int)
    calibrators: Dict[str, Calibrator] = {}
    reports: List[CalibrationReport] = []

    for name, raw in out_of_fold_scores.items():
        p = np.asarray(raw, dtype=float)
        keep = np.isfinite(p)
        p, yy = p[keep], y[keep]
        ece_raw = expected_calibration_error(yy, np.clip(p, 0, 1))
        auc_raw = safe_roc_auc(yy, p)

        if method in ("none", "identity"):
            calibrators[name] = Calibrator("identity", None)
            reports.append(
                CalibrationReport(
                    name, False, len(p), float(yy.mean()) if len(yy) else float("nan"),
                    ece_raw, ece_raw, auc_raw, auc_raw, None,
                    reason="calibration disabled",
                )
            )
            continue

        if ece_raw < min_ece:
            calibrators[name] = Calibrator("identity", None)
            reports.append(
                CalibrationReport(
                    name, False, len(p), float(yy.mean()) if len(yy) else float("nan"),
                    ece_raw, ece_raw, auc_raw, auc_raw, None,
                    reason=f"already calibrated (ECE {ece_raw:.4f} < {min_ece})",
                )
            )
            continue

        # cross-fitted estimate of what calibration buys
        cal_oof = np.full(len(p), np.nan)
        if len(np.unique(yy)) > 1 and len(p) >= n_folds * 2:
            skf = StratifiedKFold(
                n_splits=min(n_folds, int(np.bincount(yy).min())),
                shuffle=True,
                random_state=random_state,
            )
            for tr, te in skf.split(p.reshape(-1, 1), yy):
                c = fit_calibrator(p[tr], yy[tr], method)
                cal_oof[te] = c.transform(p[te])
        ok = np.isfinite(cal_oof)
        ece_cal = (
            expected_calibration_error(yy[ok], cal_oof[ok]) if ok.any() else float("nan")
        )
        auc_cal = safe_roc_auc(yy[ok], cal_oof[ok]) if ok.any() else float("nan")

        deployment = fit_calibrator(p, yy, method)
        calibrators[name] = deployment
        reports.append(
            CalibrationReport(
                name, True, len(p), float(yy.mean()) if len(yy) else float("nan"),
                ece_raw, ece_cal, auc_raw, auc_cal, deployment.slope,
                reason="recalibrated",
            )
        )

    return calibrators, reports


def calibration_caveat(reports: Sequence[CalibrationReport]) -> str:
    """The sentence to print next to a large calibrated-AUC shift."""
    touched = [r for r in reports if r.calibrated]
    if not touched:
        return "No model needed recalibration."
    worst = max(touched, key=lambda r: r.auc_shift)
    inverted = [r.name for r in touched if r.slope is not None and r.slope < 0]
    lines = [
        f"Largest absolute AUC change from calibration: {worst.auc_shift:.4f} ({worst.name})."
    ]
    if inverted:
        lines.append(
            f"  !! NEGATIVE calibrator slope for {inverted}: that map inverts the ranking. "
            f"Something is wrong upstream; do not deploy it."
        )
    else:
        lines.append(
            "  This is expected and is not an inversion (no slope is negative). The "
            "pooled out-of-fold column is assembled from one map PER FOLD, so while each "
            "map preserves order within its fold, pooling can reorder compounds that sat "
            "in different folds."
        )
    lines.append("  Quote the RAW pooled AUC for discrimination.")
    lines.append("  Quote the CALIBRATED numbers only for probability quality.")
    lines.append(
        "  A library is scored with a single deployment calibrator per model, so its "
        "ranking is untouched."
    )
    return "\n".join(lines)
