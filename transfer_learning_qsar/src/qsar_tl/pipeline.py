"""End-to-end orchestration: data -> features -> models -> evaluation -> screen.

The stages are separate methods on purpose. Running them one at a time is how the whole
thing was developed, and each stage's output is worth looking at before paying for the
next -- in particular, :meth:`Pipeline.evaluate` is what tells you whether screening is
worth doing at all.

A sanity gate runs after labelling. An earlier version of this pipeline lost a run to an
upstream download failure that surfaced six stages later as
``n_splits=10 greater than the number of samples: n_samples=3``; the gate turns that into
a message that names the likely cause.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._optional import have, require
from .config import Config, ConfigError
from .data.labels import GENERIC_SOURCE_TAGS, assemble_label_table, detect_derived_endpoint
from .evaluation.calibration import calibration_caveat, cross_fitted_calibration
from .evaluation.floors import blend_vs_best_member, compare_to_floors
from .evaluation.loco import leakage_table, random_split_reference, run_loco
from .evaluation.splits import build_loco_folds, butina_clusters_from_fingerprints
from .evaluation.thresholds import choose_threshold
from .models.registry import available_models, fit_model, get_spec, predict_proba
from .models.tuning import tune_all

__all__ = ["Pipeline", "PipelineState", "InsufficientDataError"]


class InsufficientDataError(RuntimeError):
    """Too few labelled compounds to evaluate honestly."""


@dataclass
class PipelineState:
    """Everything a stage produces, so later stages can be run independently."""

    long_frame: Optional[pd.DataFrame] = None
    wide_frame: Optional[pd.DataFrame] = None
    labels: Optional[pd.DataFrame] = None
    label_stats: Dict[str, Any] = field(default_factory=dict)
    derived_report: Optional[Any] = None

    featurised: Optional[Any] = None
    X: Optional[np.ndarray] = None
    Y: Optional[np.ndarray] = None
    fingerprints: Optional[List] = None
    descriptors: Optional[np.ndarray] = None
    descriptor_selector: Optional[Any] = None
    descriptor_scaler: Optional[Any] = None

    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    folds: Optional[Any] = None
    loco: Optional[Any] = None
    pooled: Optional[pd.DataFrame] = None
    reference: Optional[pd.DataFrame] = None
    calibrators: Dict[str, Any] = field(default_factory=dict)
    calibration_reports: List[Any] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    threshold_choices: Dict[str, Any] = field(default_factory=dict)

    final_models: Dict[str, Any] = field(default_factory=dict)
    domain: Optional[Any] = None
    training_keys: set = field(default_factory=set)


class Pipeline:
    """A configured run.

    >>> from qsar_tl import Config, Pipeline
    >>> pipe = Pipeline(Config())          # defaults: PSMA pKi classification
    >>> pipe.config.labels.primary_endpoint
    'pKi'
    """

    def __init__(self, config: Config, *, verbose: bool = True):
        self.config = config.validate()
        self.verbose = verbose
        self.state = PipelineState()
        self.config.ensure_dirs()

    # ------------------------------------------------------------------------- helpers

    def _say(self, *parts: object) -> None:
        if self.verbose:
            print(*parts)

    @property
    def output_dir(self) -> Path:
        return Path(self.config.output_dir)

    # --------------------------------------------------------------------------- data

    def load_labels(
        self,
        long_frame: Optional[pd.DataFrame] = None,
        wide_frame: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Assemble the label table, with derived-value removal and censored recovery.

        ``long_frame`` has one row per measurement (``smiles``, ``ligand_name``,
        ``affinity_raw``, ``endpoint_name``); ``wide_frame`` has one row per compound with
        a column per endpoint in p-units, and is used only by the derived-value detector.
        Pass them in, or set ``data.source`` and let :mod:`qsar_tl.data.bindingdb` fetch
        them.
        """
        cfg = self.config
        if long_frame is None:
            from .data.bindingdb import fetch_bindingdb

            long_frame, wide_frame = fetch_bindingdb(cfg, verbose=self.verbose)
        self.state.long_frame = long_frame
        self.state.wide_frame = wide_frame

        derived_smiles: set = set()
        if cfg.labels.drop_derived_endpoint_values and wide_frame is not None:
            wide = wide_frame.copy()
            if "source_tag" not in wide.columns:
                from .data.labels import source_tag

                wide["source_tag"] = wide["ligand_name"].map(source_tag)
            a, b = cfg.data.endpoint_names[0], cfg.data.endpoint_names[1]
            report = detect_derived_endpoint(
                wide, a, b,
                min_block_size=cfg.labels.derived_min_block_size,
                offset_tol=cfg.labels.derived_offset_tol,
            )
            self.state.derived_report = report
            derived_smiles = set(wide.loc[report.mask, "smiles"])
            if self.verbose:
                print("=" * 86)
                print(f"DERIVED-VALUE DETECTOR -- fixed offsets between {a} and {b}")
                print("=" * 86)
                if report.blocks:
                    print(report.to_frame().to_string(index=False))
                print(report.summary())

        extra = None
        if cfg.labels.keep_unlabelled_for_domain and wide_frame is not None:
            extra = wide_frame["smiles"].tolist()

        names = (
            dict(zip(wide_frame["smiles"], wide_frame["ligand_name"]))
            if wide_frame is not None and "ligand_name" in wide_frame.columns
            else None
        )
        table, stats = assemble_label_table(
            long_frame,
            endpoints=cfg.endpoints,
            active_thresholds=cfg.labels.active_thresholds,
            derived_endpoint=cfg.labels.derived_endpoint,
            drop_derived=cfg.labels.drop_derived_endpoint_values,
            derived_smiles=derived_smiles,
            keep_censored=cfg.labels.keep_censored_as_labels,
            min_affinity_nm=cfg.data.min_affinity_nm,
            max_affinity_nm=cfg.data.max_affinity_nm,
            extra_smiles=extra,
            ligand_names=names,
        )
        self.state.labels = table
        self.state.label_stats = stats

        if self.verbose:
            print("\n" + "=" * 86)
            print("LABEL ASSEMBLY")
            print("=" * 86)
            print(pd.DataFrame({k: v.as_row() for k, v in stats.items()}).T.to_string())
            self._report_coverage(table)

        self._sanity_gate(table)
        return table

    def _report_coverage(self, table: pd.DataFrame) -> None:
        cfg = self.config
        primary = cfg.labels.primary_endpoint
        for ep in cfg.endpoints:
            col = table[f"y_{ep}"]
            n = int(col.notna().sum())
            print(
                f"  {ep:<8s} labelled {n:>5,}   active {int((col == 1).sum()):>5,}   "
                f"inactive {int((col == 0).sum()):>5,}   "
                f"base rate {col.dropna().mean():.3f}"
                if n
                else f"  {ep:<8s} no labels"
            )
        n_primary = int(table[f"y_{primary}"].notna().sum())
        print(f"  carried through : {len(table):,} compounds")
        if len(table) > n_primary:
            print(
                f"  of which {len(table) - n_primary:,} have no {primary} label: they "
                f"cannot train the\n  model but still define the applicability domain and "
                f"shape the clustering."
            )
        counts = table["source_tag"].value_counts()
        specific = counts[~counts.index.isin(GENERIC_SOURCE_TAGS)]
        if len(specific) and specific.iloc[0] / len(table) > 0.15:
            print(
                f"\n  !! {specific.index[0]!r} -- a single filing -- supplies "
                f"{specific.iloc[0]} of {len(table)} compounds "
                f"({100 * specific.iloc[0] / len(table):.0f}%)."
            )
            print("     The evaluation holds it out as its own fold so it cannot leak.")

    def _sanity_gate(self, table: pd.DataFrame) -> None:
        cfg = self.config
        y = table[f"y_{cfg.labels.primary_endpoint}"].dropna().astype(int)
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        if (
            len(y) >= cfg.evaluation.min_train_compounds
            and min(n_pos, n_neg) >= cfg.evaluation.min_per_class
        ):
            self._say(f"\nSanity gate passed: {len(y)} labelled, {n_pos} active / {n_neg} inactive.")
            return
        raise InsufficientDataError(
            f"Only {len(y)} labelled compounds ({n_pos} active, {n_neg} inactive); this "
            f"pipeline needs at least {cfg.evaluation.min_train_compounds} with "
            f"{cfg.evaluation.min_per_class} of each class.\n"
            f"Nothing downstream would be meaningful, so it stops here.\n\n"
            f"Most likely causes, in order:\n"
            f"  1. The data acquisition failed or fell back to a synthetic set. Check the "
            f"output above for a warning.\n"
            f"  2. active_thresholds[{cfg.labels.primary_endpoint!r}] = "
            f"{cfg.active_cut(cfg.labels.primary_endpoint)} sits outside the data, so one "
            f"class is empty.\n"
            f"  3. drop_derived_endpoint_values removed more than expected -- the detector "
            f"report above says how many and from which filings.\n"
            f"  4. Replicate measurements collapsed onto far fewer compounds than "
            f"expected. Compare numeric_rows with numeric_cpds in the table above: a big "
            f"gap means the same structures are repeated, which aggregation then merges."
        )

    # ----------------------------------------------------------------------- features

    def featurise(self) -> np.ndarray:
        """Standardise, fingerprint and (if the feature set needs them) describe."""
        from .features.molecules import (
            AutoQSARDescriptorSelector,
            build_design_matrix,
            featurise,
        )

        cfg = self.config
        table = self.state.labels
        if table is None:
            raise RuntimeError("call load_labels() first")

        needs_desc = cfg.features.feature_set != "fp_only" or cfg.domain.enabled
        fs = featurise(
            table,
            radius=cfg.features.morgan_radius,
            n_bits=cfg.features.morgan_n_bits,
            standardise=cfg.features.standardise,
            with_descriptors=needs_desc,
            dedupe=cfg.labels.dedupe_on_inchikey,
            progress=self.verbose,
            label="training set",
        )
        self.state.featurised = fs
        self.state.labels = fs.frame
        self.state.fingerprints = fs.fingerprints
        self.state.training_keys = set(fs.frame.get("_key", pd.Series(dtype=str)).dropna())
        if self.verbose and (fs.n_failed or fs.n_duplicates):
            self._say(
                f"  {fs.n_failed} unparseable, {fs.n_duplicates} duplicate structures merged"
            )

        descriptors = fs.descriptors
        if needs_desc and descriptors is not None:
            selector = AutoQSARDescriptorSelector(
                near_constant_frac=cfg.features.descriptor_near_constant_frac,
                corr_threshold=cfg.features.descriptor_corr_threshold,
            ).fit(descriptors, fs.descriptor_names)
            self._say(selector.summary(descriptors.shape[1]))
            descriptors = selector.transform(descriptors)
            self.state.descriptor_selector = selector
            if cfg.features.scale_descriptors:
                from sklearn.preprocessing import StandardScaler

                self.state.descriptor_scaler = StandardScaler().fit(
                    np.nan_to_num(descriptors)
                )
        self.state.descriptors = descriptors

        self.state.X = build_design_matrix(
            fs.fingerprints,
            descriptors,
            feature_set=cfg.features.feature_set,
            scaler=self.state.descriptor_scaler
            if cfg.features.feature_set != "fp_only"
            else None,
        )
        self.state.Y = np.column_stack(
            [fs.frame[f"y_{ep}"].to_numpy(dtype=float) for ep in cfg.endpoints]
        )
        self._say(f"\nDesign matrix: {self.state.X.shape} ({cfg.features.feature_set})")
        return self.state.X

    # --------------------------------------------------------------------- evaluation

    def _enabled_models(self) -> List[str]:
        cfg = self.config
        avail = available_models()
        chosen = [m for m in cfg.models.enabled if m in avail]
        skipped = [m for m in cfg.models.enabled if m not in avail]
        if skipped:
            if not cfg.models.skip_unavailable:
                raise ConfigError(
                    f"models {skipped} are enabled but their dependencies are missing: "
                    f"{ {m: get_spec(m).missing for m in skipped} }"
                )
            self._say(f"  skipping unavailable models {skipped}")
        if not chosen:
            raise ConfigError(
                f"none of models.enabled ({list(cfg.models.enabled)}) is available here. "
                f"Available: {avail}"
            )
        return chosen

    def tune(self) -> Dict[str, Dict[str, Any]]:
        cfg = self.config
        y = self._primary_y()
        self.state.params = tune_all(
            self._enabled_models(),
            self.state.X[np.isfinite(y)],
            y[np.isfinite(y)].astype(int),
            fps=[f for f, k in zip(self.state.fingerprints, np.isfinite(y)) if k],
            fixed=cfg.models.params,
            retune=cfg.models.retune,
            cv_folds=cfg.models.tuning_cv_folds,
            random_state=cfg.evaluation.random_state,
            balanced=cfg.models.class_weight_balanced,
            skip_unavailable=cfg.models.skip_unavailable,
            verbose=self.verbose,
        )
        return self.state.params

    def _primary_y(self) -> np.ndarray:
        cfg = self.config
        idx = cfg.endpoints.index(cfg.labels.primary_endpoint)
        return self.state.Y[:, idx]

    def build_folds(self):
        """Butina clustering, then leave-one-cluster-out folds."""
        cfg = self.config
        fps = self.state.fingerprints
        groups = butina_clusters_from_fingerprints(
            fps, cutoff=cfg.evaluation.butina_cutoff, progress=self.verbose
        )
        from .domain.applicability import min_tanimoto_dissimilarity

        def distance_fn(inside: np.ndarray, outside: np.ndarray) -> float:
            if not len(inside) or not len(outside):
                return float("nan")
            d = min_tanimoto_dissimilarity(
                [fps[i] for i in inside], [fps[i] for i in outside]
            )
            return float(np.mean(d))

        folds = build_loco_folds(
            groups,
            self._primary_y(),
            min_cluster_size=cfg.evaluation.loco_min_cluster_size,
            min_fold_actives=cfg.evaluation.loco_min_fold_actives,
            max_own_folds=cfg.evaluation.loco_max_clusters,
            n_remainder_folds=cfg.evaluation.loco_n_remainder_folds,
            source_tags=self.state.labels["source_tag"].tolist(),
            distance_fn=distance_fn,
        )
        self.state.folds = folds
        if self.verbose:
            print(
                f"\nButina clustering of {len(groups):,} compounds "
                f"(cutoff {cfg.evaluation.butina_cutoff}): {folds.n_clusters} clusters | "
                f"largest {folds.largest_cluster} | singletons {folds.n_singletons}"
            )
            if folds.rejected:
                print(
                    f"  {len(folds.rejected)} large clusters pooled for being too "
                    f"unbalanced to score: {folds.rejected[:6]}"
                )
            print(f"\n{len(folds)} folds covering {folds.coverage:,} compounds:")
            print(folds.to_frame().to_string())
            for f in folds.single_source_folds():
                print(
                    f"  !! {f.name}: {100 * f.source_share:.0f}% from {f.dominant_source} "
                    f"-- holding it out removes a narrow block of training data at the "
                    f"same time as it poses an unfamiliar test set."
                )
        return folds

    def evaluate(self) -> pd.DataFrame:
        """Run the grouped evaluation, calibrate, and choose a threshold."""
        cfg = self.config
        if self.state.folds is None:
            self.build_folds()
        if not self.state.params:
            self.tune()

        models = list(self.state.params)
        loco = run_loco(
            self.state.folds,
            self.state.X,
            self._primary_y(),
            models=models,
            params=self.state.params,
            fps=self.state.fingerprints,
            min_fold_n=cfg.evaluation.loco_min_fold_n,
            random_state=cfg.evaluation.random_state,
            balanced=cfg.models.class_weight_balanced,
            enrichment_fractions=cfg.evaluation.enrichment_fractions,
            precision_ns=cfg.evaluation.precision_at_n,
            verbose=self.verbose,
        )
        self.state.loco = loco
        pooled = loco.pooled(
            enrichment_fractions=cfg.evaluation.enrichment_fractions,
            precision_ns=cfg.evaluation.precision_at_n,
        )
        self.state.pooled = pooled

        reference = random_split_reference(
            self.state.X,
            self._primary_y(),
            models=models,
            params=self.state.params,
            fps=self.state.fingerprints,
            repeats=cfg.evaluation.cv_folds,
            random_state=cfg.evaluation.random_state,
            balanced=cfg.models.class_weight_balanced,
            verbose=self.verbose,
        )
        self.state.reference = reference

        if self.verbose:
            print("\n" + "=" * 100)
            print("POOLED -- every compound scored once, by a model blind to its family.")
            print("QUOTE THESE.")
            print("=" * 100)
            print(pooled.round(3).to_string())
            print("\nLeakage: random split vs pooled grouped")
            print(leakage_table(pooled, reference).to_string())
            print("\n" + "=" * 100)
            print("FLOOR COMPARISONS -- what are the extra models buying?")
            print("=" * 100)
            metrics = ("roc_auc", f"precision@{cfg.evaluation.precision_at_n[0]}")
            for comp in compare_to_floors(
                pooled, floors=cfg.models.floors, metrics=metrics
            ):
                print(comp.summary())
            print("\n" + blend_vs_best_member(pooled))

        # calibration and threshold, both on out-of-fold predictions
        oof = {m: loco.out_of_fold[m][loco.covered] for m in models}
        y_cov = loco.y[loco.covered]
        calibrators, reports = cross_fitted_calibration(
            oof, y_cov,
            method=cfg.evaluation.calibration,
            min_ece=cfg.evaluation.recalibrate_min_ece,
            random_state=cfg.evaluation.random_state,
        )
        self.state.calibrators = calibrators
        self.state.calibration_reports = reports
        if self.verbose:
            print("\n" + pd.DataFrame([r.as_row() for r in reports]).to_string(index=False))
            print("\n" + calibration_caveat(reports))

        for name in models:
            calibrated = calibrators[name].transform(oof[name])
            choice = choose_threshold(
                y_cov, calibrated,
                objective=cfg.evaluation.threshold_objective,
                max_called_frac=cfg.evaluation.threshold_max_called_frac,
                override=cfg.evaluation.threshold_override,
            )
            self.state.threshold_choices[name] = choice
            self.state.thresholds[name] = choice.threshold
            if self.verbose:
                print(f"\n{name}")
                print("  " + choice.summary().replace("\n", "\n  "))
                print(
                    "  "
                    + choice.precision_targets().to_string(index=False).replace("\n", "\n  ")
                )
        return pooled

    # ------------------------------------------------------------------------- deploy

    def fit_final(self) -> Dict[str, Any]:
        """Refit on all labelled data, for deployment.

        Evaluation and the threshold come from out-of-fold predictions; the deployed model
        is then refitted on everything. The two must not be swapped.
        """
        cfg = self.config
        y = self._primary_y()
        keep = np.isfinite(y)
        fps = [f for f, k in zip(self.state.fingerprints, keep) if k]
        for name, params in self.state.params.items():
            self.state.final_models[name] = fit_model(
                name, self.state.X[keep], y[keep].astype(int), params, fps=fps,
                seed=cfg.evaluation.random_state,
                balanced=cfg.models.class_weight_balanced,
            )
        self._say(f"\nFinal models fitted on {int(keep.sum())} compounds: "
                  f"{list(self.state.final_models)}")
        return self.state.final_models

    def fit_domain(self):
        """Calibrate the applicability domain on the carried compounds."""
        if not self.config.domain.enabled:
            return None
        from .domain.applicability import ApplicabilityDomain

        cfg = self.config.domain
        ad = ApplicabilityDomain(
            percentile=cfg.percentile_threshold,
            consensus_min_flags=cfg.consensus_min_flags,
            tanimoto_veto=cfg.tanimoto_veto,
            unfamiliarity_k=cfg.unfamiliarity_k,
            latent_pca_components=cfg.latent_pca_components,
            random_state=self.config.evaluation.random_state,
        ).fit(self.state.fingerprints, self.state.descriptors, progress=self.verbose)
        self.state.domain = ad
        return ad

    # -------------------------------------------------------------------------- saving

    def save(self) -> Path:
        """Write the artefacts, portable formats first.

        The plain-data exports come before the pickles, and a failed pickle is a warning
        rather than an abort: the one save that can fail is the one that has to serialise a
        locally defined class, and when it did, it used to take the config and the metrics
        down with it.
        """
        import joblib

        out = self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        self.config.to_json(out / "config.json")

        if self.state.labels is not None:
            self.state.labels.drop(columns=["_key"], errors="ignore").to_csv(
                out / "training_set.csv", index=False
            )
        if self.state.pooled is not None:
            self.state.pooled.round(4).to_csv(out / "pooled_metrics.csv")
        if self.state.reference is not None:
            self.state.reference.round(4).to_csv(out / "random_split_reference.csv")
        if self.state.loco is not None and not self.state.loco.per_fold.empty:
            self.state.loco.per_fold.round(4).to_csv(out / "per_fold_metrics.csv")
        if self.state.folds is not None:
            self.state.folds.to_frame().to_csv(out / "fold_definitions.csv")
        for name, choice in self.state.threshold_choices.items():
            safe = "".join(c if c.isalnum() else "_" for c in name)
            choice.sweep.round(4).to_csv(out / f"threshold_sweep_{safe}.csv", index=False)

        deployment = {
            "endpoint": self.config.labels.primary_endpoint,
            "active_threshold": self.config.active_cut(self.config.labels.primary_endpoint),
            "feature_set": self.config.features.feature_set,
            "fingerprint": {
                "type": "Morgan",
                "radius": self.config.features.morgan_radius,
                "n_bits": self.config.features.morgan_n_bits,
                "standardisation": "Normalizer -> LargestFragmentChooser -> Uncharger",
            },
            "evaluation": "leave-one-cluster-out (Butina), pooled out-of-fold",
            "models": {},
        }
        for name in self.state.params:
            entry: Dict[str, Any] = {
                "params": self.state.params[name],
                "decision_threshold": self.state.thresholds.get(name),
            }
            cal = self.state.calibrators.get(name)
            if cal is not None:
                entry["calibrator"] = cal.to_dict()
            if self.state.pooled is not None and name in self.state.pooled.index:
                entry["pooled"] = {
                    k: (None if pd.isna(v) else float(v))
                    for k, v in self.state.pooled.loc[name].items()
                }
            deployment["models"][name] = entry
        (out / "deployment.json").write_text(json.dumps(deployment, indent=2, default=str))

        for name, model in self.state.final_models.items():
            safe = "".join(c if c.isalnum() else "_" for c in name)
            try:
                joblib.dump(model, out / f"model_{safe}.joblib")
            except Exception as exc:
                self._say(f"  (could not pickle {name}: {type(exc).__name__}: {exc})")
        self._say(f"\nSaved to {out}")
        return out

    # ----------------------------------------------------------------------------- all

    def run(self, long_frame=None, wide_frame=None) -> "Pipeline":
        """Everything except screening."""
        self.load_labels(long_frame, wide_frame)
        self.featurise()
        self.tune()
        self.build_folds()
        self.evaluate()
        self.fit_final()
        self.fit_domain()
        self.save()
        return self
