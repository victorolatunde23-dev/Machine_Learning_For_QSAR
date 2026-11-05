"""Configuration: nested dataclasses, YAML overlay, and validation.

Why this module is strict
------------------------
The pipeline this package came from lost a full run to a configuration bug: a cell read
``cfg.auto_resolve_bindingdb_url`` from a config object that did not define it, the
BindingDB download silently fell back to a synthetic dataset, and the failure surfaced six
steps later as ``n_splits=10 greater than the number of samples: n_samples=3``. Nothing in
between complained.

So two rules are enforced here:

* **Unknown keys are errors.** A typo in a YAML file is reported with the offending key,
  its section, and the closest valid names, rather than being silently ignored.
* **Cross-field consistency is checked up front.** ``validate()`` catches the combinations
  that would otherwise fail deep inside a fit, and :func:`load_config` runs it by default.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "Config",
    "DataConfig",
    "LabelConfig",
    "FeatureConfig",
    "ModelConfig",
    "TransferConfig",
    "EvalConfig",
    "DomainConfig",
    "ScreenConfig",
    "load_config",
    "ConfigError",
]

FEATURE_SETS = ("fp_only", "descriptors_only", "fp_plus_descriptors")
TASKS = ("classification", "regression")
SCREEN_SOURCES = ("none", "file", "enumeration", "enumeration+library")
CALIBRATIONS = ("none", "sigmoid", "isotonic")
THRESHOLD_OBJECTIVES = ("mcc", "f1", "balanced_accuracy", "youden")


class ConfigError(ValueError):
    """Invalid configuration: an unknown key, a bad value, or an inconsistent pair."""


# ---------------------------------------------------------------------------- sections


@dataclass
class DataConfig:
    """Where the measurements come from."""

    source: str = "bindingdb"  # "bindingdb" | "csv"
    # A local CSV instead of a BindingDB pull. Needs smiles + one column per endpoint.
    csv_path: Optional[str] = None

    # BindingDB SQL filter.
    target_uniprot_id: Optional[str] = "Q04609"  # PSMA / FOLH1 / GCPII
    target_name_like: Optional[str] = "Glutamate carboxypeptidase 2"
    organism_like: Optional[str] = "Homo sapiens"

    endpoint_names: Sequence[str] = field(default_factory=lambda: ["pIC50", "pKi"])
    affinity_columns: Sequence[str] = field(
        default_factory=lambda: ["IC50 (nM)", "Ki (nM)"]
    )
    min_affinity_nm: float = 1e-3
    max_affinity_nm: float = 1e6

    # Acquisition. auto_resolve discovers the current release rather than trusting a
    # pinned filename, which goes stale monthly.
    bindingdb_tsv_zip_name: str = "BindingDB_All_202609_tsv.zip"
    bindingdb_download_url: str = (
        "https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202609_tsv.zip"
    )
    auto_resolve_bindingdb_url: bool = True
    force_redownload: bool = False
    # When the download fails: "raise" stops, "synthetic" substitutes random molecules.
    # "synthetic" is for exercising the plumbing only and prints a loud warning; it is not
    # the default because a silent fallback is how a run gets read as real.
    on_download_failure: str = "raise"


@dataclass
class LabelConfig:
    """Turning measurements into labels. The derived-value removal lives here."""

    task: str = "classification"
    modelled_endpoints: Sequence[str] = field(default_factory=lambda: ["pKi"])
    primary_endpoint: str = "pKi"
    # Active cut per endpoint, in p-units (pKi >= 8.0 is Ki <= 10 nM).
    active_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"pIC50": 8.0, "pKi": 8.0}
    )

    # Cheng-Prusoff detection. A block of compounds whose two endpoints differ by a FIXED
    # amount was not measured twice: the second value was computed from the first. In the
    # PSMA set that is 141 of 169 dual-measured compounds, and keeping them puts a large
    # part of the pKi column on a different scale from the rest.
    drop_derived_endpoint_values: bool = True
    derived_endpoint: Optional[str] = "pKi"  # the endpoint whose values may be computed
    derived_min_block_size: int = 10
    derived_offset_tol: float = 0.03

    # A ">" bound at or below the active cut proves "not active"; a "<" bound at or above
    # it proves "active". Both are usable labels. Bounds straddling the cut are dropped.
    keep_censored_as_labels: bool = True

    # Carry compounds measured only on an unmodelled endpoint. They cannot train the
    # model but they still define the applicability domain and shape the clustering.
    keep_unlabelled_for_domain: bool = False
    dedupe_on_inchikey: bool = True


@dataclass
class FeatureConfig:
    """Molecular representation.

    ``fp_only`` is the default on evidence: on the PSMA pKi set the bake-off gave
    fp_only 0.865, fp_plus_descriptors 0.850, descriptors_only 0.844, and the descriptors
    are also by far the most expensive thing to compute over a large library.
    """

    feature_set: str = "fp_only"
    morgan_radius: int = 2
    morgan_n_bits: int = 2048
    standardise: bool = True  # normalise -> largest fragment -> uncharge
    descriptor_corr_threshold: float = 0.8
    descriptor_near_constant_frac: float = 0.90
    scale_descriptors: bool = True


@dataclass
class ModelConfig:
    """Which classical models to fit, and how hard to tune them."""

    enabled: Sequence[str] = field(
        default_factory=lambda: ["LightGBM", "RandomForest", "LogReg (L2)", "kNN (Tanimoto)"]
    )
    # Members of the equal-weight blend, per endpoint. Equal weighting beat fitted
    # simplex weights in all six comparisons tried, so there is no weight-fitting option.
    committee: Dict[str, Sequence[str]] = field(default_factory=dict)
    # Floors every model is reported against. kNN is similarity lookup; LogReg is linear.
    # A model that cannot clear both is not earning its complexity.
    floors: Sequence[str] = field(
        default_factory=lambda: ["kNN (Tanimoto)", "LogReg (L2)"]
    )
    class_weight_balanced: bool = True
    retune: bool = True
    tuning_cv_folds: int = 10
    # Fixed hyperparameters per model; used when retune is False, or as the starting point.
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    n_bootstrap: int = 0  # bootstrap ensemble size, 0 to skip
    # Skip a model whose dependency is missing rather than failing the run.
    skip_unavailable: bool = True


@dataclass
class TransferConfig:
    """The multi-task neural network and its ChEMBL pretraining.

    Off by default, and that is a finding rather than a preference: two ablations in
    opposite directions found pretraining bought nothing (+0.062 R2 on random regression
    splits and nothing on grouped ones; -0.0066 AUC on classification). See
    docs/RESULTS.md. The code is kept because the ablation is reproducible.
    """

    enabled: bool = False
    use_pretraining: bool = True
    run_pretraining_ablation: bool = False

    hidden_dims: Sequence[int] = field(default_factory=lambda: [512, 256])
    dropout: float = 0.25
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    pretrain_epochs: int = 40
    pretrain_batch_size: int = 128
    finetune_epochs: int = 60
    freeze_epochs: int = 10
    mc_dropout_passes: int = 20

    # Pretraining corpus.
    chembl_target_ids: Sequence[str] = field(default_factory=list)
    chembl_cache_path: Optional[str] = None
    chembl_min_rows: int = 200


@dataclass
class EvalConfig:
    """Grouped-split evaluation, calibration and the decision threshold."""

    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5

    # Butina leave-one-cluster-out. Holding out a whole chemical family at a time is what
    # makes the estimate prospective; a random split leaves close analogues on both sides
    # and inflated ROC-AUC by 0.06-0.13 depending on the model.
    butina_cutoff: float = 0.4
    loco_min_cluster_size: int = 25
    loco_min_fold_actives: int = 5
    loco_min_fold_n: int = 10  # below this a fold joins the pool but gets no reported row
    loco_max_clusters: int = 12
    loco_n_remainder_folds: int = 3
    loco_random_reference_repeats: int = 10

    calibration: str = "sigmoid"
    # Leave well-calibrated probabilities alone. Recalibrating pKi anyway once made the
    # blend worse (ECE 0.084 -> 0.167).
    recalibrate_min_ece: float = 0.10

    threshold_objective: str = "mcc"
    threshold_override: Optional[float] = None
    # A cut is only allowed if it calls at most this fraction active. Without it, MCC and
    # F1 both maximise at a permissive threshold on a weak model: one run chose p >= 0.11,
    # calling 86% of compounds active at recall 1.000, which over a 290k library is a
    # 250k-compound "shortlist". The point of a screen is to decide what not to look at.
    threshold_max_called_frac: float = 0.25

    enrichment_fractions: Sequence[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.10]
    )
    precision_at_n: Sequence[int] = field(default_factory=lambda: [50, 100, 200])

    # Stop early if the labelled set is too small to evaluate honestly.
    min_train_compounds: int = 50
    min_per_class: int = 10


@dataclass
class DomainConfig:
    """Applicability domain: four distance metrics, with Tanimoto as a veto."""

    enabled: bool = True
    percentile_threshold: float = 95.0
    consensus_min_flags: int = 3
    # Tanimoto is a veto, not one vote in four. Using it as one vote let every top hit of
    # an earlier run pass while failing the structural test (dissimilarity 0.73-0.83
    # against a 0.457 threshold) on the strength of bulk-property similarity alone.
    tanimoto_veto: bool = True
    unfamiliarity_k: int = 5
    latent_pca_components: int = 10


@dataclass
class ScreenConfig:
    """The library to score, and how."""

    source: str = "none"
    library_path: Optional[str] = None
    smiles_col: Optional[str] = None  # None sniffs the column
    id_col: Optional[str] = None
    chunk_size: int = 25000  # molecules in memory at once while streaming
    limit: Optional[int] = None  # cap for a trial run
    shortlist_size: int = 500
    n_bootstrap_shortlist: int = 20  # spread on the shortlist only

    prefilter: bool = False
    prefilter_min_heavy_atoms: int = 12
    prefilter_mw_range: Sequence[float] = field(default_factory=lambda: [150.0, 700.0])

    # Reaction-based enumeration.
    enum_max_products: Optional[int] = None
    enum_property_quantile: float = 0.05  # window = this..1-this of the known actives
    enum_apply_property_filter: bool = True
    # Floor on that filter: below this kept fraction it is reported and flagged rather
    # than applied, so a narrow window cannot silently empty the library.
    enum_property_min_keep_frac: float = 0.20

    # Mixing catalogue compounds in as controls.
    mix_library_path: Optional[str] = None
    mix_pool_sample: int = 60000
    mix_n_near: int = 5000  # nearest to training: the scaffold-hop candidates
    mix_n_matched: int = 10000  # property-matched decoys
    mix_n_random: int = 5000  # unbiased background
    mix_mw_tolerance: float = 25.0


@dataclass
class Config:
    """The whole run."""

    run_name: str = "run"
    output_dir: str = "./outputs"
    data_dir: str = "./data"
    random_state: int = 42

    data: DataConfig = field(default_factory=DataConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, strict: bool = True) -> "Config":
        """Overlay a nested mapping onto the defaults."""
        return _build(cls, data, strict=strict, path="")

    @classmethod
    def from_yaml(cls, path, *, strict: bool = True, validate: bool = True) -> "Config":
        import yaml

        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        raw = yaml.safe_load(p.read_text()) or {}
        if not isinstance(raw, Mapping):
            raise ConfigError(f"{p} must contain a YAML mapping, got {type(raw).__name__}")
        cfg = cls.from_dict(raw, strict=strict)
        if validate:
            cfg.validate()
        return cfg

    # ------------------------------------------------------------------ serialisation

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path) -> Path:
        import yaml

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False))
        return p

    def to_json(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return p

    # ------------------------------------------------------------------ derived values

    @property
    def endpoints(self) -> List[str]:
        """The modelled endpoints, in order. Column 0 is the primary."""
        return list(self.labels.modelled_endpoints)

    @property
    def is_multitask(self) -> bool:
        return len(self.endpoints) > 1

    def active_cut(self, endpoint: str) -> float:
        try:
            return float(self.labels.active_thresholds[endpoint])
        except KeyError as exc:
            raise ConfigError(
                f"no active_threshold for endpoint {endpoint!r}; "
                f"have {sorted(self.labels.active_thresholds)}"
            ) from exc

    def committee_for(self, endpoint: str) -> List[str]:
        """Blend members for an endpoint, defaulting to every enabled model."""
        members = self.models.committee.get(endpoint)
        if not members:
            return list(self.models.enabled)
        return [m for m in members if m in self.models.enabled]

    def ensure_dirs(self) -> "Config":
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        return self

    # ------------------------------------------------------------------ validation

    def validate(self) -> "Config":
        """Raise :class:`ConfigError` on anything that would fail later.

        Every check here corresponds to a failure actually observed while developing the
        pipeline, which is why they are checked at load time rather than at use time.
        """
        from .models.registry import available_models, known_models

        errs: List[str] = []

        if self.labels.task not in TASKS:
            errs.append(f"labels.task must be one of {TASKS}, got {self.labels.task!r}")
        if self.features.feature_set not in FEATURE_SETS:
            errs.append(
                f"features.feature_set must be one of {FEATURE_SETS}, "
                f"got {self.features.feature_set!r}"
            )
        if self.screen.source not in SCREEN_SOURCES:
            errs.append(
                f"screen.source must be one of {SCREEN_SOURCES}, got {self.screen.source!r}"
            )
        if self.evaluation.calibration not in CALIBRATIONS:
            errs.append(
                f"evaluation.calibration must be one of {CALIBRATIONS}, "
                f"got {self.evaluation.calibration!r}"
            )
        if self.evaluation.threshold_objective not in THRESHOLD_OBJECTIVES:
            errs.append(
                f"evaluation.threshold_objective must be one of {THRESHOLD_OBJECTIVES}, "
                f"got {self.evaluation.threshold_objective!r}"
            )
        if self.data.on_download_failure not in ("raise", "synthetic"):
            errs.append(
                "data.on_download_failure must be 'raise' or 'synthetic', got "
                f"{self.data.on_download_failure!r}"
            )

        # Endpoints ------------------------------------------------------------
        if not self.labels.modelled_endpoints:
            errs.append("labels.modelled_endpoints is empty; nothing would be modelled")
        unknown_eps = [
            e for e in self.labels.modelled_endpoints if e not in self.data.endpoint_names
        ]
        if unknown_eps:
            errs.append(
                f"labels.modelled_endpoints {unknown_eps} are not in "
                f"data.endpoint_names {list(self.data.endpoint_names)}"
            )
        if self.labels.primary_endpoint not in self.labels.modelled_endpoints:
            errs.append(
                f"labels.primary_endpoint {self.labels.primary_endpoint!r} is not in "
                f"modelled_endpoints {list(self.labels.modelled_endpoints)}"
            )
        for ep in self.labels.modelled_endpoints:
            if ep not in self.labels.active_thresholds:
                errs.append(f"labels.active_thresholds has no entry for {ep!r}")
        if len(self.data.endpoint_names) != len(self.data.affinity_columns):
            errs.append(
                f"data.endpoint_names ({len(self.data.endpoint_names)}) and "
                f"affinity_columns ({len(self.data.affinity_columns)}) differ in length"
            )

        # The bug this package is most careful about. With pKi promoted to primary, a
        # derived-value test keyed off the *primary* endpoint keeps every Cheng-Prusoff
        # value instead of dropping it, which is silent and changes every downstream
        # number.
        if self.labels.drop_derived_endpoint_values:
            if self.labels.derived_endpoint is None:
                errs.append(
                    "labels.drop_derived_endpoint_values is true but derived_endpoint is "
                    "null: there is nothing to drop. Set derived_endpoint to the endpoint "
                    "whose values may have been computed (usually pKi)."
                )
            elif self.labels.derived_endpoint not in self.data.endpoint_names:
                errs.append(
                    f"labels.derived_endpoint {self.labels.derived_endpoint!r} is not in "
                    f"data.endpoint_names, so the detector would never fire"
                )
            elif len(self.data.endpoint_names) < 2:
                errs.append(
                    "derived-value detection needs two endpoints pulled (it looks for a "
                    "fixed offset between them); data.endpoint_names has one"
                )

        # Models ---------------------------------------------------------------
        if not self.models.enabled and not self.transfer.enabled:
            errs.append("no models enabled and transfer.enabled is false: nothing to fit")
        unknown_models = [m for m in self.models.enabled if m not in known_models()]
        if unknown_models:
            close = {
                m: difflib.get_close_matches(m, known_models(), n=2, cutoff=0.5)
                for m in unknown_models
            }
            errs.append(
                f"models.enabled contains unknown models {unknown_models}; "
                f"did you mean {close}? Known: {known_models()}"
            )
        for ep, members in self.models.committee.items():
            if ep not in self.labels.modelled_endpoints:
                errs.append(
                    f"models.committee has an entry for {ep!r}, which is not a modelled "
                    f"endpoint {list(self.labels.modelled_endpoints)}"
                )
            orphans = [m for m in members if m not in self.models.enabled and m != "NN"]
            if orphans:
                errs.append(
                    f"models.committee[{ep!r}] names {orphans}, which are not in "
                    f"models.enabled, so they would silently drop out of the blend"
                )
        orphan_floors = [f for f in self.models.floors if f not in self.models.enabled]
        if orphan_floors:
            errs.append(
                f"models.floors names {orphan_floors}, which are not enabled. A floor has "
                f"to be fitted to be compared against; add it to models.enabled."
            )
        if self.transfer.enabled and "NN" not in list(self.models.enabled) + ["NN"]:
            pass  # NN is addressed by transfer.enabled, not models.enabled

        # Screening ------------------------------------------------------------
        if self.screen.source == "file" and not self.screen.library_path:
            errs.append("screen.source is 'file' but screen.library_path is not set")
        if self.screen.source == "enumeration+library" and not self.screen.mix_library_path:
            errs.append(
                "screen.source is 'enumeration+library' but screen.mix_library_path is "
                "not set; that is the catalogue mixed in as controls"
            )
        if not 0.0 < self.evaluation.threshold_max_called_frac <= 1.0:
            errs.append(
                "evaluation.threshold_max_called_frac must be in (0, 1]; it caps the "
                "fraction a chosen threshold may call active"
            )
        if self.evaluation.threshold_override is not None and not (
            0.0 <= self.evaluation.threshold_override <= 1.0
        ):
            errs.append("evaluation.threshold_override must be a probability in [0, 1]")
        if self.domain.consensus_min_flags < 1 or self.domain.consensus_min_flags > 4:
            errs.append("domain.consensus_min_flags must be between 1 and 4")
        if self.screen.chunk_size < 1:
            errs.append("screen.chunk_size must be positive")

        if errs:
            raise ConfigError(
                "invalid configuration:\n" + "\n".join(f"  - {e}" for e in errs)
            )
        return self

    def capability_report(self) -> str:
        """One-line-per-item summary of what this config can and cannot do here."""
        from ._optional import have
        from .models.registry import available_models

        lines = [f"run: {self.run_name}"]
        lines.append(
            f"  task          : {self.labels.task}, endpoints "
            f"{self.endpoints} (primary {self.labels.primary_endpoint})"
        )
        lines.append(f"  features      : {self.features.feature_set}")
        avail = available_models()
        enabled_ok = [m for m in self.models.enabled if m in avail]
        enabled_no = [m for m in self.models.enabled if m not in avail]
        lines.append(f"  models        : {enabled_ok}")
        if enabled_no:
            how = "skipped" if self.models.skip_unavailable else "WILL FAIL"
            lines.append(f"  unavailable   : {enabled_no}  ({how}; missing dependency)")
        if self.transfer.enabled:
            lines.append(
                f"  transfer      : on (pretraining={self.transfer.use_pretraining}, "
                f"torch {'present' if have('torch') else 'MISSING'})"
            )
        lines.append(
            f"  domain        : {'on' if self.domain.enabled else 'off'}"
            + (f" (tanimoto veto)" if self.domain.enabled and self.domain.tanimoto_veto else "")
        )
        lines.append(f"  screen        : {self.screen.source}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------- overlay


def _build(cls, data: Mapping[str, Any], *, strict: bool, path: str):
    """Recursively construct a dataclass from a mapping, rejecting unknown keys."""
    if not isinstance(data, Mapping):
        raise ConfigError(
            f"{path or 'config'} must be a mapping, got {type(data).__name__}: {data!r}"
        )
    spec = {f.name: f for f in fields(cls)}
    unknown = [k for k in data if k not in spec]
    if unknown and strict:
        suggestions = {
            k: difflib.get_close_matches(str(k), list(spec), n=3, cutoff=0.4)
            for k in unknown
        }
        where = path or "top level"
        detail = "; ".join(
            f"{k!r}" + (f" (did you mean {s}?)" if s else "") for k, s in suggestions.items()
        )
        raise ConfigError(
            f"unknown config key(s) at {where}: {detail}\n"
            f"  valid keys here: {sorted(spec)}\n"
            f"  Unknown keys are rejected rather than ignored on purpose: a silently "
            f"dropped key is indistinguishable from a setting that had no effect."
        )

    kwargs: Dict[str, Any] = {}
    for name, f in spec.items():
        if name not in data:
            continue
        value = data[name]
        if is_dataclass(f.type) or (
            isinstance(f.type, str) and f.type in _SECTION_TYPES and isinstance(value, Mapping)
        ):
            sub = _SECTION_TYPES[f.type] if isinstance(f.type, str) else f.type
            kwargs[name] = _build(
                sub, value, strict=strict, path=f"{path}.{name}" if path else name
            )
        else:
            kwargs[name] = value
    return cls(**kwargs)


_SECTION_TYPES = {
    "DataConfig": DataConfig,
    "LabelConfig": LabelConfig,
    "FeatureConfig": FeatureConfig,
    "ModelConfig": ModelConfig,
    "TransferConfig": TransferConfig,
    "EvalConfig": EvalConfig,
    "DomainConfig": DomainConfig,
    "ScreenConfig": ScreenConfig,
}


def load_config(path=None, *, overrides: Optional[Mapping[str, Any]] = None,
                validate: bool = True) -> Config:
    """Load a config file, apply ``overrides``, and validate.

    ``overrides`` uses dotted keys, so ``{"evaluation.butina_cutoff": 0.5}`` works, which
    is what the CLI's ``--set`` flag passes through.
    """
    cfg = Config.from_yaml(path, validate=False) if path else Config()
    if overrides:
        cfg = _apply_overrides(cfg, overrides)
    if validate:
        cfg.validate()
    return cfg


def _apply_overrides(cfg: Config, overrides: Mapping[str, Any]) -> Config:
    nested: Dict[str, Any] = {}
    for dotted, value in overrides.items():
        parts = str(dotted).split(".")
        cursor = nested
        for p in parts[:-1]:
            cursor = cursor.setdefault(p, {})
        cursor[parts[-1]] = value
    merged = _deep_merge(cfg.to_dict(), nested)
    return Config.from_dict(merged)


def _deep_merge(base: Dict[str, Any], extra: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(dict(out[k]), v)
        else:
            out[k] = v
    return out
