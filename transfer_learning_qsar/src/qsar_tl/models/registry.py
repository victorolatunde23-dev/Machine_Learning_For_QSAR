"""The model zoo: several classifiers behind one fit/predict interface.

Adding a model means adding one :class:`ModelSpec`; the tuning, fold, screening and
saving code all iterate over the registry. Each spec declares the optional dependency it
needs, so a model whose package is missing is *skipped with a reason* rather than
exploding mid-run -- and :func:`known_models` lets the config validate a model list
without importing anything heavy.

Two of these are here as much for diagnosis as for prediction. ``kNN (Tanimoto)`` is
nearest-neighbour lookup and ``LogReg (L2)`` is a linear model on fingerprints; they are
the floors every other model is reported against. See
:mod:`qsar_tl.evaluation.floors`.

Every model returns a full ``(n, 2)`` probability matrix even on a fold that never saw one
of the classes, so an equal-weight blend is always well defined. Leave-one-cluster-out
produces badly unbalanced folds routinely and used to break members that assumed
otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .._optional import have, require, tqdm

__all__ = [
    "ModelSpec",
    "MODEL_ZOO",
    "known_models",
    "available_models",
    "unavailable_models",
    "get_spec",
    "fit_model",
    "predict_proba",
    "bootstrap_fit",
    "tanimoto_gram",
    "N_CLASSES",
]

N_CLASSES = 2
POSITIVE_CLASS = 1


# ------------------------------------------------------------------------------- helpers


def _align_proba(p: np.ndarray, classes) -> np.ndarray:
    """Pad to the full class set so a single-class fold still returns ``(n, 2)``."""
    p = np.asarray(p, dtype=float)
    classes = list(classes)
    if p.shape[1] == N_CLASSES and classes == list(range(N_CLASSES)):
        return p
    out = np.zeros((p.shape[0], N_CLASSES))
    for j, c in enumerate(classes):
        out[:, int(c)] = p[:, j]
    return out


def _class_weight(balanced: bool) -> Optional[str]:
    return "balanced" if balanced else None


class ConstantClassifier:
    """Predicts one class with certainty. Used when a training set has only one class.

    Several estimators -- logistic regression and therefore KPLS-DA among them -- refuse to
    fit single-class data outright, while trees fit it happily. Leave-one-cluster-out
    produces such folds routinely, and without this the two groups would behave
    differently: the trees would contribute a prediction and the rest would raise, be
    caught by the fold loop, and silently leave holes in their out-of-fold column. Uneven
    coverage between models is far worse than a trivial prediction, because it makes the
    pooled metrics incomparable without saying so.
    """

    def __init__(self, cls: int, n_classes: int = N_CLASSES):
        self.cls = int(cls)
        self.n_classes = int(n_classes)
        self.classes_ = np.array([self.cls])

    def predict_proba(self, X) -> np.ndarray:
        out = np.zeros((len(X), self.n_classes))
        out[:, self.cls] = 1.0
        return out

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.cls, dtype=int)


def tanimoto_gram(query_fps, ref_fps, *, dtype=np.float32) -> np.ndarray:
    """``(n_query, n_ref)`` Tanimoto similarities via RDKit's bulk routine.

    float32 halves the memory, which matters: a library Gram against a few hundred
    training compounds is otherwise hundreds of megabytes. Callers scoring a whole library
    go through :func:`predict_proba`, which chunks.
    """
    DataStructs = require("rdkit").DataStructs  # type: ignore[attr-defined]
    ref = list(ref_fps)
    K = np.empty((len(query_fps), len(ref)), dtype=dtype)
    for i, q in enumerate(query_fps):
        K[i] = DataStructs.BulkTanimotoSimilarity(q, ref)
    return K


# -------------------------------------------------------------------------------- models


def _fit_lightgbm(X, y, params, fps=None, sample_weight=None, seed=42, balanced=True):
    LGBMClassifier = require("lightgbm").LGBMClassifier  # type: ignore[attr-defined]
    m = LGBMClassifier(
        random_state=seed, verbose=-1, n_jobs=-1,
        class_weight=_class_weight(balanced), **params,
    )
    m.fit(X, y, sample_weight=sample_weight)
    return m


def _proba_sklearn(m, X, fps=None):
    return _align_proba(m.predict_proba(X), m.classes_)


def _fit_random_forest(X, y, params, fps=None, sample_weight=None, seed=42, balanced=True):
    from sklearn.ensemble import RandomForestClassifier

    m = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_features=params.get("max_features", "sqrt"),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        class_weight="balanced_subsample" if balanced else None,
        n_jobs=-1,
        random_state=seed,
    )
    m.fit(X, y, sample_weight=sample_weight)
    return m


def _fit_kpls(X, y, params, fps=None, sample_weight=None, seed=42, balanced=True):
    """Nystroem kernel map -> PLS latent space -> logistic layer (kernel PLS-DA).

    After An, Sherman & Dixon (2013). Landmarks are clamped to the sample count because
    leave-one-cluster-out produces small training halves.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import LogisticRegression

    gamma = params.get("gamma") or 1.0 / max(1, np.asarray(X).shape[1])
    n_landmarks = int(min(params.get("n_landmarks", 200), max(2, len(X) - 1)))
    nys = Nystroem(kernel="rbf", gamma=gamma, n_components=n_landmarks, random_state=seed)
    Z = nys.fit_transform(X)
    n_comp = max(1, min(params.get("n_components", 4), min(50, Z.shape[1] - 1)))
    y = np.asarray(y, dtype=int)
    one_hot = np.zeros((len(y), N_CLASSES))
    one_hot[np.arange(len(y)), y] = 1.0
    pls = PLSRegression(n_components=n_comp).fit(Z, one_hot)
    logit = LogisticRegression(max_iter=2000, class_weight=_class_weight(balanced))
    logit.fit(pls.transform(Z), y, sample_weight=sample_weight)
    return {"nystroem": nys, "pls": pls, "logit": logit, "classes_": logit.classes_}


def _proba_kpls(m, X, fps=None):
    T = m["pls"].transform(m["nystroem"].transform(X))
    return _align_proba(m["logit"].predict_proba(T), m["classes_"])


def _fit_svm_tanimoto(X, y, params, fps=None, sample_weight=None, seed=42, balanced=True):
    """SVC on a *precomputed* Tanimoto kernel -- the chemists' own similarity measure.

    ``probability=True`` runs sklearn's internal Platt cross-validation, so this returns
    probabilities rather than decision margins.
    """
    from sklearn.svm import SVC

    fps = list(fps)
    K = tanimoto_gram(fps, fps)
    m = SVC(
        C=params.get("C", 1.0), kernel="precomputed", probability=True,
        class_weight=_class_weight(balanced), random_state=seed,
    )
    m.fit(K, y, sample_weight=sample_weight)
    return {"svc": m, "train_fps": fps}


def _proba_svm_tanimoto(m, X, fps=None):
    K = tanimoto_gram(fps, m["train_fps"])
    return _align_proba(m["svc"].predict_proba(K), m["svc"].classes_)


def _fit_knn_tanimoto(X, y, params, fps=None, sample_weight=None, seed=42, balanced=True):
    """Lazy learner: keeps the training fingerprints and labels."""
    return {
        "fps": list(fps),
        "y": np.asarray(y, dtype=int),
        "k": int(params.get("k", 5)),
    }


def _proba_knn_tanimoto(m, X, fps=None):
    """Similarity-weighted vote over the k nearest training compounds."""
    S = tanimoto_gram(fps, m["fps"])
    k = int(min(m["k"], S.shape[1]))
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    rows = np.arange(S.shape[0])[:, None]
    w = S[rows, idx].astype(np.float64) + 1e-9
    hit = (m["y"][idx] == POSITIVE_CLASS).astype(np.float64)
    p1 = (w * hit).sum(axis=1) / w.sum(axis=1)
    return np.column_stack([1.0 - p1, p1])


def _fit_logreg(X, y, params, fps=None, sample_weight=None, seed=42, balanced=True):
    from sklearn.linear_model import LogisticRegression

    # L2 is the default; naming it explicitly raises a FutureWarning on sklearn >= 1.8.
    m = LogisticRegression(
        C=params.get("C", 1.0), max_iter=3000,
        class_weight=_class_weight(balanced), random_state=seed,
    )
    m.fit(X, y, sample_weight=sample_weight)
    return m


def _fit_bernoulli_nb(X, y, params, fps=None, sample_weight=None, seed=42, balanced=True):
    """Laplacian-corrected naive Bayes on binarised features.

    With ``feature_set="fp_only"`` those are exactly the fingerprint bits, which is the
    classical Pipeline Pilot formulation. With descriptors in the matrix the binarisation
    of those columns is crude but harmless.
    """
    from sklearn.naive_bayes import BernoulliNB

    m = BernoulliNB(alpha=params.get("alpha", 1.0), binarize=0.5)
    m.fit(X, y, sample_weight=sample_weight)
    return m


# ------------------------------------------------------------------------------ registry


@dataclass
class ModelSpec:
    """How to fit one model, tune it, and what it needs installed."""

    name: str
    fit: Callable[..., Any]
    proba: Callable[..., np.ndarray]
    grid: List[Dict[str, Any]] = field(default_factory=list)
    needs_fps: bool = False
    requires: Sequence[str] = ()  # optional modules
    note: str = ""

    @property
    def available(self) -> bool:
        return all(have(mod) for mod in self.requires)

    @property
    def missing(self) -> List[str]:
        return [m for m in self.requires if not have(m)]


MODEL_ZOO: Dict[str, ModelSpec] = {
    "LightGBM": ModelSpec(
        name="LightGBM",
        fit=_fit_lightgbm,
        proba=_proba_sklearn,
        requires=("lightgbm",),
        grid=[
            {"n_estimators": n, "learning_rate": lr, "num_leaves": nl, "max_depth": -1}
            for n in (100, 200, 400)
            for lr in (0.02, 0.05, 0.1)
            for nl in (15, 31, 63)
        ],
        note="gradient-boosted trees; strongest single model on the PSMA pKi set (0.804)",
    ),
    "RandomForest": ModelSpec(
        name="RandomForest",
        fit=_fit_random_forest,
        proba=_proba_sklearn,
        grid=[
            {"n_estimators": n, "max_features": mf, "min_samples_leaf": ml}
            for n in (300, 600)
            for mf in ("sqrt", 0.05, 0.1)
            for ml in (1, 3)
        ],
        note="bagged trees; best MCC and the only strong model needing no recalibration",
    ),
    "KPLS-DA": ModelSpec(
        name="KPLS-DA",
        fit=_fit_kpls,
        proba=_proba_kpls,
        grid=[
            {"n_components": nc, "gamma": g}
            for nc in (2, 4, 6, 8, 12)
            for g in (None, 0.01, 0.05, 0.1)
        ],
        note="Nystroem -> PLS -> logistic; best at the very top of the ranking, worst ECE",
    ),
    "SVM (Tanimoto)": ModelSpec(
        name="SVM (Tanimoto)",
        fit=_fit_svm_tanimoto,
        proba=_proba_svm_tanimoto,
        needs_fps=True,
        requires=("rdkit",),
        grid=[{"C": c} for c in (0.1, 1.0, 10.0)],
        note="precomputed Tanimoto kernel; most consistent across families",
    ),
    "kNN (Tanimoto)": ModelSpec(
        name="kNN (Tanimoto)",
        fit=_fit_knn_tanimoto,
        proba=_proba_knn_tanimoto,
        needs_fps=True,
        requires=("rdkit",),
        grid=[{"k": k} for k in (1, 3, 5, 9)],
        note="FLOOR, not a contender: similarity lookup. What models gain over it is real",
    ),
    "LogReg (L2)": ModelSpec(
        name="LogReg (L2)",
        fit=_fit_logreg,
        proba=_proba_sklearn,
        grid=[{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
        note="FLOOR: the linear model. If ensembles cannot beat it, complexity is not paying",
    ),
    "BernoulliNB": ModelSpec(
        name="BernoulliNB",
        fit=_fit_bernoulli_nb,
        proba=_proba_sklearn,
        grid=[{"alpha": a} for a in (0.1, 1.0, 5.0)],
        note="near-free; badly overconfident on this data (ECE 0.389)",
    ),
}


def known_models() -> List[str]:
    """Every registered model name, regardless of whether its dependency is installed."""
    return list(MODEL_ZOO)


def available_models() -> List[str]:
    """Models whose optional dependencies are importable right now."""
    return [n for n, s in MODEL_ZOO.items() if s.available]


def unavailable_models() -> Dict[str, List[str]]:
    """``{model: [missing modules]}`` for models that cannot be fitted here."""
    return {n: s.missing for n, s in MODEL_ZOO.items() if not s.available}


def get_spec(name: str) -> ModelSpec:
    try:
        return MODEL_ZOO[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown model {name!r}; known models are {known_models()}"
        ) from exc


# ------------------------------------------------------------------------- fit / predict


def fit_model(
    name: str,
    X,
    y,
    params: Optional[Dict[str, Any]] = None,
    *,
    fps=None,
    sample_weight=None,
    seed: int = 42,
    balanced: bool = True,
):
    """Fit one model by name. Fingerprints are passed only to models that want them."""
    spec = get_spec(name)
    if not spec.available:
        raise ImportError(
            f"{name} needs {spec.missing}, which is not installed. "
            f"Install it, or drop {name!r} from models.enabled."
        )
    if spec.needs_fps and fps is None:
        raise ValueError(f"{name} needs fingerprints but none were passed")
    y = np.asarray(y, dtype=int)
    classes = np.unique(y)
    if len(classes) < 2:
        # Uniform behaviour across the zoo; see ConstantClassifier for why.
        return ConstantClassifier(int(classes[0]) if len(classes) else 0)
    return spec.fit(
        X,
        np.asarray(y, dtype=int),
        dict(params or {}),
        fps=fps if spec.needs_fps else None,
        sample_weight=sample_weight,
        seed=seed,
        balanced=balanced,
    )


def predict_proba(
    name: str, model, X, *, fps=None, chunk: Optional[int] = None
) -> np.ndarray:
    """``(n, 2)`` probabilities, chunked for models that need a kernel against training.

    Without chunking, scoring a large library with the Tanimoto SVM or kNN would
    materialise a ``n_library x n_train`` Gram in one allocation. ``chunk`` keeps peak
    memory at ``chunk x n_train``.
    """
    spec = get_spec(name)
    if isinstance(model, ConstantClassifier):
        return model.predict_proba(X)
    if not spec.needs_fps:
        return spec.proba(model, X, None)
    if fps is None:
        raise ValueError(f"{name} needs fingerprints but none were passed")
    if chunk is None or len(X) <= chunk:
        return spec.proba(model, X, fps)
    out = np.empty((len(X), N_CLASSES))
    for start in range(0, len(X), chunk):
        end = min(start + chunk, len(X))
        out[start:end] = spec.proba(model, X[start:end], fps[start:end])
    return out


def _stratified_bootstrap_idx(y: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Resample within each class, so a draw cannot lose a class entirely."""
    by_class = {c: np.flatnonzero(y == c) for c in np.unique(y)}
    return np.concatenate(
        [rng.choice(v, size=len(v), replace=True) for v in by_class.values()]
    )


def bootstrap_fit(
    name: str,
    X,
    y,
    params: Optional[Dict[str, Any]] = None,
    *,
    n_bootstrap: int = 20,
    fps=None,
    sample_weight=None,
    seed: int = 42,
    balanced: bool = True,
    progress: bool = True,
) -> List[Any]:
    """Stratified bootstrap ensemble, for a spread on the predictions."""
    y = np.asarray(y, dtype=int)
    rng = np.random.RandomState(seed)
    models: List[Any] = []
    it = range(n_bootstrap)
    if progress:
        it = tqdm(it, desc=f"Bootstrap {name}", leave=False)
    for b in it:
        idx = _stratified_bootstrap_idx(y, rng)
        f = None if fps is None else [fps[j] for j in idx]
        sw = None if sample_weight is None else np.asarray(sample_weight)[idx]
        try:
            models.append(
                fit_model(
                    name, X[idx], y[idx], params, fps=f, sample_weight=sw,
                    seed=seed + b, balanced=balanced,
                )
            )
        except Exception:
            continue  # a degenerate resample is not worth failing the ensemble over
    return models
