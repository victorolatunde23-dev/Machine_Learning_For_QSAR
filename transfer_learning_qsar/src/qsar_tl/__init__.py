"""qsar-tl: a configurable QSAR pipeline for affinity classification and library screening.

Grew out of a PSMA (FOLH1 / GCPII) inhibitor project and is configured for that target by
default, but nothing in the machinery is target-specific: point ``data`` at another
UniProt id or a CSV and the rest follows.

Quick start
-----------
>>> from qsar_tl import Config, Pipeline
>>> cfg = Config()                      # PSMA pKi classification, fingerprints only
>>> cfg.labels.primary_endpoint
'pKi'

Then, with a chemistry toolkit installed and the data to hand::

    pipe = Pipeline(cfg).run()
    print(pipe.state.pooled)            # the honest, grouped-split metrics

What this package is careful about
----------------------------------
The defaults encode findings rather than preferences, and each is documented where it
lives:

* **Grouped evaluation is the default** -- a random split of a congeneric dataset inflated
  ROC-AUC by 0.06 to 0.13 here. See :mod:`qsar_tl.evaluation.splits`.
* **Derived values are removed** -- 141 of 169 dual-measured compounds carried a pKi
  computed from the IC50 by Cheng-Prusoff rather than measured. See
  :mod:`qsar_tl.data.labels`.
* **Models are reported against floors** -- nearest-neighbour lookup and a linear model,
  on both AUC and precision@N, because they disagree. See
  :mod:`qsar_tl.evaluation.floors`.
* **Thresholds are chosen on out-of-fold predictions with a cap on the called fraction.**
  See :mod:`qsar_tl.evaluation.thresholds`.
* **Transfer learning is off** -- two ablations found it bought nothing on this data. The
  code is kept so the ablation can be re-run elsewhere. See :mod:`qsar_tl.models.neural`
  and ``docs/RESULTS.md``.
* **Heavy dependencies are optional.** Importing this package needs only
  numpy/pandas/scipy/scikit-learn, so the data-cleaning and evaluation logic is testable
  anywhere. See :mod:`qsar_tl._optional`.
"""

from ._optional import OptionalDependencyError, have, missing
from .config import (
    Config,
    ConfigError,
    DataConfig,
    DomainConfig,
    EvalConfig,
    FeatureConfig,
    LabelConfig,
    ModelConfig,
    ScreenConfig,
    TransferConfig,
    load_config,
)
from .pipeline import InsufficientDataError, Pipeline, PipelineState

__version__ = "0.1.0"

__all__ = [
    "Config",
    "ConfigError",
    "DataConfig",
    "DomainConfig",
    "EvalConfig",
    "FeatureConfig",
    "InsufficientDataError",
    "LabelConfig",
    "ModelConfig",
    "OptionalDependencyError",
    "Pipeline",
    "PipelineState",
    "ScreenConfig",
    "TransferConfig",
    "__version__",
    "have",
    "load_config",
    "missing",
]


def capabilities() -> str:
    """One-line report of which optional dependencies are present.

    >>> "numpy" not in capabilities()   # only optional deps are listed
    True
    """
    from .models.registry import available_models, unavailable_models

    rows = [f"qsar-tl {__version__}"]
    for mod, purpose in (
        ("rdkit", "molecules, fingerprints, enumeration"),
        ("lightgbm", "LightGBM"),
        ("torch", "the neural network / transfer learning"),
        ("duckdb", "BindingDB queries"),
        ("matplotlib", "figures"),
        ("tqdm", "progress bars"),
    ):
        rows.append(f"  {'yes' if have(mod) else ' no'}  {mod:<10s} {purpose}")
    rows.append(f"  models available : {available_models()}")
    un = unavailable_models()
    if un:
        rows.append(f"  models unavailable: {un}")
    return "\n".join(rows)
