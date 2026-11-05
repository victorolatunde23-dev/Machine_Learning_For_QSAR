# qsar-tl — transfer-learning QSAR pipeline

A configurable pipeline for binding-affinity classification, grouped-split evaluation and
virtual library screening. It grew out of a PSMA (FOLH1 / GCPII) inhibitor project and
ships configured for that target, but nothing in the machinery is target-specific.

The headline result from the project that produced it is worth stating up front, because
it is why several defaults look the way they do: **on this dataset, transfer learning from
a broad potency corpus bought nothing.** Two ablations in opposite directions both came
back empty. The code is kept so the ablation can be re-run on another target; it is off by
default. See [docs/RESULTS.md](docs/RESULTS.md).

## Install

```bash
pip install -e .                  # core: numpy, pandas, scipy, scikit-learn, pyyaml
pip install -e '.[chem]'          # + rdkit: molecules, fingerprints, enumeration
pip install -e '.[all,dev]'       # everything, plus pytest
```

The core install deliberately needs no chemistry toolkit. Data cleaning, split
construction, metrics, calibration and threshold selection are pure numpy/pandas/sklearn,
so they are importable and testable anywhere; anything that needs a molecule fails with a
message naming the extra to install. `qsar-tl info` reports what is available.

## Use it

```bash
qsar-tl info                                  # what is installed, which models can run
qsar-tl validate configs/psma_pki_screen.yaml # costs nothing, catches typos and conflicts
qsar-tl train    configs/psma_pki_screen.yaml
qsar-tl screen   configs/psma_pki_screen.yaml --library Enamine.sdf
```

Or from Python, one stage at a time — which is how it was developed, because each stage's
output is worth looking at before paying for the next:

```python
from qsar_tl import Config, Pipeline

pipe = Pipeline(Config.from_yaml("configs/psma_pki_screen.yaml"))
pipe.load_labels()        # derived-value removal, censored recovery, provenance tags
pipe.featurise()          # standardise -> Morgan fingerprints
pipe.tune()               # gap-aware grid search per model
pipe.build_folds()        # Butina clustering -> leave-one-cluster-out folds
pipe.evaluate()           # the honest numbers, calibration, decision threshold
pipe.fit_final()          # refit on everything, for deployment
pipe.save()
```

`pipe.evaluate()` is the one that tells you whether screening is worth doing at all.

## Configs

| file | what it is |
|---|---|
| `psma_pki_screen.yaml` | the validated default: single-task pKi, two models, library screening |
| `psma_multitask.yaml` | both endpoints jointly, one head per endpoint, seven models |
| `psma_transfer_ablation.yaml` | the transfer-learning arm, as a reproducible ablation |
| `psma_enumeration.yaml` | reaction-based enumeration with catalogue controls |
| `quickstart_csv.yaml` | your own measurements from a CSV, no network needed |

Unknown keys are **rejected, not ignored**, and cross-field combinations are checked at
load time. That is not pedantry: a real run was lost to a config field that was read but
never defined, and the failure surfaced six stages later as
`n_splits=10 greater than the number of samples: n_samples=3`.

## What the package is careful about

Each of these is a default that encodes a finding. The reasoning lives next to the code.

**Grouped evaluation, not random splits.** A random split of a congeneric dataset leaves
close analogues on both sides, so the model is asked about molecules it has effectively
seen. Here that inflated ROC-AUC by 0.06–0.13 depending on the model, and the inflation
was *largest* for nearest-neighbour lookup — the mechanism made visible. Butina
leave-one-cluster-out holds out a whole chemical family at a time.
→ `qsar_tl/evaluation/splits.py`

**Derived values are removed.** A block of compounds whose two endpoints differ by a
*fixed* amount was not measured twice; the second value was computed from the first by
Cheng–Prusoff. That was 141 of 169 dual-measured compounds, and keeping them puts most of
the pKi column on a different scale from the rest, which makes any single activity
threshold wrong for most of it. The test keys off a separate `derived_endpoint` field
rather than off whichever endpoint is primary — those coincided once, and promoting pKi to
primary silently kept every converted value. → `qsar_tl/data/labels.py`

**Censored measurements are recovered, not discarded.** A `>` bound at or below the active
cut proves "not active"; a `<` bound at or above it proves "active". On a target whose
literature reports mostly actives, that is where a large share of the negatives come from.

**Models are reported against floors.** Nearest-neighbour lookup and L2 logistic
regression, on **both** ROC-AUC and precision@N, because they disagree and the
disagreement is the finding: the best model here cleared lookup by +0.092 AUC — real
signal, not neighbour recall — but *nothing* cleared the linear floor on precision@50, and
a shortlist only uses the top of the ranking. → `qsar_tl/evaluation/floors.py`

**No blending.** Equal-weight blending failed to beat its best single member on seven
consecutive endpoints. (Fitted simplex weights were worse still, and were fitted on the
data they were scored against.) Models are reported separately and combined only as a
consensus *call*. → `qsar_tl/evaluation/metrics.py`

**Thresholds come from out-of-fold predictions, with a cap on the called fraction.**
Choosing on the training fit puts the cut wherever the model overfits to. And MCC or F1
will happily maximise at a very permissive cut on a weak model: one run chose p ≥ 0.11,
calling 86% of compounds active at recall 1.000 — over a 290,000-compound library, a
250,000-compound "shortlist". → `qsar_tl/evaluation/thresholds.py`

**Platt, not isotonic, and only where needed.** Isotonic collapses ties at a few hundred
compounds. And recalibrating an already-calibrated model made one blend *worse*
(ECE 0.084 → 0.167), so pairs below `recalibrate_min_ece` are left alone. A pooled
out-of-fold column is assembled from one map per fold, so pooling can reorder compounds
that sat in different folds — quote the **raw** AUC for discrimination and the calibrated
numbers for probability quality. → `qsar_tl/evaluation/calibration.py`

**Tanimoto is a veto in the applicability domain, not one vote in four.** The other three
metrics compare bulk properties, so a molecule of roughly the right size and lipophilicity
passes them while being structurally unrelated to anything in training. In one run every
top hit was flagged `tanimoto_OOD` and every one was still reported as in-domain.
→ `qsar_tl/domain/applicability.py`

**Reaction templates self-test, and a failure disables them.** A wrong reaction SMARTS
does not raise — it silently matches nothing or builds the wrong regiochemistry. Every
template ships a `(reactants → expected product)` test that runs at load time. That is how
the click-chemistry template here was caught producing nothing at all (the azide's formal
charges were carrying into the product and breaking the ring valences).
→ `qsar_tl/screen/enumerate.py`

**A missing library raises; it does not substitute.** An earlier loader rewrote absolute
paths, missed the file, and silently wrote and screened an 8-compound demo set while
printing `Loading: …` as though nothing were wrong. → `qsar_tl/screen/library.py`

**Large libraries stream.** A 290,000 × 2048 float32 matrix is 2.4 GB before anything
else, so scoring runs in chunks and appends to disk. → `qsar_tl/screen/stream.py`

## Layout

```
src/qsar_tl/
  config.py            nested dataclasses, YAML overlay, strict validation
  pipeline.py          stage-by-stage orchestration + the sanity gate
  _optional.py         lazy optional imports with actionable errors
  cli.py               info / validate / train / screen / enumerate
  data/                BindingDB or CSV acquisition; labels and derived-value detection
  features/            standardisation, Morgan fingerprints, descriptors, selection
  models/              registry (7 classifiers), gap-aware tuning, the neural network
  evaluation/          metrics, Butina/LOCO splits, calibration, thresholds, floors
  domain/              four-metric applicability domain
  screen/              library loading, streamed scoring, reaction enumeration
configs/               one config per lineage stage
notebooks/             the original F3–F10 notebooks, outputs stripped
tests/                 201 tests, no chemistry toolkit required
docs/RESULTS.md        every finding, with the numbers
docs/LINEAGE.md        how F3–F10 map onto this package
```

## Tests

```bash
pytest                     # 201 pass, 2 skip without rdkit
pytest -m chem             # the rdkit-dependent tests
```

The suite runs without RDKit, LightGBM or torch. The `chem`-marked tests are the ones that
check the reaction SMARTS, which cannot be verified any other way — **run them in an
environment with RDKit before trusting the enumeration.** They were written but could not
be executed where this package was assembled, so treat them as unverified until they have
run once.

Several tests exist because writing them found real bugs: the derived-value detector
reported spurious blocks on independently-measured data (now guarded by an enrichment
test against chance), logistic regression and KPLS-DA could not fit single-class folds at
all while the tree models could (now uniform, via `ConstantClassifier`), a threshold that
selected *nothing* satisfied any called-fraction cap, and a precision target met by three
tail compounds was reported as reachable.

## Caveat worth repeating

On the library this was built for, the applicability domain excludes almost everything:
PSMA ligands are polar Glu–urea peptidomimetics and a lead-like diversity deck contains
very little resembling them. That is a real chemistry mismatch, not a threshold to loosen.
Screening it anyway is defensible, but then **the ranking is the deliverable and the
probability is not** — it is calibrated against held-out *training* chemistry, and nothing
here can calibrate it against chemistry the training set does not cover.

## Licence

MIT.
