# Lineage: how F3–F10 map onto this package

The notebooks in `../notebooks/` are the original work, in order. Each one is a full
pipeline rather than a step, because each was a response to what the previous one showed.
This file says what changed at each stage and where that code now lives.

Outputs are stripped from the committed notebooks; the numbers are in
[RESULTS.md](RESULTS.md).

| notebook | what it was | what it established |
|---|---|---|
| **F3** | BindingDB → multi-task regression with ChEMBL pretraining | the starting point: transfer learning + multi-task on pIC50/pKi |
| **F4** | + scaffold split, transfer ablation, progress bars | pretraining gave +0.062 R² on random splits and **nothing** on grouped ones |
| **F5** | + applicability domain, blend weighting | fitted blend weights lost to equal weighting 6/6; the AD's Tanimoto flag was being outvoted |
| **F6** | regression → **classification**, single-task pIC50 | the AD Tanimoto **veto** (21,320 → 36 in domain); random 0.89 → pooled 0.69 |
| **F7** | back to **multi-task**, 2 classes | recovered 229 pKi-only compounds (+53%); found and removed 141 Cheng–Prusoff values |
| **F8** | **7-model zoo** + floors + library health report | the floor comparison; corrected the catalogue misdiagnosis |
| **F9** | **single-task pKi** + reaction enumeration | pKi is where the signal is; the click SMARTS failure; the positive-control false negative |
| **F10** | stripped to **2 models**, full 290K streamed screen | the threshold cap; streamed scoring; fingerprints-only |

## Where each piece went

| notebook part | module |
|---|---|
| BindingDB resolve / download / DuckDB query | `data/bindingdb.py` |
| derived-value detector, censored recovery, provenance | `data/labels.py` |
| standardisation, Morgan fingerprints, descriptor selector | `features/molecules.py` |
| the model zoo (7 classifiers, Tanimoto kernel, KPLS-DA) | `models/registry.py` |
| AutoQSAR gap-aware grid search | `models/tuning.py` |
| multi-task MLP, pretraining, fine-tuning, MC dropout | `models/neural.py` |
| metrics, enrichment factor, precision@N, ECE | `evaluation/metrics.py` |
| Butina clustering, LOCO fold construction | `evaluation/splits.py` |
| fold running, pooled / macro / random-split reference | `evaluation/loco.py` |
| cross-fitted calibration and its caveats | `evaluation/calibration.py` |
| threshold sweep and selection | `evaluation/thresholds.py` |
| floor comparisons, blend-vs-member | `evaluation/floors.py` |
| four-metric applicability domain | `domain/applicability.py` |
| library loading, health report | `screen/library.py` |
| chunked streaming scorer | `screen/stream.py` |
| reaction enumeration, self-tests, property window | `screen/enumerate.py` |
| shortlist, positive control, model agreement | `screen/runner.py` |

## What was dropped

- **Optuna and scikit-optimize hyperparameter search.** The gap-aware grid search over a
  small grid was doing the work; the Bayesian search added two dependencies and a long
  runtime for configurations that the gap penalty then rejected anyway.
- **The equal-weight blend as a predictor.** It never beat its best member. It survives
  only as a consensus *call*.
- **MACCS keys and the broader descriptor pool as model inputs.** `fp_only` won the
  bake-off.
- **The Colab-specific plumbing** (Drive mounting, the leading-`/` path convention that
  caused the silent demo-file substitution).

## What was fixed in the port

Beyond the bugs listed in RESULTS.md §12, writing the test suite surfaced four more:

- the derived-value detector reported spurious blocks on independently-measured data;
  it now requires enrichment over chance and defaults to a minimum block size of 10
- logistic regression and KPLS-DA could not fit single-class folds *at all*, while the
  tree models could — so a fold loop that caught exceptions per model would have left
  uneven out-of-fold coverage across models without saying so. Now uniform, via
  `ConstantClassifier`
- a threshold that selected *nothing* satisfied any called-fraction cap, so the
  "no threshold is stringent enough" branch could never be reached
- a precision target met by three compounds in the distribution's tail was reported as
  reachable; `min_called` now guards it
