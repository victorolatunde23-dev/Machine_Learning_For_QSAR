# Results

Findings from the PSMA inhibitor project this package came out of, with the numbers. Each
default in the package traces back to something here.

Target: PSMA / FOLH1 / glutamate carboxypeptidase II (UniProt Q04609). Data: BindingDB.
Library screened: a ~290,000-compound Enamine diversity catalogue.

Numbers come from the notebook lineage in `notebooks/` (F3 → F10). Where a figure is a
single run on a few hundred compounds it is quoted as such — the fold-to-fold spread on
this dataset is larger than most of the differences between models, and that is itself one
of the findings.

---

## 1. Transfer learning bought nothing

Two ablations, in opposite directions, both empty.

| ablation | setup | result |
|---|---|---|
| F4, regression | shared trunk pretrained on a broad potency corpus, then fine-tuned | **+0.062 R²** on random splits, **nothing** on grouped splits |
| F6, classification | same architecture, with and without a pretrained trunk, paired on the same folds | **−0.0066 AUC** |

The F4 result is the more interesting one: the apparent gain existed *only* under the split
that leaks analogues. Under a grouped split it vanished. A single ablation showing nothing
is weak evidence; two in opposite directions on the same data is considerably stronger.

The network itself was also the joint-worst model on pKi (0.688 pooled ROC-AUC, tied with
nearest-neighbour lookup) and it actively diluted the blend it was in (0.763 with it,
0.796 without).

**Consequence:** `transfer.enabled: false` is the default, and the whole torch dependency
is optional. The code is retained — `configs/psma_transfer_ablation.yaml` reproduces the
comparison — because the finding is about *this dataset*, not about transfer learning. On a
target where the pretraining corpus is genuinely related, it may well help; the ablation is
how you would find out.

---

## 2. Analogue leakage is large, and model-dependent

Random split (10 repeats) against pooled leave-one-cluster-out, pKi, n = 278:

| model | random split | pooled grouped | gap |
|---|---|---|---|
| LightGBM | 0.865 | **0.804** | −0.061 |
| Blend (5) | 0.875 | 0.783 | −0.092 |
| RandomForest | 0.868 | 0.777 | −0.091 |
| KPLS-DA | 0.857 | 0.775 | −0.082 |
| SVM (Tanimoto) | 0.857 | 0.763 | −0.094 |
| LogReg (L2) | 0.867 | 0.763 | −0.104 |
| kNN (Tanimoto) | 0.841 | 0.712 | −0.129 |
| BernoulliNB | 0.819 | 0.686 | −0.133 |

The gap is largest for nearest-neighbour lookup, which can only work by recognising
neighbours. That is the mechanism made visible, and it is why the grouped split is the
default rather than an option.

Earlier in the lineage the effect was starker still: F6's single-task pIC50 model went from
0.89 on a random split to 0.69 pooled.

---

## 3. The floor comparison, and why it needs two metrics

Pooled pKi, whole families held out, n = 278, base rate 0.507:

| model | ROC-AUC | PR-AUC | precision@50 | MCC | ECE (raw) |
|---|---|---|---|---|---|
| **LightGBM** | **0.804** | 0.762 | 0.80 | 0.454 | 0.114 |
| Blend (5) | 0.783 | 0.744 | 0.74 | 0.437 | 0.064 |
| RandomForest | 0.777 | 0.716 | 0.74 | **0.478** | **0.043** |
| KPLS-DA | 0.775 | **0.766** | **0.82** | 0.375 | 0.193 |
| SVM (Tanimoto) | 0.763 | 0.730 | 0.76 | 0.441 | 0.059 |
| LogReg (L2) | 0.763 | 0.735 | 0.80 | 0.417 | 0.052 |
| kNN (Tanimoto) | 0.712 | 0.683 | 0.76 | 0.301 | 0.168 |
| BernoulliNB | 0.686 | 0.654 | 0.72 | 0.187 | 0.389 |

**Against nearest-neighbour lookup:** LightGBM clears it by **+0.092 AUC**. There is real
structure–activity signal being learned here, not just neighbour recall.

**Against the linear floor:** LightGBM clears LogReg by only +0.041 AUC, and on
**precision@50 nothing clears it** (both 0.80). A shortlist only uses the top of the
ranking, and there a linear model on fingerprint bits is as good as the best ensemble.

Reporting AUC alone would have hidden the second half; reporting precision alone would have
hidden the first. Hence both, with different margins — precision at a fixed N moves in
steps of 1/N, so a one-compound difference is not a result.

**On pIC50 the picture was worse:** *nothing* cleared the nearest-neighbour floor at all.
That asymmetry is why the final pipeline models pKi rather than the larger pIC50 set.

---

## 4. Blending never paid

Seven consecutive endpoints across F4–F10 where the equal-weight blend failed to beat its
best single member. On pKi: 0.783 against LightGBM's 0.804.

Fitted simplex weights were worse again — they lost to a plain average in all six
scheme × endpoint comparisons tried, by up to 0.08 R² under grouped splits, and they were
fitted on the very test set they were scored against.

**Consequence:** no blending. Models are reported separately; where a single list is
needed, they are combined as a consensus *call* (both models agree) and ordered by mean
*percentile* rank — scale-free, so averaging it is legitimate where averaging two
differently-calibrated probability columns is not.

---

## 5. The data cleaning mattered more than the modelling

**Cheng–Prusoff-derived values: 141 of 169 dual-measured compounds (83%).** Three
fixed-offset blocks:

| offset (log units) | implied ratio | n | dominant source | source share |
|---|---|---|---|---|
| 2.1464 | 140.1 | 106 | US20250213705 | 1.000 |
| 0.6997 | 5.01 | 30 | US10736974 | 0.633 |
| 2.1151 | 130.4 | 5 | US20250213705 | 1.000 |

A constant difference in p-units across many compounds from one filing means the second
endpoint was computed from the first, not measured. Keeping them puts most of the pKi
column on a different scale from the rest, so no single activity threshold is right for
most of it.

The detector is a heuristic and has a false-positive rate: ordinary assay scatter will put
five or six compounds in one narrow window by chance. The implementation now requires a
block to be enriched over what a Gaussian with the data's own spread predicts, and the
default minimum block size is 10. Confirm a block by looking at the report — a genuine
conversion comes overwhelmingly from one filing, has a tiny `offset_sd`, and its implied
ratio is a plausible `1 + [S]/Km`.

**Censored recovery: 25 labels** that would otherwise have been discarded (22 negatives
from `>` bounds at or below the cut, 3 positives from `<` bounds above it). On a target
whose literature reports mostly actives, those are a meaningful share of the negatives.

**Deduplication: 31%** of the screening catalogue turned out to be protonation and
tautomer variants of structures already in it.

**Single source concentration:** one patent filing supplied 119 of 706 compounds (17%),
which is why provenance is tagged and a dominant filing is held out as its own fold.

---

## 6. Multi-task recovered a third of the data, and it did not help

Single-task pKi uses 282 labelled compounds. Multi-task across pIC50 and pKi reaches 706
carried / 483 + 282 labelled — recovering 229 pKi-only compounds, a third of the data, via
a masked loss that lets each compound train the head it has a label for.

Pooled: pIC50 0.770, pKi 0.796. But on pIC50 nothing cleared the nearest-neighbour floor,
so the extra endpoint added data without adding signal — and the classical models are
fitted per endpoint anyway, so only the neural network ever shared anything, and it was the
weakest model on pKi.

**Consequence:** single-task pKi is the default. `configs/psma_multitask.yaml` keeps the
multi-task path, which is the right starting point on a target where both endpoints carry
signal.

---

## 7. Fingerprints beat fingerprints-plus-descriptors

Feature-set bake-off on pKi:

| feature set | CV ROC-AUC |
|---|---|
| `fp_only` | **0.865** |
| `fp_plus_descriptors` | 0.850 |
| `descriptors_only` | 0.844 |

The 70 curated RDKit descriptors (reduced from 217 by near-constant and
correlation-cluster filtering) were costing accuracy. They are also by far the most
expensive thing to compute — several, `BCUT2D` among them, need an eigendecomposition per
molecule — which over a few hundred thousand compounds dominates the runtime.

**Consequence:** `feature_set: fp_only`. The descriptor machinery is kept because the
applicability domain needs it and because the bake-off should be re-runnable.

---

## 8. Calibration

Platt beat isotonic at this sample size: ECE improved in 10 of 10 model × endpoint pairs
against isotonic's 7 of 10, and isotonic's tie-collapsing moved pooled AUC by 0.10 in one
run.

Recalibrating an already well-calibrated model made things worse — one endpoint arrived at
raw ECE 0.072–0.084 (its base rate is near 0.5 and balanced class weighting naturally
targets that) and recalibrating anyway took the blend from 0.084 to 0.167. Hence
`recalibrate_min_ece`.

Final state on pKi: RandomForest, LogReg, SVM and the blend needed nothing
(ECE 0.043–0.064); LightGBM 0.114 → 0.062, KPLS-DA 0.193 → 0.089, kNN 0.168 → 0.113,
BernoulliNB 0.389 → 0.114.

**One caveat that caused a wrong diagnosis once.** A pooled out-of-fold column is assembled
from one calibration map *per fold*. Each map preserves order within its own fold, but
pooling can reorder compounds that sat in different folds, so the pooled calibrated AUC can
shift without any map being inverted. An earlier version of this pipeline attributed a
0.0925 shift to a negative calibrator slope and then correctly reported that no slope was
negative — the explanation was simply wrong. Quote the raw AUC for discrimination and the
calibrated numbers for probability quality. A library scored with a single deployment
calibrator does not have this problem.

---

## 9. The applicability domain excludes almost everything, and that is the answer

The Tanimoto veto, not Tanimoto as one vote in four: using it as one vote let compounds
pass while failing the structural test, and the other three metrics compare bulk
properties, so a molecule of roughly the right size and lipophilicity passes them while
being structurally unrelated to anything in training. In F5 every one of the 15 top hits
was flagged `tanimoto_OOD` (dissimilarity 0.73–0.83 against a 0.457 threshold) and every
one was reported as in-domain.

With the veto on, F7 put **36 of ~200,000** catalogue compounds in domain. F9, on a
deliberately near-sampled subset, put 239 of 17,399 (1.37%).

This was initially misdiagnosed. The catalogue's first entries are solvents and reagents —
butyl chloride, β-propiolactone, diethylene glycol — and it was called a building-block
collection four times on that basis. A random 20,000-compound sample settled it: median MW
337, 24 heavy atoms, 1.0% fragment-like. **It is a perfectly normal drug-like deck,
ordered small-to-large.** The 100% out-of-domain rate is a genuine chemistry mismatch —
PSMA ligands are polar Glu–urea peptidomimetics, diversity decks are lead-like and less
acidic — not a bad file and not a loose threshold.

**Consequence:** the library health report samples randomly rather than taking the head,
and says so. And when nothing is in domain, the honest options are to accept that every
prediction is an extrapolation and use the ranking only, or to screen a different library.

---

## 10. Enumeration is circular by construction

Reaction-based enumeration around the Glu–urea pharmacophore: 7 validated reactions × 6
cores × 62 caps → 279 products after deduplication.

Scored against catalogue strata at matched molecular weight: enumerated median 0.765
against property-matched decoys at 0.144 (MW 445 vs 430). Real separation, not a size
artefact — **but partly circular**, because the model is recognising the pharmacophore it
was trained on. 83.9% of enumerated products were in domain against ~0% of the catalogue,
which is in-domain by construction.

Zero scaffold hops: no catalogue compound reached the enumerated top decile while staying
in domain.

Three things the enumeration run exposed:

- **The click-chemistry template failed its self-test**: "produced nothing". A mapped
  product atom inherits its reactant's formal charge unless the template says otherwise, so
  the azide's +1/−1 were carrying into the triazole, breaking the ring valences and failing
  sanitisation. The self-test is the only reason this was caught; a wrong SMARTS does not
  raise.
- **The positive control gave a false negative.** It reported "0 enumerated products match
  a known training compound" — and told the user the shortlist was unfalsifiable — while
  two products showed Tanimoto distance **exactly 0.000** and descriptor distance ~2e-13 to
  a training structure. They were 4-fluorobenzoyl- and 4-iodobenzoyl-KuE, both well-known
  PSMA ligands. The full InChIKey differed in a layer the achiral fingerprint ignores
  (stereochemistry, tautomer or salt form). The control is now two-tier: full key, then the
  14-character connectivity block.
- **The enumeration explores the wrong axis.** The property window from the known actives
  came out at MW 322–1464, median 847 — the real ligands carry a linker and a chelator
  (DOTA, HBED) on top of the binding motif. The enumerated products are MW ~445, at the
  bottom edge. The library is building *truncated* analogues of a class defined partly by
  what was left off.

---

## 11. The threshold needs a cap

A classifier emits a probability; "active" needs a cut. Choosing it on the training fit
puts it wherever the model overfits to — ~0.99 train AUC here. So it is chosen on the
out-of-fold predictions.

But MCC and F1 will both maximise at a very permissive cut when the model is weak. One run
chose **p ≥ 0.11, calling 86% of compounds active at recall 1.000** — over a
290,000-compound library, a 250,000-compound "shortlist". The point of a screen is to
decide what *not* to look at, so a cut that narrows nothing is not a candidate. Hence
`threshold_max_called_frac` (default 0.25), with the rejected unconstrained optimum
reported so the trade is visible.

Expected yield at the chosen cut, measured out-of-fold on held-out families: roughly **80%
precision in the top 50**. On a library far outside the training distribution that is an
upper bound, not a forecast.

---

## 12. Things that were wrong along the way

Recorded because they shaped the design, and because a results file that only lists
successes is not a useful record.

- **The catalogue was called a reagent collection four times**, on the evidence of a
  `.head()`. A random sample disproved it. The health report now samples randomly.
- **A config field was read but never defined.** `cfg.auto_resolve_bindingdb_url` did not
  exist on the config object; the BindingDB download failed, a synthetic fallback took
  over, and the failure surfaced six stages later as
  `n_splits=10 greater than the number of samples: n_samples=3`. The build-time check that
  should have caught it only looked for `CONFIG.x`, not `cfg.x`. Now: unknown keys are
  rejected, cross-field consistency is validated at load time, the config is checked
  against an actually-instantiated dataclass, and a sanity gate after labelling names the
  likely cause.
- **A library loader silently substituted a demo file.** It rewrote absolute paths
  unconditionally, missed the file, and wrote and screened 8 molecules while printing
  `Loading: …`. Now a missing library raises.
- **A property filter emptied a library** and every downstream number became vacuous. Now
  there is a floor below which it flags instead of filtering.
- **A validated reaction contributed nothing** because no building block carried the role
  it needed, and the "0 produced nothing" counter only covered combinations that were
  *tried*. Now unused reactions are reported.
- **A duplicated config field.** `loco_min_cluster_size` was defined twice, 25 then 12;
  Python took the later value and the config misreported what ran.
- **A calibration diagnostic blamed the wrong cause** (see §8).
- **A distance-vs-difficulty correlation was quoted from an underpowered test** — r =
  −0.056 with four of eight folds unscoreable. That was a flaw in the fold design, not a
  finding, and it is why folds must now be *scorable* (enough of both classes) to get a
  reported row.
- **An argument against multi-task focused on the 169 overlapping compounds** and missed
  that single-task pKi discards 229 pKi-only compounds.

---

## What to read first on a new target

1. The derived-value report. If a large block appears, understand it before anything else.
2. The leakage table: random split against pooled grouped. If the gap is small, the dataset
   is less congeneric than this one and more of the literature's usual numbers apply.
3. The floor comparison, on both metrics. If nothing clears nearest-neighbour lookup,
   stop — there is no structure–activity signal to model, only similarity.
4. The applicability domain against the library you actually intend to screen. If nothing
   is in domain, no modelling choice fixes it.
