# Notebooks

The original work, in order, with **outputs stripped** (27.4 MB → 2.2 MB). The numbers they
produced are recorded in [../docs/RESULTS.md](../docs/RESULTS.md); what changed at each
stage and where that code now lives is in [../docs/LINEAGE.md](../docs/LINEAGE.md).

These are kept as a record, not as the interface. Use the package for new work: the
notebooks carry bugs that were fixed in the port (see LINEAGE.md), and each is a whole
pipeline rather than a composable piece.

| notebook | in one line |
|---|---|
| `F3_transfer_multitask_regression.ipynb` | multi-task regression with ChEMBL pretraining |
| `F4_scaffold_split_and_transfer_ablation.ipynb` | the first grouped split, and the first ablation |
| `F5_applicability_domain_and_blending.ipynb` | four-metric domain; blend weighting |
| `F6_single_task_classification.ipynb` | regression → classification; the Tanimoto veto |
| `F7_multitask_classification.ipynb` | multi-task again; Cheng–Prusoff removal |
| `F8_model_zoo_and_floors.ipynb` | seven models, floor comparisons, library health |
| `F9_pki_and_reaction_enumeration.ipynb` | single-task pKi; reaction-based enumeration |
| `F10_enamine_290k_screen.ipynb` | two models, streamed over the full catalogue |

`threshold_diagnostic_cell.py` is the standalone label-distribution diagnostic used to
place the activity cut (shape, censored audit, class counts, boundary noise); its logic
now lives in `qsar_tl/data/labels.py` and the configs' `active_thresholds`.
