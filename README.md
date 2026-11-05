# Machine Learning for QSAR

Machine-learning work on quantitative structure–activity relationships.

## Sub-projects

### [`transfer_learning_qsar/`](transfer_learning_qsar/) — `qsar-tl`

A configurable pipeline for binding-affinity classification, grouped-split evaluation and
virtual library screening, built from a PSMA (FOLH1 / GCPII) inhibitor project. Installable
package, five ready configs, 201 tests, and a
[results record](transfer_learning_qsar/docs/RESULTS.md) documenting what the project
actually established — including that transfer learning from a broad potency corpus bought
nothing on this dataset, across two ablations in opposite directions.

```bash
cd transfer_learning_qsar
pip install -e '.[chem]'
qsar-tl info
qsar-tl validate configs/psma_pki_screen.yaml
```

Each sub-project is self-contained: its own `pyproject.toml`, tests and docs. Any one can
be split out into a repository of its own with history intact via `git subtree split`.
