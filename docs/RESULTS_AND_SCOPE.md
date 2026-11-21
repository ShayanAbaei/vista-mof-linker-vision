# Verified results and evaluation scope

This document distinguishes the completed historical GPU vision experiment from the separately evaluated molecular baselines.

## Historical VISTA test metrics — sample-level split only

| Metric | Recorded value |
|:--|--:|
| Macro-AUPRC | **0.987924** |
| Macro-AUROC | **0.997189** |
| Macro-F1 | **0.960403** |
| Micro-F1 | **0.980930** |
| Exact-match accuracy | **0.873229** |

Sources: `reference_current_results/metrics/test_metrics_overall.csv`, `reference_current_results/metrics/thresholds.csv`, `reference_current_results/predictions/test_predictions.csv`.

A 2,000-resample canonical-linker cluster bootstrap on the **same historical** test set yielded 95% percentile intervals of **0.982404–0.992261** for macro-AUPRC and **0.949438–0.969158** for macro-F1. Bootstrap intervals describe uncertainty conditional on this test set; they do not remove any train/test linker identity overlap.

## Molecular baseline test metrics

| Evaluation split | Baseline | Macro-AUPRC | Macro-F1 |
|:--|:--|--:|--:|
| Historical sample-level | Morgan logistic regression | 0.916885 | 0.873984 |
| Historical sample-level | Morgan random forest | 0.909990 | 0.869493 |
| Historical sample-level | RDKit leave-exact-rule-out LR | 0.940695 | 0.918494 |
| Canonical-first-linker disjoint | Morgan logistic regression | 0.925766 | 0.854116 |
| Canonical-first-linker disjoint | Morgan random forest | 0.875810 | 0.815339 |
| Canonical-first-linker disjoint | RDKit leave-exact-rule-out LR | 0.920499 | 0.915722 |

Source: `phase_c_results/metrics/baseline_metrics_overall.csv` and raw per-row data under `phase_c_results/source_data/`. The descriptor baseline excludes the exact rule-defining descriptor where specified; correlated descriptors can remain.

## Data overlap and similarity audit

| Audit | Original sample split | Canonical-disjoint split |
|:--|--:|--:|
| Total sample records | 9,414 | 9,414 |
| Train / validation / test | 6,590 / 1,412 / 1,412 | 6,590 / 1,412 / 1,412 |
| Canonical first-linker groups crossing partitions | 598 | 0 |
| Test images with train-exact canonical match | 630 | 0 |
| Median nearest-train ECFP4 Tanimoto (record-weighted) | 0.840 | 0.702 |

There are **330 distinct canonical linker identities** within the strict test set, so record-weighted and linker-weighted statistics can differ. See `data/frozen/` and `phase_c_results/metrics/tanimoto_summary.csv`.

## Interpretation limits

- Original VISTA scores **are real archived GPU-trained image-model results**, but they measure the *historical sample-level evaluation*, not unseen-canonical-linker generalization.
- Strict-split results presented here are from **molecular baselines**. No strict-split VISTA, shared-Swin or GIN experiment is claimed to have run.
- Targets are deterministic RDKit-derived **proxy annotations**, not experimental whole-MOF functionalities.
- A single-seed, historical architecture comparison is not evidence for multi-seed statistically significant model superiority.
- Saved attention examples are descriptive, not proof that highlighted pixels or atoms causally determine predictions.
- Historical trained weights were not located in the audited source archive; the repository contains recorded predictions and code, not a downloadable historical model checkpoint.
