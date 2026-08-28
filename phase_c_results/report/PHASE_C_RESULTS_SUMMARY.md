# VISTA Phase C Results Summary

**Status:** PASS

## Data and rule audit

- Samples: 9,414
- Unique canonical first linkers: 5,727
- Master metadata and compact `Data.csv` agree exactly.
- All first-linker SMILES parse with RDKit.
- All ten deterministic label rules regenerate the frozen labels with zero mismatch.

## CPU baseline test results

| Split | Model | Macro AUPRC | Macro AUROC | Macro F1 | Micro F1 |
|---|---|---:|---:|---:|---:|
| current | historical_vista_reference | 0.9879 | 0.9972 | 0.9604 | 0.9809 |
| current | morgan_logreg | 0.9169 | 0.9806 | 0.8740 | 0.9545 |
| current | morgan_rf | 0.9100 | 0.9836 | 0.8695 | 0.9513 |
| current | rdkit_descriptor_lro | 0.9407 | 0.9821 | 0.9185 | 0.9569 |
| strict | morgan_logreg | 0.9258 | 0.9786 | 0.8541 | 0.9478 |
| strict | morgan_rf | 0.8758 | 0.9732 | 0.8153 | 0.9359 |
| strict | rdkit_descriptor_lro | 0.9205 | 0.9758 | 0.9157 | 0.9561 |

The RDKit rule-access reconstruction is an upper-bound/label-definition audit and is **not** treated as an independent ML baseline.

## Split sensitivity

- `morgan_logreg`: strict minus current macro-AUPRC = +0.0089.
- `morgan_rf`: strict minus current macro-AUPRC = -0.0342.
- `rdkit_descriptor_lro`: strict minus current macro-AUPRC = -0.0202.

## Nearest-training ECFP4 similarity

- **current**: canonical-level median max Tanimoto 0.700; sample-weighted median 0.840; exact canonical matches 630; fraction similarity=1 46.317%.
- **strict**: canonical-level median max Tanimoto 0.631; sample-weighted median 0.702; exact canonical matches 0; fraction similarity=1 2.833%.

## Historical VISTA canonical-linker cluster bootstrap

- macro_auprc: 0.9879 (95% cluster-bootstrap CI 0.9824–0.9923).
- macro_auroc: 0.9972 (95% cluster-bootstrap CI 0.9963–0.9980).
- macro_f1: 0.9604 (95% cluster-bootstrap CI 0.9494–0.9692).
- micro_f1: 0.9809 (95% cluster-bootstrap CI 0.9779–0.9839).
- subset_accuracy: 0.8732 (95% cluster-bootstrap CI 0.8545–0.8924).
- hamming_loss: 0.0152 (95% cluster-bootstrap CI 0.0128–0.0176).

## Scientific interpretation boundary

Phase C measures how conventional 2D molecular representations behave under the historical and canonical-disjoint splits. It does not replace the pending strict VISTA GPU reruns. Any superiority claim between VISTA and the new CPU baselines should wait until the strict VISTA seeds are returned from EPFL.
