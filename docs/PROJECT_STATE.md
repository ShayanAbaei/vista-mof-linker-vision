# VISTA project state — pre-EPFL

## Stable current facts

- 9,414 linker-image records.
- 5,727 unique canonical first-linker SMILES.
- 10 deterministic RDKit-derived proxy labels.
- Current sample-level split: 6,590 / 1,412 / 1,412.
- Current reference model: label-token Swin-B.
- Current sample-level results are preserved in `reference_current_results/`.
- A canonical-first-linker-disjoint split already exists and must be the primary robustness rerun.
- New repeated-seed, GNN, embedding/UMAP, and attention-control results are not yet claimed.

## Phase A goal

Build and validate the repository/data foundation without altering the scientific training configuration.

## Do not do in Phase A

- no strict GPU training;
- no new UMAP claims;
- no GNN training;
- no final manuscript rewriting;
- no replacement of the historic benchmark by invented future values.
