# VISTA Phase C Protocol — Local CPU Scientific Strengthening

## Scope

Phase C is CPU-only. It adds conventional chemical/molecular baselines and similarity/uncertainty audits without modifying the frozen VISTA architecture.

Primary analyses:

1. **Morgan/ECFP4 baseline** — radius 2, 2048 bits, Logistic Regression.
2. **Morgan/ECFP4 Random Forest baseline** — same fingerprints, small validation-selected grid.
3. **RDKit descriptor baseline** — a leave-rule-out (LRO) construction: for each target, descriptors that directly encode that target's deterministic labeling rule are excluded before fitting.
4. **Rule-access audit/oracle** — the exact deterministic RDKit rules are regenerated and must reproduce the ten labels exactly. This is a label-definition consistency/upper-bound audit, **not an independent ML baseline**.
5. **Tanimoto nearest-training audit** — maximum ECFP4 similarity from each test linker to the training set for both historical sample split and canonical-first-linker-disjoint split.
6. **Canonical-linker cluster bootstrap** — uncertainty for the existing historical VISTA sample-split test predictions, resampling canonical linker identities rather than individual rows.

## Frozen evaluation policy

- Hyperparameters are selected **only on validation macro-AUPRC**.
- Per-label decision thresholds are selected **only on validation F1**.
- Selected hyperparameters and thresholds are frozen before evaluating the test split.
- No test result is used to choose a model, threshold, descriptor set, or fingerprint setting.
- The current and strict split use the same fingerprint definition and candidate grids.

## Morgan definition

- RDKit Morgan generator
- radius = 2 (ECFP4 convention)
- 2048 bits
- chirality disabled (`includeChirality=False`)
- bond types enabled (`useBondTypes=True`)
- no feature invariants beyond the RDKit Morgan defaults

## Descriptor LRO policy

The descriptor matrix contains a transparent mix of standard RDKit descriptors and a few explicitly named rule descriptors. For each target, descriptors that directly define that target are removed before training that target's classifier. Other descriptors are allowed, including descriptors defining *other* labels; therefore this is a leave-target-rule-out baseline, not a claim of complete chemical independence.

Direct exclusions:

- Hydrophobicity: `MolLogP`
- Polarity: `TPSA`
- Flexibility: `NumRotatableBonds`
- Aromaticity: aromatic-atom/aromatic-ring count descriptors
- H-bond donors: `NumHDonors`, `NHOHCount`
- H-bond acceptors: `NumHAcceptors`, `NOCount`
- Single-ring proxy: direct ring-count descriptors
- Ring-rich (>6 rings) proxy: direct ring-count descriptors
- Charge present: `NetFormalCharge`
- Conjugation present: direct count/indicator of bonds with order > 1.5

## Rule-access audit

The exact target rules are regenerated from the first MOFid linker:

- Crippen logP > 2.5
- TPSA > 90
- rotatable bonds > 5
- any aromatic atom
- HBD > 2
- HBA > 2
- ring count == 1
- ring count > 6
- net formal charge != 0
- any bond order > 1.5

A mismatch is a blocking Phase-C error.

## Tanimoto policy

For each test sample, the nearest training fingerprint is found with the same ECFP4 definition. The output records similarity, nearest train filename/canonical linker, and whether the canonical linker is exactly identical. The strict split should have zero exact canonical identity matches by construction.

## Bootstrap policy

The historical VISTA test predictions are merged with the frozen current-split canonical identity. Bootstrap units are unique canonical first-linker identities. All rows belonging to a sampled identity are carried together. The reported interval is the percentile 95% interval across bootstrap replicates.

## Git policy

Commit code/config/protocol separately from generated scientific results. Heavy caches and logs are ignored; compact CSV/JSON/PDF/PNG source-data/result artifacts are committed.


## Duplicate-aware evaluation

Every CPU model writes both:
- ordinary sample-weighted metrics;
- unique-canonical-linker metrics, where probabilities are averaged within each canonical linker identity before metrics are computed.

This second view avoids over-weighting duplicated QMOF records that share the same canonical first linker.

## Random Forest implementation

Morgan RF uses ten independent binary Random Forest classifiers. This avoids the potentially confusing multiplication of class weights across outputs in a single multi-output Random Forest.

## Threshold comparator

Predictions use `probability >= threshold`, matching the frozen VISTA evaluation convention.


## RDKit HBA version drift

The frozen VISTA `h_bond_acceptors` labels were generated with the historical
RDKit NumHBA v2.0.1 semantics. RDKit 2026.03 introduced NumHBA v2.0.2 to fix an
N-heterocycle H-bond-acceptor counting bug.

Phase C therefore:
- embeds the RDKit 2025 NumHBA v2.0.1 SMARTS explicitly;
- uses that frozen SMARTS for the HBA rule-access audit;
- uses the frozen HBA count as the descriptor feature named
  `NumHAcceptors_FrozenV201`;
- records the runtime-vs-frozen HBA count differences in
  `phase_c_results/audit/rdkit_hba_version_drift_rows.csv`;
- never relabels the dataset using the newer RDKit definition.

This preserves the scientific target definition while making descriptor-version
drift explicit and auditable.
