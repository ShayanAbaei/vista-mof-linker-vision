# VISTA Dataset Card

## Scope

VISTA uses standardized two-dimensional depictions of QMOF/MOFid-derived first-linker records for ten deterministic binary proxy-label tasks.

## Size

- Samples: 9,414
- Unique canonical first-linker SMILES: 5,727
- Sample-level split: 6,590 train / 1,412 validation / 1,412 test
- Canonical-first-linker-disjoint split: same target sample counts; identical canonical linker identities may not cross splits

## Labels

| Label | Rule | Important limitation |
|---|---|---|
| Hydrophobicity | Crippen logP > 2.5 | free-linker threshold |
| Polarity | TPSA > 90 | global threshold |
| Flexibility | rotatable bonds > 5 | not framework flexibility |
| Aromaticity | ≥1 aromatic atom | does not quantify aromatic extent |
| HBD | HBD count > 2 | no accessibility information |
| HBA | HBA count > 2 | no accessibility information |
| Single-ring proxy | ring count = 1 | not geometric planarity |
| Ring-rich proxy | ring count > 6 | not a universal macrocycle definition |
| Charge present | net formal charge != 0 | representation/protonation dependent |
| Conjugation present | any bond order > 1.5 | simplified conjugation proxy |

## Intended use

Controlled benchmarking of image-based multi-label molecular annotation and auditable representation analyses.

## Non-intended use

The labels should not be treated as experimental ground truth, universal chemical definitions, or direct descriptors of the full MOF framework environment.

## Provenance

Derived from QMOF/MOFid-linked records. QMOF data are available under CC BY 4.0.
