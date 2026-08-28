# VISTA Phase A Validation Report

**STATUS: PASS**

- Data rows: 9,414
- Unique image names: 9,414
- ZIP PNG members: 9,414
- ZIP/Data missing: 0
- ZIP/Data extra: 0
- Unique canonical first-linker SMILES (current manifest): 5,727
- Current split canonical groups crossing splits: 598
- Strict split canonical groups crossing splits: 0
- Strict split counts: {'train': 6590, 'test': 1412, 'validation': 1412}

## Positive counts

- hydrophobicity: 1,457
- polarity: 3,639
- flexibility: 1,399
- aromaticity: 6,620
- h_bond_donors: 418
- h_bond_acceptors: 7,659
- planarity_single_ring: 2,613
- macrocycle_gt6_rings: 196
- charge_present: 5,459
- conjugation_present: 7,906

## Frozen outputs

- `current_sample_split.csv`
- `canonical_group_disjoint_split.csv`
- `label_definitions.yaml`
- `image_manifest.csv`
- `SHA256SUMS.txt`

No image files were extracted during validation.
