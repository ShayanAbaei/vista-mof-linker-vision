# VISTA dataset, frozen split protocol, and labels

VISTA uses **9,414** 2D depictions of selected linkers associated with QMOF/MOFid records, corresponding to **5,727** unique canonical first-linker SMILES. The images describe selected linker structures—not 9,414 independent MOF chemistry measurements.

## Obtain the image archive

The frozen images are distributed separately at [Zenodo DOI 10.5281/zenodo.22141996](https://doi.org/10.5281/zenodo.22141996). From the repository root:

```bash
conda env create -f environment.yml
conda activate vista
python scripts/fetch_zenodo_images.py --repo-root .
```

The download tool validates the expected ZIP hash and image count.

## Data files

- `data/source/Data.csv`: compact image-to-label source table.
- `data/source/qmof_with_12_labels_final_cleaned.csv`: QMOF/MOFid-linked enriched table used in validation.
- `data/frozen/current_sample_split.csv`: historical row-level split used for the reported VISTA image-model scores.
- `data/frozen/canonical_group_disjoint_split.csv`: alternative canonical-first-linker-identity-disjoint split, used by the completed molecular baselines.
- `data/frozen/label_definitions.yaml`: frozen binary label rules.
- `data/frozen/PHASE_A_VALIDATION_REPORT.json`: recorded checks and counts.
- `data/frozen/SHA256SUMS.txt`: frozen source/archive hash records.

Both manifests have 6,590 training, 1,412 validation and 1,412 test **image records**. The original split has 598 canonical groups crossing partitions, including 630 test images with an exact canonical linker identity in the training data. The alternative split has zero crossing canonical identities. Identity-disjoint does not mean scaffold-disjoint or low-Tanimoto.

## Chemical target interpretation

All labels were generated using ten deterministic RDKit-derived binary **proxy** rules, as frozen in `label_definitions.yaml`. These are not experimentally measured MOF properties.

In particular, the label historically named `planarity_single_ring` tests **ring count = 1**, not geometry or aromatic planarity. `macrocycle_gt6_rings` tests **ring count > 6**, not the existence of an isolated macrocycle.

Historical H-bond-acceptor annotation uses the earlier RDKit NumHBA v2.0.1 SMARTS semantics. This must not be silently substituted with a new toolkit default.

## Origin and licensing

Credit QMOF/MOFid and the underlying research: Rosen *et al.*, *Matter* (2021), DOI 10.1016/j.matt.2021.02.015, and Rosen *et al.*, *npj Computational Materials* (2022), DOI 10.1038/s41524-022-00796-6. Check Zenodo record licensing and original data attribution before redistributing derived material.
