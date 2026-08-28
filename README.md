# VISTA

**VISTA: Chemically Auditable Vision Transformers for Multi-Label Annotation of MOF Linker Functionality**

> **Development status:** pre-EPFL validation / reproducibility hardening.

VISTA is an image-based multi-label learning framework for standardized two-dimensional depictions of organic linkers derived from the QMOF/MOFid ecosystem. The current benchmark contains 9,414 linker-image records corresponding to 5,727 unique canonical first-linker SMILES and ten deterministic RDKit-derived binary proxy labels.

## Scientific status

The repository preserves the current sample-level reference results while the strengthened canonical-first-linker-disjoint benchmark, repeated-seed runs, graph comparison, latent-space analysis, and attention controls are being prepared.

Current sample-level values are retained only as **reference results**, not as evidence that the strict canonical-disjoint experiment has already been completed.

## Data

The frozen dataset is archived separately on Zenodo.

**Zenodo DOI:** `10.5281/zenodo.22141996`  
**Zenodo record:** https://zenodo.org/records/22141996

The heavy molecular-image archive is intentionally not tracked in normal Git history.

See [`data/README.md`](data/README.md).

## Repository structure

- `data/` â€” source and frozen manifests; no extracted 9,414-image corpus in Git.
- `src/vista/` â€” imported authoritative VISTA source modules, pending reproducibility refactor.
- `reference_current_results/` â€” current sample-level predictions, metrics, and selected source artifacts.
- `figures/` â€” figure source tables/current reference outputs.
- `scripts/` â€” dataset validation, download, and later reproducible execution entry points.
- `docs/` â€” project state, dataset/model documentation, and EPFL handover documentation.
- `manuscript/` â€” reserved for the verified latest Overleaf source.

## Ten deterministic proxy labels

1. Hydrophobicity â€” Crippen logP > 2.5
2. Polarity â€” TPSA > 90
3. Flexibility â€” rotatable bonds > 5
4. Aromaticity â€” at least one aromatic atom
5. H-bond donors â€” RDKit HBD count > 2
6. H-bond acceptors â€” RDKit HBA count > 2
7. Single-ring proxy â€” ring count = 1
8. Ring-rich (>6 rings) proxy â€” ring count > 6
9. Charge present â€” net formal charge != 0
10. Conjugation present â€” any bond order > 1.5

These labels are reproducible **proxy annotations**, not experimental measurements or universal chemical definitions.

## Strict split

The canonical-first-linker-disjoint split prevents the same canonical first-linker SMILES from occurring in more than one of train, validation, and test. It tests unseen canonical linker identity, not complete scaffold/chemical dissimilarity.

## Reproducibility

Before GPU work:

```bash
python scripts/validate_dataset.py
```

The exact EPFL training/launch commands will be added after the final code/config/environment audit.

## QMOF provenance

VISTA is derived from QMOF/MOFid-linked data. QMOF data are distributed under CC BY 4.0. Users should cite the QMOF database publications:

- Rosen et al., *Matter* (2021), DOI: 10.1016/j.matt.2021.02.015
- Rosen et al., *npj Computational Materials* (2022), DOI: 10.1038/s41524-022-00796-6

## Citation

A final paper citation will be added after publication. See [`CITATION.cff`](CITATION.cff) for project authorship metadata.

## License

Dataset licensing is documented with the Zenodo record. A software license for the code will be finalized with the code authors before the repository is made public.


