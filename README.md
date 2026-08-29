# VISTA

**VISTA: Chemically Auditable Vision Transformers for Multi-Label Annotation of MOF Linker Functionality**

> **Development status:** final pre-EPFL handover review. Phase A/B/C and stable manuscript cleanup are complete; GPU scientific execution has not started.

VISTA is an image-based multi-label learning framework for standardized two-dimensional depictions of organic linkers derived from the QMOF/MOFid ecosystem. The frozen benchmark contains 9,414 linker-image records corresponding to 5,727 unique canonical first-linker SMILES and ten deterministic RDKit-derived binary proxy labels.

## Scientific status

Completed: frozen data/manifests/Zenodo archive, canonical-first-linker-disjoint split, reproducibility/code freeze, historical current-split VISTA reference preservation, Phase-C CPU baselines/Tanimoto/bootstrap, and stable pre-EPFL manuscript cleanup.

Still pending: E0 sample-level seed-42 reproduction on EPFL, strict VISTA seeds 17/42/123, strict shared-Swin seeds 17/42/123, one validation-selected GIN-family baseline, strict seed-42 embeddings/PCA/UMAP/attention controls, independent returned-result validation, and the final manuscript/release.

Current sample-level VISTA values are retained only as **historical reference results**, not as evidence that the strict canonical-first-linker-disjoint experiment has already been completed.

## EPFL collaborator: start here

1. Environment: [`docs/EPFL_ENVIRONMENT.md`](docs/EPFL_ENVIRONMENT.md)
2. Full run/monitor/return protocol: [`docs/EPFL_HANDOVER.md`](docs/EPFL_HANDOVER.md)
3. GPU runtime gate: [`docs/GPU_RUNTIME_GATE.md`](docs/GPU_RUNTIME_GATE.md)
4. Output contract: [`docs/OUTPUT_CONTRACT.md`](docs/OUTPUT_CONTRACT.md)
5. Fresh-clone rehearsal: [`docs/FRESH_CLONE_REHEARSAL.md`](docs/FRESH_CLONE_REHEARSAL.md)
6. Exact launch order: [`launch/README_GPU.md`](launch/README_GPU.md)

The frozen order is: runtime gate → E0 seed42 → **STOP for Shayan approval** → strict VISTA → exact shared-Swin → validation-selected GIN-family → seed-42 exports/controls → repeated-seed summaries → automated return bundle.

Do not tune on the strict test set, select a best seed, reconstruct the split, or silently change the frozen scientific matrix.

## Data

The frozen dataset is archived separately on Zenodo.

**Zenodo DOI:** `10.5281/zenodo.22141996`  
**Zenodo record:** https://zenodo.org/records/22141996

The heavy molecular-image archive is intentionally not tracked in normal Git history.

```bash
python scripts/fetch_zenodo_images.py --repo-root .
python scripts/pre_epfl_rehearsal.py --repo-root . --require-images
```

See [`data/README.md`](data/README.md).

## Strict split

The canonical-first-linker-disjoint split prevents the same canonical first-linker SMILES from occurring in more than one of train, validation, and test.

Frozen counts: train 6,590; validation 1,412; test 1,412; total 9,414.

It tests generalization to **unseen canonical first-linker identities**. It is not a scaffold-disjoint or complete-chemical-dissimilarity split. Final strict reporting includes both sample-weighted and unique-canonical-linker metrics.

## Ten deterministic proxy labels

1. Hydrophobicity — Crippen logP > 2.5
2. Polarity — TPSA > 90
3. Flexibility — rotatable bonds > 5
4. Aromaticity — at least one aromatic atom
5. H-bond donors — RDKit HBD count > 2
6. H-bond acceptors — frozen historical RDKit NumHBA semantics > 2
7. Single-ring proxy — ring count = 1
8. Ring-rich (>6 rings) proxy — ring count > 6
9. Charge present — net formal charge != 0
10. Conjugation present — any bond order > 1.5

These labels are reproducible **proxy annotations**, not experimental measurements or universal chemical definitions.

## Repository structure

- `data/` — frozen source/manifests and validation artifacts.
- `src/vista/` — authoritative VISTA source and frozen baseline implementations.
- `reference_current_results/` — historical current-split predictions, metrics, and provenance.
- `phase_c_results/` — completed local CPU baseline/Tanimoto/bootstrap outputs.
- `configs/` — frozen VISTA/shared-Swin/GIN/dimred/control configs.
- `scripts/` — validation, training, export, post-processing, and handover utilities.
- `launch/` — direct Linux two-GPU launch queue.
- `docs/` — reproducibility, holdout, environment, handover, and output-contract documentation.
- `manuscript/` — manuscript-state documentation; the verified working Overleaf project is maintained separately until final integration.

## Reproducibility gates

Local/static:
```bash
python scripts/pre_epfl_rehearsal.py --repo-root .
```

Fresh clone:
```bash
python scripts/fresh_clone_rehearsal.py --repo-url https://github.com/ShayanAbaei/vista-mof-linker-vision.git --ref <exact-reviewed-ref> --fetch-images
```

EPFL GPU:
```bash
bash launch/00_runtime_gate.sh
```

The GPU runtime gate must pass before any scientific GPU job.

## Run outputs and return bundle

Every final model run returns config/command/Git/environment provenance, history, checkpoint, validation-derived thresholds, validation/test predictions, metrics, confusion matrices, and runtime metadata. Strict runs additionally return unique-canonical metrics.

After all runs/exports:
```bash
bash launch/99_build_return_bundle.sh
```

This creates the structured `VISTA_EPFL_RETURN_v1` research object for independent validation and later plotting/manuscript work.

## Embeddings and figures

Strict VISTA seed 42 is the reference model for the five frozen latent representations, canonical aggregates, train-fit PCA/UMAP coordinates, strict attention, negative-label controls, occlusion, rendering stability, Tanimoto-stratified performance, and the optional predeclared MOF-level consequence screen.

Final publication figures must be regenerated from machine-readable source data rather than preserved only as PDFs/PNGs.

## QMOF provenance

VISTA is derived from QMOF/MOFid-linked data. QMOF data are distributed under CC BY 4.0. Users should cite:

- Rosen et al., *Matter* (2021), DOI: 10.1016/j.matt.2021.02.015
- Rosen et al., *npj Computational Materials* (2022), DOI: 10.1038/s41524-022-00796-6

## Citation

A final paper citation will be added after publication. See [`CITATION.cff`](CITATION.cff) for project authorship metadata.

## License

Dataset licensing is documented with the Zenodo record. A software license for the code will be finalized with the code authors before the repository is made public.
