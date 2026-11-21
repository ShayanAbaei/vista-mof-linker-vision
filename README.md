# VISTA — Vision Transformers for MOF Linker Annotation

**Scientific source code, the frozen 2D-linker dataset, and machine-readable research results.**

VISTA is a multi-label computer-vision framework for predicting ten chemically defined *proxy labels* from standardized two-dimensional depictions of selected metal–organic framework (MOF) linkers. Its image model uses an ImageNet-pretrained Swin-B backbone followed by a six-layer label-token Transformer. Ground-truth labels were generated with documented RDKit rules; they are **not measurements of framework adsorption, stability, or other bulk MOF properties**.

The repository archives the results of the **original GPU-trained VISTA experiment**, together with the corresponding predictions, validation-derived thresholds, training records, attention examples, and completed Morgan-fingerprint and RDKit-descriptor comparisons. It also supplies frozen data manifests and scripts for independent checking and subsequent model training.

## Key results

| Record or metric | Value |
|:--|--:|
| 2D linker-image records | **9,414** |
| Unique canonical first-linker SMILES | **5,727** |
| Train / validation / test records | **6,590 / 1,412 / 1,412** |
| Original VISTA test macro-AUPRC | **0.9879** |
| Original VISTA test macro-AUROC | **0.9972** |
| Original VISTA test macro-F1 | **0.9604** |
| Canonical-disjoint Morgan logistic-regression test macro-AUPRC | **0.9258** |

**How to interpret the numbers:** VISTA was genuinely trained on GPU, but the reported image-model test scores come from the *historical sample-level split*. This split includes repeated canonical linker identities across training and evaluation; **630 historical test records have an exact canonical identity in the training data**. These scores must not be called unseen-linker generalization. The separately prepared **canonical-first-linker-disjoint split** has zero identity overlap, but only the completed *molecular* baselines—not a retrained VISTA image model—have been evaluated on it.

See [Results and evaluation scope](docs/RESULTS_AND_SCOPE.md) for comparisons, confidence intervals, and precise limitations.

## Quick start: inspect the archived results

Clone the repository and run the lightweight, standard-library evidence check (no GPU required):

```bash
git clone https://github.com/ShayanAbaei/vista-mof-linker-vision.git
cd vista-mof-linker-vision
python scripts/verify_public_results.py
```

The check verifies selected archived values and file contracts. It **does not retrain** the historical GPU model.

Available source data:

- [Historical image-model predictions](reference_current_results/predictions/) and [metrics / thresholds](reference_current_results/metrics/)
- [Selected attention source arrays](reference_current_results/selected_attention/)
- [Completed molecular baseline metrics](phase_c_results/metrics/), [per-row predictions](phase_c_results/source_data/) and [cluster-bootstrap intervals](phase_c_results/bootstrap/)
- [Current and identity-disjoint split manifests and label definitions](data/frozen/)

## Download the 2D molecular images

The 9,414-image dataset is deposited on Zenodo: **[DOI 10.5281/zenodo.22141996](https://doi.org/10.5281/zenodo.22141996)**. Large image files are not duplicated in Git.

```bash
conda env create -f environment.yml
conda activate vista
python scripts/fetch_zenodo_images.py --repo-root .
```

The downloader checks the archived image ZIP against its frozen hash and expected content. See [Dataset and labels](docs/DATA_AND_LABELS.md).

## Reproduce analyses or train a new model

For the recorded molecular baselines, install RDKit in an isolated environment and run:

```bash
python -m pip install rdkit
python scripts/phase_c/run_phase_c.py --stage preflight
python scripts/phase_c/run_phase_c.py --profile standard --stage all
```

**Run analyses in a disposable clone** because these commands generate output files. The CPU baseline pipeline uses validation-only model selection and thresholds.

Model source is under [`src/vista/`](src/vista/) and the training entry point is [`scripts/train_vista.py`](scripts/train_vista.py). Training a new model requires a compatible PyTorch/torchvision CUDA stack, a GPU, and pretrained Swin-B weights. The original model's trained checkpoint is not distributed; archived predictions and summary metrics are provided instead.

Read the [Reproducibility and usage guide](docs/REPRODUCIBILITY.md) before attempting a new GPU training run.

## Frozen annotation targets

| Label | RDKit-derived binary rule |
|:--|:--|
| `hydrophobicity` | Crippen MolLogP > 2.5 |
| `polarity` | TPSA > 90 |
| `flexibility` | Rotatable-bond count > 5 |
| `aromaticity` | At least one aromatic atom |
| `h_bond_donors` | H-bond donor count > 2 |
| `h_bond_acceptors` | Frozen historical NumHBA v2.0.1 count > 2 |
| `planarity_single_ring` | Ring count = 1 (a **single-ring proxy**, not measured planarity) |
| `macrocycle_gt6_rings` | Ring count > 6 (a **ring-rich proxy**, not necessarily a macrocycle) |
| `charge_present` | Non-zero net formal charge |
| `conjugation_present` | At least one bond with order > 1.5 |

The original label names and rules are preserved in [`data/frozen/label_definitions.yaml`](data/frozen/label_definitions.yaml). Pay particular attention to historical H-bond-acceptor descriptor semantics when using newer RDKit versions.

## Repository contents

| Directory | Contents |
|:--|:--|
| [`src/vista/`](src/vista/) | Vision model, training, inference, loss and analysis source |
| [`data/`](data/) | Frozen input tables, split manifests, validation and label metadata |
| [`reference_current_results/`](reference_current_results/) | Recorded original GPU-model predictions, metrics and attention cases |
| [`phase_c_results/`](phase_c_results/) | Completed Morgan / RDKit benchmarks, similarity audits and uncertainty |
| [`figures/`](figures/) | Original figure files and machine-readable figure sources |
| [`scripts/`](scripts/) | Data download, integrity checks and computational entry points |
| [`docs/`](docs/) | Dataset, results and reproducibility documentation |

## Citation and attribution

See [`CITATION.cff`](CITATION.cff) for the original five-author citation metadata. Cite the frozen [VISTA Zenodo dataset](https://doi.org/10.5281/zenodo.22141996), and credit the QMOF sources: Rosen *et al.*, *Matter* (2021), [10.1016/j.matt.2021.02.015](https://doi.org/10.1016/j.matt.2021.02.015); and Rosen *et al.*, *npj Computational Materials* (2022), [10.1038/s41524-022-00796-6](https://doi.org/10.1038/s41524-022-00796-6).

**Use and licensing:** Dataset rights and attribution are documented with Zenodo and the upstream sources. No software license is asserted here without agreement of the relevant rights holders; public visibility alone does not grant unrestricted reuse rights.
