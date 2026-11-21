# Reproducibility and usage

The repository supports three distinct tasks. Do not confuse **checking archived results** with **reproducing a historical trained checkpoint**.

## 1. Check recorded results without a GPU

```bash
python scripts/verify_public_results.py
```

This runs read-only integrity checks against frozen CSV/JSON files, including original VISTA metrics, original/identity-disjoint manifests, molecular baselines, Tanimoto similarity and bootstrap metadata. It uses the Python standard library.

This is **not** a model inference or training test. The original trained VISTA checkpoint was not located in the audited handover and is not distributed.

## 2. Recompute the completed molecular baselines

The archived Phase-C pipeline uses RDKit, NumPy, pandas, scipy and scikit-learn. Use an isolated environment and preferably a fresh disposable clone, since the workflow writes analysis files.

```bash
conda env create -f environment.yml
conda activate vista
python -m pip install rdkit
python scripts/phase_c/run_phase_c.py --stage preflight
python scripts/phase_c/run_phase_c.py --profile standard --stage all
```

The exact archived baseline configuration is `configs/phase_c_local_baselines.json`. CPU baselines include Morgan ECFP4 logistic regression/random forest and a leave-exact-rule-out RDKit descriptor model. Hyperparameters and per-label classification thresholds are selected using validation data, not the strict test set. Keep the frozen historical HBA label semantics.

Relevant machine-readable outputs: `phase_c_results/metrics/`, `phase_c_results/source_data/`, `phase_c_results/bootstrap/` and `phase_c_results/audit/`.

## 3. Train a new VISTA image model on a GPU

The model is implemented in `src/vista/` and its training entry point is `scripts/train_vista.py`. A new training run requires a suitable CUDA/GPU stack, a compatible version of torch and torchvision, pretrained Swin-B weights, and the frozen 9,414 molecular images.

```bash
python scripts/fetch_zenodo_images.py --repo-root .
python scripts/train_vista.py --config configs/vista_sample_repro.yaml --seed 42 --output runs/vista_sample_reproduction --device cuda:0
```

Create the output directory only through the training script; it expects a new destination. Install the PyTorch CUDA packages appropriate for your host separately from `environment.yml`.

**This executes a new GPU training job.** It does not recover the original checkpoint or automatically reproduce its exact floating-point values. Any reported rerun must disclose environment, seeds, data split, checkpoint selection and validation thresholds.

## Experimental scope

The archived original image-model results used the historical sample-level split; they do **not** demonstrate generalization to unseen canonical linker identities. The strict identity-disjoint split currently has recorded **molecular baseline** results, not a strict image-VISTA benchmark. Archived attention maps are descriptive visualizations, not causal attribution validation.

See `docs/RESULTS_AND_SCOPE.md` for verified numerical values and source paths.
