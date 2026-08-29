# VISTA EPFL handover — FINAL run/monitor/return protocol

The collaborator should **run, monitor, and return**. The scientific matrix is frozen.

## 0. Use the exact tagged repository state

Do not reconstruct splits or copy historical folders.

## 1. Environment

Follow `docs/EPFL_ENVIRONMENT.md`.

## 2. Frozen data

```bash
python scripts/fetch_zenodo_images.py --repo-root .
python scripts/pre_epfl_rehearsal.py --repo-root . --require-images
```

## 3. GPU runtime gate

```bash
bash launch/00_runtime_gate.sh
```

Do not start science if it fails.

## 4. E0 reproduction — first and alone

```bash
bash launch/01_e0_vista_seed42_gpu0.sh
```

The command runs the numerical E0 acceptance gate. **Stop after E0 and send the
run folder/log to Shayan.** Even a numeric PASS requires Shayan's review.

After approval, record it:

```bash
python scripts/record_e0_approval.py \
  --note "E0 reviewed by Shayan; proceed with frozen strict matrix."
```

## 5. E1 strict VISTA

Two terminals/tmux sessions:

```bash
bash launch/02_e1_vista_seed17_gpu0.sh
bash launch/03_e1_vista_seed42_gpu1.sh
```

Then first free GPU:

```bash
GPU=0 bash launch/04_e1_vista_seed123_first_free.sh
```

Seeds are 17/42/123. Do not pick the best seed.

## 6. E2 exact strict shared-Swin

This uses the **existing frozen repository architecture**
`src/vista/baselines.py::SharedCNNMultiLabel`:

- Swin-B spatial backbone;
- adaptive global-average pooling;
- ten independent `Linear(1024,1)` heads;
- no label tokens;
- no Transformer;
- no added LayerNorm.

Run launch scripts 05–07.

## 7. E3 one GIN-family baseline

Activate the separate `vista-epfl-gin` environment.

Run:
- 08 = validation-only grid;
- 09/10/11 = final 17/42/123 using the selected validation config.

The tuning run does not evaluate test.

## 8. Strict seed-42 exports

```bash
bash launch/12_seed42_exports_gpu1.sh
```

This produces:
- all frozen latent representations;
- row and canonical aggregates;
- train-fit PCA/UMAP coordinates;
- strict attention summaries;
- full 59x59 final-layer mean-head matrices for candidate cases;
- train-only hub list;
- high/low/random equal-area occlusion;
- explicit negative-label attention controls;
- rotation/translation/scale/padding/line-width stability;
- rendering Brier scores;
- strict performance vs nearest-training Tanimoto;
- optional predeclared MOF-level consequence screen (or an explicit NOT_RUN reason).

## 9. Three-seed summaries

```bash
bash launch/13_summarize_seed_families.sh
```

## 10. Return package

```bash
bash launch/99_build_return_bundle.sh
```

Return the resulting:
- `VISTA_EPFL_RETURN_v1/`
- `VISTA_EPFL_RETURN_v1.tar.gz`

Do not return only screenshots or figures.

## Holdout rules

- train = fit;
- validation = hyperparameters + thresholds;
- test = once after experiment definition is frozen;
- strict test never drives architecture/hyperparameters;
- attention hubs used for confirmatory summaries come from train only;
- no seed selection;
- no split reconstruction;
- no silent source-code changes.

If anything fails, stop and send the exact log.
