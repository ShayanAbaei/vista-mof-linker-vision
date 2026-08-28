# VISTA Code Freeze Contract

This document freezes the primary VISTA architecture/training semantics before EPFL reruns.

## Architecture
- TorchVision `swin_b`
- pretrained ImageNet weights (`IMAGENET1K_V1` through the current backbone wrapper)
- 7×7×1024 Swin feature map
- 49 image-feature tokens
- projection 1024→384
- 10 learnable label tokens
- 59-token sequence
- 6 Transformer encoder blocks
- 12 attention heads
- 10 label-specific linear output heads
- unmasked label-token attention

## Training
- historical augmentation (`chemistry_safe=false`) for primary reproduction/strict run
- frequency-adapted ASL from training-split frequencies only
- ASL clip 0.005
- AdamW
- base LR 1e-4
- backbone LR ratio 0.1
- weight decay 1e-2
- first 10 epochs backbone frozen
- 50 epochs maximum
- patience 10
- 10-epoch linear warmup + cosine decay
- minimum LR ratio 0.01
- validation macro-AUPRC checkpoint selection

## Evaluation
- per-label thresholds selected on validation only
- F1 threshold objective
- 200-point grid from 0.001 to 0.999
- thresholds frozen before test

## Labels
CSV columns and display names are separate ordered lists. The model-token order is fixed across both.

## Change policy
Any deviation is a new named experiment. Do not silently alter the primary configuration.


## Historical-reference clarification
- E0 sample-level reproduction uses `chemistry_safe=false` because this reproduces the completed historical augmentation regime.
- The primary strict split remains `chemistry_safe=false` for an apples-to-apples split robustness comparison.
- `configs/vista_group_disjoint_chemistry_safe.yaml` is a separately named sensitivity experiment, not a silent replacement of the primary strict run.
- Historical baseline/ablation outputs are single-seed reference evidence and must not be described as multi-seed confidence estimates.
- Final attention-hub ablations must obey `docs/TEST_HOLDOUT_POLICY.md`.
