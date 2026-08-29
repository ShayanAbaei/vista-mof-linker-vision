#!/usr/bin/env bash
set -euo pipefail
test -f approvals/E0_APPROVED.txt || { echo "E0 is not approved by Shayan. STOP."; exit 20; }
mkdir -p logs runs/epfl
RUN="runs/epfl/e3_gin_tuning"; test ! -e "$RUN"
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_gin.py \
  --config configs/gin_group_disjoint.yaml --mode tune --output "$RUN" --device cuda:0 \
  2>&1 | tee logs/08_e3_gin_tune_gpu0.log
