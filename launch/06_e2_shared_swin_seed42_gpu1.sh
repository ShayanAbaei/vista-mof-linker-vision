#!/usr/bin/env bash
set -euo pipefail
test -f approvals/E0_APPROVED.txt || { echo "E0 is not approved by Shayan. STOP."; exit 20; }
mkdir -p logs runs/epfl
RUN="runs/epfl/e2_shared_swin_strict_seed42"
test ! -e "$RUN"
CUDA_VISIBLE_DEVICES=1 python -u scripts/train_shared_swin.py \
  --config configs/shared_swin_group_disjoint.yaml --seed 42 --output "$RUN" --device cuda:0 \
  2>&1 | tee logs/06_e2_shared_swin_seed42_gpu1.log
python scripts/postprocess_unique_canonical.py --run-dir "$RUN"
python scripts/validate_output_contract.py --run-dir "$RUN" --require-unique-canonical
