#!/usr/bin/env bash
set -euo pipefail
test -f approvals/E0_APPROVED.txt || { echo "E0 is not approved by Shayan. STOP."; exit 20; }
mkdir -p logs runs/epfl
RUN="runs/epfl/e2_shared_swin_strict_seed17"
test ! -e "$RUN"
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_shared_swin.py \
  --config configs/shared_swin_group_disjoint.yaml --seed 17 --output "$RUN" --device cuda:0 \
  2>&1 | tee logs/05_e2_shared_swin_seed17_gpu0.log
python scripts/postprocess_unique_canonical.py --run-dir "$RUN"
python scripts/validate_output_contract.py --run-dir "$RUN" --require-unique-canonical
