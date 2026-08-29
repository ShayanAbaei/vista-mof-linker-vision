#!/usr/bin/env bash
set -euo pipefail
test -f approvals/E0_APPROVED.txt || { echo "E0 is not approved by Shayan. STOP."; exit 20; }
mkdir -p logs runs/epfl
RUN="runs/epfl/e1_vista_strict_seed17"
test ! -e "$RUN"
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_vista.py \
  --config configs/vista_group_disjoint.yaml --seed 17 --output "$RUN" --device cuda:0 \
  2>&1 | tee logs/02_e1_vista_seed17_gpu0.log
python scripts/finalize_vista_run.py --run-dir "$RUN" --repo-root . \
  --manifest data/frozen/canonical_group_disjoint_split.csv --seed 17
python scripts/postprocess_unique_canonical.py --run-dir "$RUN"
python scripts/validate_output_contract.py --run-dir "$RUN" --require-unique-canonical
