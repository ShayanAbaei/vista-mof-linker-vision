#!/usr/bin/env bash
set -euo pipefail
test -f approvals/E0_APPROVED.txt || { echo "E0 is not approved by Shayan. STOP."; exit 20; }
mkdir -p logs runs/epfl
RUN="runs/epfl/e1_vista_strict_seed42"
test ! -e "$RUN"
CUDA_VISIBLE_DEVICES=1 python -u scripts/train_vista.py \
  --config configs/vista_group_disjoint.yaml --seed 42 --output "$RUN" --device cuda:0 \
  2>&1 | tee logs/03_e1_vista_seed42_gpu1.log
python scripts/finalize_vista_run.py --run-dir "$RUN" --repo-root . \
  --manifest data/frozen/canonical_group_disjoint_split.csv --seed 42
python scripts/postprocess_unique_canonical.py --run-dir "$RUN"
python scripts/validate_output_contract.py --run-dir "$RUN" --require-unique-canonical
