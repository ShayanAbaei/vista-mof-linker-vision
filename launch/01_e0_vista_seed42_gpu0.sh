#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs runs/epfl
RUN="runs/epfl/e0_vista_sample_seed42"
test ! -e "$RUN"
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_vista.py \
  --config configs/vista_sample_repro.yaml --seed 42 --output "$RUN" --device cuda:0 \
  2>&1 | tee logs/01_e0_vista_seed42_gpu0.log
python scripts/finalize_vista_run.py --run-dir "$RUN" --repo-root . \
  --manifest data/frozen/current_sample_split.csv --seed 42
python scripts/postprocess_unique_canonical.py --run-dir "$RUN"
python scripts/validate_output_contract.py --run-dir "$RUN" --require-unique-canonical
python scripts/e0_acceptance_gate.py --run-dir "$RUN" --repo-root .
echo
echo "STOP. Send E0 outputs/log to Shayan. Do not start E1 until approved."
