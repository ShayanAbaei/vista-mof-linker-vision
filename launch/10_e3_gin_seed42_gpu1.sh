#!/usr/bin/env bash
set -euo pipefail
test -f approvals/E0_APPROVED.txt || { echo "E0 is not approved by Shayan. STOP."; exit 20; }
mkdir -p logs runs/epfl
SEL="runs/epfl/e3_gin_tuning/selected_gin_config.json"
test -f "$SEL" || { echo "Missing GIN validation-selected config: $SEL"; exit 21; }
RUN="runs/epfl/e3_gin_strict_seed42"; test ! -e "$RUN"
CUDA_VISIBLE_DEVICES="1" python -u scripts/train_gin.py \
  --config configs/gin_group_disjoint.yaml --mode final --selected-config "$SEL" \
  --seed 42 --output "$RUN" --device cuda:0 \
  2>&1 | tee logs/10_e3_gin_seed42_gpu1.log
python scripts/postprocess_unique_canonical.py --run-dir "$RUN"
python scripts/validate_output_contract.py --run-dir "$RUN" --require-unique-canonical
