#!/usr/bin/env bash
set -euo pipefail
test -f approvals/E0_APPROVED.txt || { echo "E0 is not approved by Shayan. STOP."; exit 20; }
mkdir -p logs runs/epfl
SEL="runs/epfl/e3_gin_tuning/selected_gin_config.json"
test -f "$SEL" || { echo "Missing GIN validation-selected config: $SEL"; exit 21; }
RUN="runs/epfl/e3_gin_strict_seed17"; test ! -e "$RUN"
CUDA_VISIBLE_DEVICES="0" python -u scripts/train_gin.py \
  --config configs/gin_group_disjoint.yaml --mode final --selected-config "$SEL" \
  --seed 17 --output "$RUN" --device cuda:0 \
  2>&1 | tee logs/09_e3_gin_seed17_gpu0.log
python scripts/postprocess_unique_canonical.py --run-dir "$RUN"
python scripts/validate_output_contract.py --run-dir "$RUN" --require-unique-canonical
