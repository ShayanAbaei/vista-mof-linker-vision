#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs/epfl_summaries
python scripts/summarize_seed_family.py \
  --family vista_strict \
  --run-dirs runs/epfl/e1_vista_strict_seed17 runs/epfl/e1_vista_strict_seed42 runs/epfl/e1_vista_strict_seed123 \
  --output outputs/epfl_summaries/vista_strict
python scripts/summarize_seed_family.py \
  --family shared_swin_strict \
  --run-dirs runs/epfl/e2_shared_swin_strict_seed17 runs/epfl/e2_shared_swin_strict_seed42 runs/epfl/e2_shared_swin_strict_seed123 \
  --output outputs/epfl_summaries/shared_swin_strict
python scripts/summarize_seed_family.py \
  --family gin_strict \
  --run-dirs runs/epfl/e3_gin_strict_seed17 runs/epfl/e3_gin_strict_seed42 runs/epfl/e3_gin_strict_seed123 \
  --output outputs/epfl_summaries/gin_strict
