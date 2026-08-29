#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs outputs/epfl_runtime_gate
python -u scripts/epfl_runtime_gate.py --output outputs/epfl_runtime_gate --device cuda:0 --batch-size 32 \
  2>&1 | tee logs/00_runtime_gate.log
