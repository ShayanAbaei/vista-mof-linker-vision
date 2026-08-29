#!/usr/bin/env bash
set -euo pipefail
test -f approvals/E0_APPROVED.txt || { echo "E0 is not approved by Shayan. STOP."; exit 20; }
mkdir -p logs outputs/epfl_exports
CK="runs/epfl/e1_vista_strict_seed42/best_checkpoint.pt"
test -f "$CK" || { echo "Missing strict VISTA seed-42 checkpoint: $CK"; exit 22; }

CUDA_VISIBLE_DEVICES=1 python -u scripts/export_embeddings.py \
  --checkpoint "$CK" --config configs/vista_group_disjoint.yaml --repo-root . \
  --output outputs/epfl_exports/embeddings_seed42 --device cuda:0 \
  2>&1 | tee logs/12a_embeddings_seed42.log

python -u scripts/make_dimred_source.py \
  --embeddings outputs/epfl_exports/embeddings_seed42 --config configs/dimred.yaml \
  --output outputs/epfl_exports/dimred_seed42 \
  2>&1 | tee logs/12b_dimred_seed42.log

CUDA_VISIBLE_DEVICES=1 python -u scripts/export_attention_cases.py \
  --checkpoint "$CK" --config configs/vista_group_disjoint.yaml --repo-root . \
  --output outputs/epfl_exports/attention_seed42 --device cuda:0 \
  2>&1 | tee logs/12c_attention_seed42.log

CUDA_VISIBLE_DEVICES=1 python -u scripts/run_attention_controls.py \
  --checkpoint "$CK" --config configs/vista_group_disjoint.yaml \
  --controls-config configs/attention_controls.yaml \
  --attention-dir outputs/epfl_exports/attention_seed42 --repo-root . \
  --output outputs/epfl_exports/occlusion_seed42 --device cuda:0 \
  2>&1 | tee logs/12d_occlusion_seed42.log

python -u scripts/run_negative_attention_controls.py \
  --attention-dir outputs/epfl_exports/attention_seed42 --checkpoint "$CK" \
  --output outputs/epfl_exports/negative_attention_seed42 \
  2>&1 | tee logs/12e_negative_attention_seed42.log

CUDA_VISIBLE_DEVICES=1 python -u scripts/run_rendering_stability.py \
  --checkpoint "$CK" --config configs/vista_group_disjoint.yaml \
  --controls-config configs/attention_controls.yaml --repo-root . \
  --output outputs/epfl_exports/rendering_stability_seed42 --device cuda:0 \
  2>&1 | tee logs/12f_rendering_stability_seed42.log

python -u scripts/performance_by_tanimoto.py \
  --run-dir runs/epfl/e1_vista_strict_seed42 --repo-root . \
  --output outputs/epfl_exports/similarity_seed42 \
  2>&1 | tee logs/12g_similarity_seed42.log

python -u scripts/run_mof_level_consequence.py \
  --repo-root . --output outputs/epfl_exports/mof_consequence \
  2>&1 | tee logs/12h_mof_consequence.log

echo "PASS — strict seed-42 export/control bundle complete."
