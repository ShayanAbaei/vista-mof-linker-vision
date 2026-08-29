# EPFL output contract

Every final model run must contain the existing VISTA-style source files:

```text
resolved_config.yaml
command.txt
git_commit.txt
environment.txt
nvidia_smi.txt
pip_freeze.txt
history.csv
best_checkpoint.pt
best_epoch.txt
thresholds.csv
validation_predictions.csv
test_predictions.csv
validation_metrics_overall.csv
test_metrics_overall.csv
validation_metrics_per_label.csv
test_metrics_per_label.csv
confusion_matrices.csv
runtime.json
README_RUN.md
```

Strict model runs additionally require:

```text
unique_canonical_test_predictions.csv
unique_canonical_test_metrics_overall.csv
unique_canonical_test_metrics_per_label.csv
```

Interpretability/robustness return additionally includes:

```text
strict_performance_by_tanimoto_bin.csv
negative_label_attention_manifest.csv
negative_label_attention_summary.csv
attention_occlusion_pairs.csv
attention_occlusion_summary.csv
rendering_stability_rows.csv
rendering_stability_summary.csv
mof_level_consequence_status.json
```

Large scientific outputs are bundled by:

```bash
bash launch/99_build_return_bundle.sh
```

No screenshot-only result is accepted.
