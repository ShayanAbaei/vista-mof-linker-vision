# VISTA Phase-B Pre-Scaffold Audit

- **PASS** — Phase-A validation report exists: D:\UNIVERSITY\PROJECTS\MEHRDAD\vista-mof-linker-vision\data\frozen\PHASE_A_VALIDATION_REPORT.json
- **PASS** — Phase-A status PASS: PASS
- **PASS** — Strict canonical overlap = 0: 0
- **PASS** — Strict counts frozen: {'train': 6590, 'test': 1412, 'validation': 1412}
- **PASS** — Zenodo metadata file exists: D:\UNIVERSITY\PROJECTS\MEHRDAD\vista-mof-linker-vision\data\zenodo_record.json
- **PASS** — Zenodo record ID correct: 22141996
- **PASS** — Zenodo DOI correct: 10.5281/zenodo.22141996
- **PASS** — No Zenodo placeholders remain
- **PASS** — VISTA source modules found: 22 .py files
- **PASS** — All VISTA Python modules parse
- **PASS** — Contract: swin_b default
- **PASS** — Contract: embed 384
- **PASS** — Contract: layers 6
- **PASS** — Contract: heads 12
- **PASS** — Contract: labels 10
- **PASS** — Contract: epochs 50
- **PASS** — Contract: patience 10
- **PASS** — Contract: unfreeze 10
- **PASS** — Contract: macro auprc selection
- **PASS** — Contract: base lr 1e-4
- **PASS** — Contract: backbone ratio .1
- **PASS** — Contract: weight decay 1e-2
- **PASS** — Contract: Swin torchvision
- **PASS** — Contract: warmup 10
- **PASS** — Contract: total epochs 50
- **PASS** — Contract: min lr .01
- **PASS** — Contract: ASL clip .005
- **PASS** — Contract: threshold grid 200
- **PASS** — Contract: threshold objective f1
- **PASS** — Contract: original aug documented

## Warnings
- 45 legacy intra-module imports detected; preserve with bootstrap for E0/E1.
- 3 hard-coded CUDA calls detected; inspect before EPFL.

## Legacy internal imports
- `attention_label_image.py:26: from attention import label_to_image_maps, label_to_image_mass, rollout_label_to_image_maps`
- `attention_label_image.py:96: from inference import iter_attention`
- `attention_label_image.py:137: from plotting import academic_rc, save_figure, SINGLE_COL_W`
- `attention_label_image.py:223: from inference import iter_attention, dataset_sample_ids, dataset_filenames, base_dataset`
- `attention_label_label.py:84: from inference import iter_attention`
- `attention_label_label.py:159: from inference import iter_attention, dataset_sample_ids`
- `baselines.py:109: from backbone import create_backbone`
- `baselines.py:115: from backbone import create_backbone`
- `baselines.py:129: from seeding import set_all_seeds`
- `baselines.py:130: from optim import build_optimizer, build_scheduler, LRScheduleConfig`
- `baselines.py:131: from train import MultiLabelTrainer`
- `baselines.py:132: from loss import AsymmetricLossOptimized`
- `baselines.py:133: from experiments import evaluate_split`
- `baselines.py:165: from seeding import set_all_seeds`
- `baselines.py:166: from optim import build_optimizer, build_scheduler, LRScheduleConfig`
- `baselines.py:167: from train import MultiLabelTrainer`
- `baselines.py:168: from loss import AsymmetricLossOptimized`
- `baselines.py:169: from inference import collect_predictions`
- `baselines.py:198: from evaluation import select_thresholds, compute_metric_table`
- `baselines.py:216: from experiments import aggregate_seeds`
- `build_notebook.py:168: from masking import AttentionMaskConfig`
- `build_notebook.py:169: from data import image_augmentation, MultiLabelMoleculeDataset`
- `build_notebook.py:446: from ml_vit import MultiLabelVisionTransformer`
- `build_notebook.py:515: from train import MultiLabelTrainer`
- `checkpointing.py:60: from backbone import create_backbone, BackboneWithModel`
- `checkpointing.py:61: from ml_vit import MultiLabelVisionTransformer`
- `checkpointing.py:62: from masking import AttentionMaskConfig`
- `compare_plots.py:30: from plotting import academic_rc, save_figure, symmetric_robust_limit, DIVERGING_CMAP, SINGLE_COL_W`
- `compare_plots.py:65: from plotting import academic_rc, save_figure, SINGLE_COL_W, PRIMARY`
- `compare_plots.py:99: from plotting import academic_rc, save_figure, PALETTE, DOUBLE_COL_W`
- `evaluation.py:132: from train import MultiLabelMetrics`
- `evaluation.py:190: from plotting import academic_rc, save_figure, SEQUENTIAL_CMAP, DOUBLE_COL_W`
- `experiments.py:29: from masking import AttentionMaskConfig`
- `experiments.py:75: from backbone import create_backbone, BackboneWithModel`
- `experiments.py:76: from ml_vit import MultiLabelVisionTransformer`
- `experiments.py:98: from loss import AsymmetricLossOptimized, BCELoss`
- `experiments.py:205: from inference import collect_predictions`
- `experiments.py:206: from evaluation import select_thresholds, compute_metric_table`
- `experiments.py:231: from seeding import set_all_seeds`
- `experiments.py:232: from optim import build_optimizer, build_scheduler, LRScheduleConfig`
- `experiments.py:233: from train import MultiLabelTrainer`
- `label_groups.py:112: from plotting import academic_rc, save_figure, SPLIT_COLORS, PALETTE, DOUBLE_COL_W`
- `label_label_plots.py:59: from plotting import (academic_rc, save_figure, symmetric_robust_limit,`
- `label_label_plots.py:158: from plotting import academic_rc, save_figure, SEQUENTIAL_CMAP, SINGLE_COL_W`
- `ml_vit.py:8: from masking import (`

## Hard-coded Windows paths
- None

## Hard-coded CUDA calls
- `loss.py:217: There is intentionally **no** hard-coded ``.to("cuda")`` so the loss is`
- `loss.py:271: # these on GPU/CPU (no hard-coded ``.to("cuda")``).`
- `train.py:159: live on the right device (portable; no hard-coded ``.to('cuda')``).`
