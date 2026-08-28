"""
Baselines (spec section 10).

  1. Independent single-label : ten separately trained Swin-B binary classifiers
     (one Swin-B + GAP + a single sigmoid head per property; each property is a
     completely independent model, so there is no information sharing at all).
  2. Shared multi-label CNN    : one Swin-B backbone + global average pooling +
     ten independent linear heads (shares the backbone but has NO label tokens and
     NO label-to-label attention -- isolates the value of the token/attention head
     over a plain shared-CNN multi-label model).
  3. Label-token model         : the main model (Swin-B + 10 label tokens + 6-layer
     Transformer with label-to-label + label-to-image attention), trained via
     ``experiments.ExperimentConfig`` (now unmasked -- the diagonal mask was removed).

The comparison isolates the contribution of each component: (1) vs (2) shows the
value of sharing a backbone; (2) vs (3) shows the value of the label tokens and
their attention. All best models are checkpointed by ``MultiLabelTrainer`` as
``best_model.pth`` under ``out_dir``.

Both baseline modules expose the same ``forward(x, return_attention_weights=False)
-> (logits, None)`` contract and ``freeze_backbone`` / ``unfreeze_backbone`` so the
existing ``MultiLabelTrainer`` can train them unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class _GAPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    def forward(self, x):
        return self.pool(x)


class SharedCNNMultiLabel(nn.Module):
    """Swin-B (spatial) + global average pooling + ten independent linear heads."""

    def __init__(self, backbone, n_labels: int = 10):
        super().__init__()
        self.backbone = backbone
        self.gap = _GAPHead()
        self.heads = nn.ModuleList([nn.Linear(backbone.feature_dim, 1) for _ in range(n_labels)])
        self.n_labels = n_labels

    def forward(self, x, return_attention_weights: bool = False):
        feats = self.gap(self.backbone(x))            # (B, C)
        logits = torch.cat([h(feats) for h in self.heads], dim=1)  # (B, L)
        return (logits, None)

    def freeze_backbone(self):
        self.backbone.freeze()

    def unfreeze_backbone(self):
        self.backbone.unfreeze()


class SingleLabelModel(nn.Module):
    """Swin-B (spatial) + GAP + a single binary head (one per property)."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.gap = _GAPHead()
        self.head = nn.Linear(backbone.feature_dim, 1)

    def forward(self, x, return_attention_weights: bool = False):
        return (self.head(self.gap(self.backbone(x))), None)

    def freeze_backbone(self):
        self.backbone.freeze()

    def unfreeze_backbone(self):
        self.backbone.unfreeze()


class LabelSubsetDataset(Dataset):
    """Adapter exposing a single label column of a multi-label dataset."""

    def __init__(self, base: Dataset, label_index: int):
        self.base = base
        self.label_index = label_index

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, labels = self.base[idx]
        return image, labels[self.label_index:self.label_index + 1]


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def build_shared_cnn(backbone_name: str = "swin_b", n_labels: int = 10,
                     weights: Optional[str] = "default") -> SharedCNNMultiLabel:
    from backbone import create_backbone
    return SharedCNNMultiLabel(create_backbone(backbone_name, weights=weights, spatial=True), n_labels)


def build_single_label_model(backbone_name: str = "swin_b",
                             weights: Optional[str] = "default") -> SingleLabelModel:
    from backbone import create_backbone
    return SingleLabelModel(create_backbone(backbone_name, weights=weights, spatial=True))


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------- #
# Training wrappers (authored; run when data is available)
# --------------------------------------------------------------------------- #
def train_shared_cnn(loaders, label_names, label_frequencies, device="cuda",
                     out_dir="outputs/baselines", epochs=50, base_lr=1e-4, seed=0) -> Dict:
    """Train the shared multi-label CNN baseline and evaluate on test."""
    from seeding import set_all_seeds
    from optim import build_optimizer, build_scheduler, LRScheduleConfig
    from train import MultiLabelTrainer
    from loss import AsymmetricLossOptimized
    from experiments import evaluate_split

    seeds = set_all_seeds(seed)
    model = build_shared_cnn(n_labels=len(label_names))
    criterion = AsymmetricLossOptimized(label_frequencies=label_frequencies)
    optimizer = build_optimizer(model, base_lr=base_lr)
    scheduler = build_scheduler(optimizer, LRScheduleConfig(total_epochs=epochs))
    trainer = MultiLabelTrainer(model, loaders["train"], loaders["val"], criterion,
                                list(label_names), optimizer, scheduler, device=device,
                                checkpoint_dir=str(Path(out_dir) / "shared_cnn"),
                                experiment_name=f"seed{seed}", seeds=seeds)
    trainer.train(num_epochs=epochs)
    best = torch.load(Path(out_dir) / "shared_cnn" / f"seed{seed}" / "best_model.pth",
                      map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    ev = evaluate_split(model, loaders["val"], loaders["test"], label_names, device)
    return {"baseline": "shared_cnn", "seed": seed, "n_params": count_parameters(model),
            "best_epoch": best["epoch"], "per_label": ev["tables"]["per_label"],
            "overall": ev["tables"]["overall"]}


def train_independent_single_label(
    datasets, label_names, label_frequencies, build_loader, device="cuda",
    out_dir="outputs/baselines", epochs=50, base_lr=1e-4, seed=0,
) -> Dict:
    """
    Train ten independent single-label Swin-B classifiers.

    ``datasets`` maps split -> base multi-label Dataset; ``build_loader(dataset,
    shuffle)`` constructs a DataLoader (so augmentation/seeding policy is shared).
    Returns combined per-label test metrics (AUPRC/AUROC/F1/...).
    """
    from seeding import set_all_seeds
    from optim import build_optimizer, build_scheduler, LRScheduleConfig
    from train import MultiLabelTrainer
    from loss import AsymmetricLossOptimized
    from inference import collect_predictions

    test_probs = np.zeros((len(datasets["test"]), len(label_names)))
    test_tgts = np.zeros((len(datasets["test"]), len(label_names)), dtype=int)
    n_params_total = 0
    best_epochs = {}

    for li, name in enumerate(label_names):
        set_all_seeds(seed)
        loaders = {s: build_loader(LabelSubsetDataset(datasets[s], li), shuffle=(s == "train"))
                   for s in ("train", "val", "test")}
        model = build_single_label_model()
        criterion = AsymmetricLossOptimized(label_frequencies=[label_frequencies[li]])
        optimizer = build_optimizer(model, base_lr=base_lr)
        scheduler = build_scheduler(optimizer, LRScheduleConfig(total_epochs=epochs))
        trainer = MultiLabelTrainer(model, loaders["train"], loaders["val"], criterion, [name],
                                    optimizer, scheduler, device=device,
                                    checkpoint_dir=str(Path(out_dir) / "single_label" / name),
                                    experiment_name=f"seed{seed}")
        trainer.train(num_epochs=epochs)
        best = torch.load(Path(out_dir) / "single_label" / name / f"seed{seed}" / "best_model.pth",
                          map_location=device, weights_only=False)
        model.load_state_dict(best["model_state_dict"])
        probs, tgts = collect_predictions(model, loaders["test"], device)
        test_probs[:, li] = probs[:, 0]
        test_tgts[:, li] = tgts[:, 0]
        n_params_total += count_parameters(model)
        best_epochs[name] = best["epoch"]

    from evaluation import select_thresholds, compute_metric_table
    # thresholds from validation reusing the per-label models would require val
    # probs; for the combined table we threshold on test-derived 0.5-free grid via
    # the val split is recommended -- left to the caller. Here we report AUPRC/AUROC
    # (threshold-free) plus 0.5 placeholder thresholds.
    thr = np.full(len(label_names), 0.5)
    tables = compute_metric_table(test_probs, test_tgts, thr, label_names)
    return {"baseline": "independent_single_label", "seed": seed,
            "n_params": n_params_total, "best_epochs": best_epochs,
            "per_label": tables["per_label"], "overall": tables["overall"]}


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate_baseline_metrics(results_by_model: Dict[str, List[Dict]],
                               metric: str = "auprc") -> pd.DataFrame:
    """Mean/std per label across seeds for each model -> long-form DataFrame."""
    from experiments import aggregate_seeds
    frames = []
    for model_name, results in results_by_model.items():
        agg = aggregate_seeds(results, metric).reset_index()
        agg["model"] = model_name
        frames.append(agg)
    return pd.concat(frames, ignore_index=True)
