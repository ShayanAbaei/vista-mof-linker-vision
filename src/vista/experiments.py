"""
Controlled intervention and ablation experiments (spec section 9).

Experiment families (all reuse one training path, no duplicated training code):
  A. mask comparison         : none / diagonal-all / diagonal-last-three
  B. dominant-supervision    : zero a dominant label's loss weight, retrain
  C. directed communication  : mask attention edge j->i during train + eval
  D. test-time intervention  : apply an edge mask only at inference (OOD probe)

Every retrained ablation uses identical splits/augmentations, paired seeds, the
same optimizer/schedule and stopping criterion, and >= 3 seeds (5 preferred).
The primary outcome is the change in rare-label test AUPRC vs the full model.

A negative change is described as "the rare-label predictor benefited from that
supervision / communication path under the controlled intervention" -- never as
chemical causation.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from masking import AttentionMaskConfig


# --------------------------------------------------------------------------- #
# Model / criterion construction
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    name: str
    backbone: str = "swin_b"
    embed_dim: int = 384
    n_encoder_layers: int = 6
    n_heads: int = 12
    n_labels: int = 10
    # masking
    attention_mask_mode: str = "none"
    masked_layers: Optional[Sequence[int]] = None
    blocked_label_edges: Optional[Sequence[Tuple[int, int]]] = None
    # supervision ablation: per-label loss weights (None == all ones)
    label_loss_weights: Optional[Sequence[float]] = None
    # loss-function ablation: "asl" | "bce" | "weighted_bce"
    loss_type: str = "asl"
    # training
    base_lr: float = 1e-4
    backbone_lr_ratio: float = 0.1
    weight_decay: float = 1e-2
    epochs: int = 50
    patience: int = 10
    unfreeze_epoch: int = 10
    selection_metric: str = "macro_auprc"
    seeds: Sequence[int] = (0, 1, 2)

    def mask_config(self) -> AttentionMaskConfig:
        return AttentionMaskConfig(
            n_label_tokens=self.n_labels, mode=self.attention_mask_mode,
            masked_layers=self.masked_layers, blocked_label_edges=self.blocked_label_edges,
        )

    def to_dict(self) -> Dict:
        d = self.__dict__.copy()
        d["mask_config"] = self.mask_config().to_dict()
        return d


def build_label_token_model(config: ExperimentConfig, weights: Optional[str] = "default"):
    """Construct ``BackboneWithModel(swin_b, MultiLabelVisionTransformer(...))``."""
    from backbone import create_backbone, BackboneWithModel
    from ml_vit import MultiLabelVisionTransformer

    backbone = create_backbone(config.backbone, weights=weights, spatial=True)
    feat_dim = backbone.feature_dim
    vit = MultiLabelVisionTransformer(
        feature_size=(7, 7), feature_channels=feat_dim, embed_dim=config.embed_dim,
        n_encoder_layers=config.n_encoder_layers, n_heads=config.n_heads,
        n_labels=config.n_labels, mask_config=config.mask_config(),
    )
    return BackboneWithModel(backbone, vit, freeze_backbone=True)


def build_criterion(label_frequencies, label_weights=None, loss_type="asl"):
    """
    Build the training criterion.

    loss_type:
      "asl"          -> frequency-adapted Asymmetric Loss (supports per-label
                        ``label_weights`` for the supervision ablation),
      "bce"          -> plain binary cross-entropy (baseline loss ablation),
      "weighted_bce" -> BCE with per-label pos_weight = (1-freq)/freq.
    """
    from loss import AsymmetricLossOptimized, BCELoss
    if loss_type == "asl":
        return AsymmetricLossOptimized(label_frequencies=label_frequencies, label_weights=label_weights)
    if loss_type == "bce":
        return BCELoss()
    if loss_type == "weighted_bce":
        freq = np.asarray(label_frequencies, dtype=float)
        pos_weight = (1.0 - freq) / np.clip(freq, 1e-6, None)
        return BCELoss(pos_weight=pos_weight.tolist())
    raise ValueError(f"Unknown loss_type={loss_type!r}")


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------- #
# Six-case ablation builders (dominant-supervision and directed-communication)
# --------------------------------------------------------------------------- #
def dominant_removal_sets(dominant_idx: Sequence[int]) -> List[List[int]]:
    """
    The five removal sets used by the 6-case ablations (the 6th case is the full
    model with nothing removed): remove each dominant label, then the first two,
    then the first three. With dominants [Aromaticity, H-Bond Acceptors,
    Conjugation] this yields [[a],[h],[c],[a,h],[a,h,c]].
    """
    d = list(dominant_idx)
    sets = [[i] for i in d]
    if len(d) >= 2:
        sets.append(d[:2])
    if len(d) >= 3:
        sets.append(d[:3])
    return sets


def _short(label_names, idxs):
    return "+".join(label_names[k] for k in idxs)


def supervision_ablation_configs(base: ExperimentConfig, target_idx, label_names,
                                 name_prefix: str = "drop_sup__"):
    """Supervision-removal configs (zero the target label(s)' loss weight). Works
    for dominant labels or attention hubs; pass a distinct ``name_prefix`` to keep
    the two families separable."""
    import dataclasses
    cfgs = []
    for rm in dominant_removal_sets(target_idx):
        w = [0.0 if k in rm else 1.0 for k in range(base.n_labels)]
        cfgs.append(dataclasses.replace(
            base, name=f"{name_prefix}{_short(label_names, rm)}", label_loss_weights=w,
            blocked_label_edges=None))
    return cfgs


def communication_ablation_configs(base: ExperimentConfig, source_idx, receiver_idx,
                                   label_names, name_prefix: str = "block_comm__"):
    """
    Directed-communication configs. For each removal set of *source* labels j, block
    every edge ``j -> i`` to every receiver i (matrix cell A[i,j], encoded as
    blocked_label_edges entry ``(i, j)``; self-edges i==j skipped). Use
    ``receiver_idx = rare labels`` for dominant->rare, or all labels for a hub->all
    block; pass a distinct ``name_prefix`` to separate the families.
    """
    import dataclasses
    cfgs = []
    for rm in dominant_removal_sets(source_idx):
        edges = [(i, j) for j in rm for i in receiver_idx if i != j]
        cfgs.append(dataclasses.replace(
            base, name=f"{name_prefix}{_short(label_names, rm)}",
            blocked_label_edges=edges, label_loss_weights=None))
    return cfgs


# --------------------------------------------------------------------------- #
# Test-time edge intervention (D)
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def test_time_edge_intervention(model, blocked_label_edges, masked_layers=None,
                                mode="none"):
    """
    Temporarily install an attention mask on a *trained* model for inference only.

    This is an out-of-distribution test-time intervention, NOT equivalent to
    controlled retraining. Restores the original mask configuration on exit.
    """
    inner = getattr(model, "model", model)
    old_cfg = inner.mask_config
    old_set = inner._masked_layer_set
    new_cfg = AttentionMaskConfig(
        n_label_tokens=getattr(inner, "n_labels", 10), mode=mode,
        masked_layers=masked_layers, blocked_label_edges=blocked_label_edges,
    )
    inner.mask_config = new_cfg
    inner._masked_layer_set = set(new_cfg.resolve_masked_layers(inner.n_encoder_layers))
    try:
        yield model
    finally:
        inner.mask_config = old_cfg
        inner._masked_layer_set = old_set


# --------------------------------------------------------------------------- #
# Evaluation of a (trained) model -> per-label test metric table
# --------------------------------------------------------------------------- #
def evaluate_split(model, val_loader, test_loader, label_names, device="cpu",
                   objective="f1") -> Dict:
    """Select thresholds on val, evaluate on test; returns metric tables + thresholds."""
    from inference import collect_predictions
    from evaluation import select_thresholds, compute_metric_table

    val_probs, val_tgts = collect_predictions(model, val_loader, device)
    sel = select_thresholds(val_probs, val_tgts, label_names, objective=objective)
    test_probs, test_tgts = collect_predictions(model, test_loader, device)
    tables = compute_metric_table(test_probs, test_tgts, sel["thresholds"], label_names)
    return {"thresholds": sel["thresholds"], "tables": tables,
            "test_probs": test_probs, "test_targets": test_tgts}


# --------------------------------------------------------------------------- #
# One-seed training + evaluation (authored; runs when data is available)
# --------------------------------------------------------------------------- #
def run_single_seed(
    config: ExperimentConfig, loaders: Dict, label_names: Sequence[str],
    label_frequencies, device: str = "cuda", out_dir: str = "outputs/ablations",
    seed: int = 0,
) -> Dict:
    """
    Build -> train -> select thresholds -> evaluate for one seed.

    ``loaders`` must contain ``train``, ``val``, ``test`` (val/test use the
    deterministic eval transforms). Returns a results dict including the per-label
    test metric table, best epoch, parameter count and wall-clock training time.
    """
    from seeding import set_all_seeds
    from optim import build_optimizer, build_scheduler, LRScheduleConfig
    from train import MultiLabelTrainer

    seeds = set_all_seeds(seed)
    model = build_label_token_model(config)
    criterion = build_criterion(label_frequencies, config.label_loss_weights, config.loss_type)

    optimizer = build_optimizer(model, base_lr=config.base_lr,
                                backbone_lr_ratio=config.backbone_lr_ratio,
                                weight_decay=config.weight_decay)
    # Warmup ends when the backbone unfreezes; cosine targets the final epoch.
    scheduler = build_scheduler(optimizer, LRScheduleConfig(
        warmup_epochs=config.unfreeze_epoch, total_epochs=config.epochs))

    trainer = MultiLabelTrainer(
        model, loaders["train"], loaders["val"], criterion, list(label_names),
        optimizer, scheduler, device=device,
        checkpoint_dir=str(Path(out_dir) / config.name), experiment_name=f"seed{seed}",
        selection_metric=config.selection_metric,
        experiment_config=config.to_dict(), seeds=seeds,
    )
    t0 = time.time()
    trainer.train(num_epochs=config.epochs, patience=config.patience,
                  unfreeze_epoch=config.unfreeze_epoch,
                  unfreeze_lr_ratio=config.backbone_lr_ratio)
    train_time = time.time() - t0

    # Reload best checkpoint then evaluate.
    import torch
    best = torch.load(Path(out_dir) / config.name / f"seed{seed}" / "best_model.pth",
                      map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    ev = evaluate_split(model, loaders["val"], loaders["test"], label_names, device)

    return {
        "config": config.name, "seed": seed, "best_epoch": best["epoch"],
        "train_time_s": train_time, "n_params": count_parameters(model),
        "selection_metric": config.selection_metric,
        "thresholds": ev["thresholds"],
        "per_label": ev["tables"]["per_label"], "overall": ev["tables"]["overall"],
    }


def run_experiment(config: ExperimentConfig, loaders, label_names, label_frequencies,
                   device="cuda", out_dir="outputs/ablations") -> List[Dict]:
    """Run all seeds of an experiment; returns a list of per-seed result dicts."""
    return [run_single_seed(config, loaders, label_names, label_frequencies, device,
                            out_dir, seed) for seed in config.seeds]


# --------------------------------------------------------------------------- #
# Seed-level aggregation
# --------------------------------------------------------------------------- #
def aggregate_seeds(results: Sequence[Dict], metric: str = "auprc") -> pd.DataFrame:
    """
    Aggregate a per-label ``metric`` across seeds -> mean / std / 95% CI per label.
    """
    mats = np.stack([r["per_label"][metric].values for r in results], axis=0)  # (S, L)
    labels = results[0]["per_label"].index.tolist()
    mean = mats.mean(0)
    std = mats.std(0, ddof=1) if mats.shape[0] > 1 else np.zeros(mats.shape[1])
    sem = std / np.sqrt(mats.shape[0])
    ci = 1.96 * sem
    return pd.DataFrame({
        "label": labels, f"{metric}_mean": mean, f"{metric}_std": std,
        f"{metric}_ci_low": mean - ci, f"{metric}_ci_high": mean + ci,
        "n_seeds": mats.shape[0],
    }).set_index("label")


def rare_auprc_change_matrix(
    full_agg: pd.DataFrame, ablated_aggs: Dict[str, pd.DataFrame],
    rare_labels: Sequence[str], metric: str = "auprc",
) -> pd.DataFrame:
    """
    Build the change matrix: rows = rare target labels, cols = removed/masked
    dominant source labels, values = (ablated - full) mean metric for the rare
    target. Negative => the rare predictor benefited from that source.
    """
    col = f"{metric}_mean"
    rows = {}
    for rare in rare_labels:
        full_v = float(full_agg.loc[rare, col])
        rows[rare] = {src: float(ab.loc[rare, col]) - full_v for src, ab in ablated_aggs.items()}
    return pd.DataFrame(rows).T  # rows=rare, cols=source
