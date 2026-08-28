"""
Checkpoint loading and compatibility reporting (spec section 3F).

Handles both the new rich checkpoints (with ``model_config`` / ``mask_config`` /
``best_score`` / ``selection_metric``) and the original executed checkpoints
(which used the ``best_val`` loss key), so existing weights keep loading.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch


def load_checkpoint(path: Union[str, Path], map_location: str = "cpu") -> Dict:
    """Load a checkpoint dict (``weights_only=False`` to recover config/metadata)."""
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    return ckpt


def checkpoint_report(ckpt: Dict) -> Dict:
    """Normalize the selection/metadata fields across old and new checkpoints."""
    report = {
        "epoch": ckpt.get("epoch"),
        "label_names": ckpt.get("label_names"),
        "selection_metric": ckpt.get("selection_metric", "val_loss (legacy)"),
        "best_score": ckpt.get("best_score"),
        "best_val_loss": ckpt.get("best_val_loss", ckpt.get("best_val")),
        "best_val_f1": ckpt.get("best_val_f1"),  # legacy F1-selected checkpoints
        "mask_config": ckpt.get("mask_config") or (ckpt.get("model_config") or {}).get("mask_config"),
        "loss_config": ckpt.get("loss_config"),
        "thresholds": ckpt.get("thresholds"),
        "seeds": ckpt.get("seeds"),
        "has_optimizer": "optimizer_state_dict" in ckpt,
        "has_scheduler": ckpt.get("scheduler_state_dict") is not None,
        "n_params_saved": len(ckpt.get("model_state_dict", {})),
    }
    return report


def load_state_dict_tolerant(model: torch.nn.Module, state_dict: Dict) -> Dict:
    """
    Load weights without raising on key mismatch; return the missing/unexpected
    key lists so the notebook can surface any incompatibility explicitly.
    """
    result = model.load_state_dict(state_dict, strict=False)
    return {"missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys)}


def build_model_from_checkpoint(
    ckpt: Dict, backbone: str = "swin_b", weights: Optional[str] = None,
    embed_dim: int = 384, n_heads: int = 12, n_encoder_layers: int = 6,
) -> Tuple[torch.nn.Module, Dict]:
    """
    Reconstruct a ``BackboneWithModel`` from a checkpoint's saved config (falling
    back to defaults), load its weights tolerantly, and return ``(model, report)``.
    """
    from backbone import create_backbone, BackboneWithModel
    from ml_vit import MultiLabelVisionTransformer
    from masking import AttentionMaskConfig

    cfg = ckpt.get("model_config") or {}
    label_names = ckpt.get("label_names") or [f"label_{i}" for i in range(cfg.get("n_labels", 10))]
    n_labels = cfg.get("n_labels", len(label_names))
    n_layers = cfg.get("n_encoder_layers", n_encoder_layers)
    mask_cfg = AttentionMaskConfig.from_dict(cfg.get("mask_config"))

    bb = create_backbone(backbone, weights=weights, spatial=True)
    vit = MultiLabelVisionTransformer(
        feature_size=(7, 7), feature_channels=bb.feature_dim, embed_dim=embed_dim,
        n_encoder_layers=n_layers, n_heads=n_heads, n_labels=n_labels, mask_config=mask_cfg,
    )
    model = BackboneWithModel(bb, vit, freeze_backbone=True)
    load_report = load_state_dict_tolerant(model, ckpt["model_state_dict"])
    report = {**checkpoint_report(ckpt), **load_report,
              "compatible": not load_report["missing_keys"] and not load_report["unexpected_keys"]}
    return model, report
