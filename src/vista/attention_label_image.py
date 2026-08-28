"""
Label-to-image attention maps and publication PDFs (spec section 6).

For each label and sample we extract the 49 label->image attention values
(direct final-layer mean-over-heads, or attention rollout), reshape to the 7x7
Swin grid, record the **total label->image mass before spatial normalization**,
and render a PDF (original image | overlay | map+colorbar) plus a numerical NPZ.

Color scale is **fixed from training data** (robust 1st-99th percentiles per
label); individual samples are never independently min-max normalized for the
overlay. The 7x7 map is interpolated only for display, and the maps are coarse
Swin-feature localization -- NOT atom-level attention.

Directory layout (created under ``out_root``):

    <out_root>/<split>/<label_name>/pdf/<sample_id>.pdf
    <out_root>/<split>/<label_name>/data/<sample_id>.npz
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from attention import label_to_image_maps, label_to_image_mass, rollout_label_to_image_maps

METHOD_DIRECT = "direct"
METHOD_ROLLOUT = "rollout"

import re


def safe_dirname(name: str) -> str:
    """
    Make a label name safe to use as a directory/file component on all platforms.

    Replaces characters that are illegal in Windows paths (``< > : " / \\ | ? *``)
    with ``_`` so labels like ``Macrocycle (>6 Rings)`` become
    ``Macrocycle (_6 Rings)``. The original (pretty) label name is still used for
    titles/metadata; only the path component is sanitized.
    """
    return re.sub(r'[<>:"/\\|?*]', '_', str(name)).strip().rstrip('.')


# --------------------------------------------------------------------------- #
# Map extraction
# --------------------------------------------------------------------------- #
def batch_label_image_maps(
    attn_list: Sequence,
    n_labels: int,
    grid_hw: Tuple[int, int] = (7, 7),
    method: str = METHOD_DIRECT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return ``(maps, mass)`` for a batch.

    maps : (B, L, gh, gw) unnormalized attention values.
    mass : (B, L) total label->image attention mass (sum over the 49 keys of the
           final layer, mean over heads) before spatial normalization.

    ``method`` is ``"direct"`` (final-layer mean-over-heads) or ``"rollout"``.
    """
    import torch
    if method == METHOD_DIRECT:
        last = attn_list[-1]
        maps = label_to_image_maps(last, n_labels, grid_hw).cpu().numpy()
        mass = label_to_image_mass(last, n_labels).cpu().numpy()
    elif method == METHOD_ROLLOUT:
        maps = rollout_label_to_image_maps(attn_list, n_labels, grid_hw).cpu().numpy()
        mass = maps.reshape(maps.shape[0], n_labels, -1).sum(-1)
    else:
        raise ValueError(f"Unknown method {method!r}")
    return maps, mass


def spatial_normalize(map_2d: np.ndarray) -> np.ndarray:
    """Sum-to-one spatial normalization (for the saved normalized array)."""
    s = map_2d.sum()
    return map_2d / s if s > 0 else map_2d


# --------------------------------------------------------------------------- #
# Fixed color scale from training data
# --------------------------------------------------------------------------- #
def accumulate_training_color_limits(
    model, train_loader, n_labels: int, device: str = "cpu",
    grid_hw: Tuple[int, int] = (7, 7), method: str = METHOD_DIRECT,
    q_low: float = 1.0, q_high: float = 99.0, max_batches: Optional[int] = None,
) -> np.ndarray:
    """
    Robust per-label color limits ``(L, 2)`` = (1st, 99th percentile) of the
    unnormalized label->image map values across the **training** split. Pass the
    result to the PDF generators so every split shares one fixed scale.
    """
    from inference import iter_attention
    buckets: List[List[np.ndarray]] = [[] for _ in range(n_labels)]
    for _, _, _, attn in iter_attention(model, train_loader, device, max_batches=max_batches):
        maps, _ = batch_label_image_maps(attn, n_labels, grid_hw, method)
        for li in range(n_labels):
            buckets[li].append(maps[:, li].reshape(-1))
    limits = np.zeros((n_labels, 2), dtype=float)
    for li in range(n_labels):
        vals = np.concatenate(buckets[li]) if buckets[li] else np.array([0.0, 1.0])
        limits[li, 0] = float(np.percentile(vals, q_low))
        limits[li, 1] = float(np.percentile(vals, q_high))
        if limits[li, 1] <= limits[li, 0]:
            limits[li, 1] = limits[li, 0] + 1e-6
    return limits


# --------------------------------------------------------------------------- #
# Per-sample PDF
# --------------------------------------------------------------------------- #
def save_label_image_pdf(
    out_dir: Union[str, Path],
    sample_id: str,
    label_name: str,
    raw_image: np.ndarray,
    attn_map: np.ndarray,
    total_mass: float,
    ground_truth: int,
    prob: float,
    pred: int,
    threshold: float,
    method: str,
    color_limit: Tuple[float, float],
    third_panel: bool = True,
) -> Dict[str, str]:
    """
    Write one label->image PDF (+ NPZ) for a single sample/label.

    ``raw_image`` is the original, untouched molecular image as an (H, W, 3)
    uint8/float array. ``attn_map`` is the unnormalized 7x7 map. ``color_limit``
    is the fixed (vmin, vmax) for this label from training data.
    """
    from plotting import academic_rc, save_figure, SINGLE_COL_W
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    pdf_dir = out_dir / "pdf"
    data_dir = out_dir / "data"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    img = np.asarray(raw_image)
    H, W = img.shape[0], img.shape[1]
    vmin, vmax = color_limit
    n_panels = 3 if third_panel else 2

    with academic_rc():
        fig, axes = plt.subplots(1, n_panels, figsize=(SINGLE_COL_W * n_panels, SINGLE_COL_W),
                                 constrained_layout=True)
        # Panel 1: original untouched image.
        axes[0].imshow(img)
        axes[0].set_title("molecule", fontsize=8)
        axes[0].axis("off")

        # Panel 2: overlay (low opacity, magma, interpolated for display only).
        axes[1].imshow(img)
        im = axes[1].imshow(attn_map, cmap="magma", alpha=0.38, vmin=vmin, vmax=vmax,
                            extent=(0, W, H, 0), interpolation="bilinear")
        try:
            axes[1].contour(np.linspace(0, W, attn_map.shape[1]),
                            np.linspace(0, H, attn_map.shape[0]),
                            attn_map, levels=3, colors="white", linewidths=0.4, alpha=0.6)
        except Exception:
            pass
        axes[1].set_title("attention overlay", fontsize=8)
        axes[1].axis("off")

        # Panel 3: attention map alone + colorbar.
        if third_panel:
            im2 = axes[2].imshow(attn_map, cmap="magma", vmin=vmin, vmax=vmax,
                                 interpolation="bilinear")
            axes[2].set_title("attention (7x7, coarse)", fontsize=8)
            axes[2].axis("off")
            fig.colorbar(im2, ax=axes[2], fraction=0.046, label="attention weight")

        fig.suptitle(
            f"{label_name} | id={sample_id} | gt={ground_truth} "
            f"p={prob:.3f} pred={pred} thr={threshold:.3f}\n"
            f"label->image mass={total_mass:.4f} | method={method} "
            f"(coarse Swin-feature localization, not atom-level)",
            fontsize=7,
        )
        written = save_figure(
            fig, pdf_dir / sample_id,
            metadata={"label": label_name, "sample_id": sample_id, "method": method,
                      "ground_truth": int(ground_truth), "prob": float(prob),
                      "pred": int(pred), "threshold": float(threshold),
                      "total_mass": float(total_mass),
                      "color_limit": [float(vmin), float(vmax)]},
        )

    np.savez_compressed(
        data_dir / f"{sample_id}.npz",
        map_unnormalized=attn_map.astype(np.float32),
        map_spatial_normalized=spatial_normalize(attn_map).astype(np.float32),
        total_mass=np.float32(total_mass),
        color_limit=np.asarray(color_limit, np.float32),
        ground_truth=np.int32(ground_truth), prob=np.float32(prob),
        pred=np.int32(pred), threshold=np.float32(threshold),
    )
    return written


def generate_split_label_image_outputs(
    model, loader, split: str, label_names: Sequence[str],
    out_root: Union[str, Path], thresholds: np.ndarray,
    color_limits: np.ndarray, image_dir: Optional[Union[str, Path]] = None,
    device: str = "cpu", method: str = METHOD_DIRECT,
    grid_hw: Tuple[int, int] = (7, 7), max_samples: Optional[int] = None,
    labels_subset: Optional[Sequence[int]] = None,
) -> Dict[str, int]:
    """
    Generate the full label->image PDF/NPZ tree for one split.

    ``image_dir`` (defaults to the dataset's ``image_dir``) is where the original
    untouched images are read for the left panel. ``max_samples`` bounds the work
    (None == all); ``labels_subset`` restricts which labels are rendered.
    """
    from inference import iter_attention, dataset_sample_ids, dataset_filenames, base_dataset
    from PIL import Image

    out_root = Path(out_root)
    image_dir = Path(image_dir) if image_dir is not None else Path(getattr(base_dataset(loader), "image_dir", "."))
    filenames = dataset_filenames(loader)   # mapped through any Subset
    sample_ids = dataset_sample_ids(loader)
    label_idx = list(labels_subset) if labels_subset is not None else list(range(len(label_names)))

    written = 0
    sample_cursor = 0
    for images, targets, probs, attn in iter_attention(model, loader, device):
        maps, mass = batch_label_image_maps(attn, len(label_names), grid_hw, method)
        B = maps.shape[0]
        for b in range(B):
            gidx = sample_cursor + b
            if max_samples is not None and gidx >= max_samples:
                return {"written_pdfs": written}
            sid = sample_ids[gidx] if gidx < len(sample_ids) else f"{split}_{gidx}"
            # Load the original untouched image for display.
            if gidx < len(filenames):
                try:
                    raw = np.asarray(Image.open(image_dir / filenames[gidx]).convert("RGB"))
                except Exception:
                    raw = (images[b].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            else:
                raw = (images[b].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

            for li in label_idx:
                name = label_names[li]
                save_label_image_pdf(
                    out_root / split / safe_dirname(name), sid, name, raw, maps[b, li],
                    float(mass[b, li]), int(targets[b, li].item()),
                    float(probs[b, li].item()),
                    int(probs[b, li].item() > thresholds[li]), float(thresholds[li]),
                    method, (float(color_limits[li, 0]), float(color_limits[li, 1])),
                )
                written += 1
        sample_cursor += B
    return {"written_pdfs": written}
