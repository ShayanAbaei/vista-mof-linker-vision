"""
Inference helpers: collect probabilities, targets, filenames and per-layer
attention from a model over a (deterministic, non-augmented) loader.

IMPORTANT: always build the loader with the *evaluation* transforms (see
``data.image_augmentation`` -> ``val_transforms``), even for the training split,
so attention maps are never produced from randomly augmented images.
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch


@torch.no_grad()
def collect_predictions(model, loader, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(probs, targets)`` as (N, L) numpy arrays."""
    model.eval()
    probs, tgts = [], []
    for images, targets in loader:
        logits, _ = model(images.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        tgts.append(np.asarray(targets))
    return np.concatenate(probs), np.concatenate(tgts)


def _base_and_indices(ds):
    """Return (underlying_dataset, list_of_base_indices), unwrapping a Subset."""
    from torch.utils.data import Subset
    if isinstance(ds, Subset):
        return ds.dataset, list(ds.indices)
    return ds, list(range(len(ds)))


def base_dataset(loader):
    """The underlying dataset (unwraps a Subset) -- e.g. for ``.image_dir``."""
    return _base_and_indices(loader.dataset)[0]


def dataset_filenames(loader) -> List[str]:
    base, idxs = _base_and_indices(loader.dataset)
    names = getattr(base, "filenames", None)
    if names is not None:
        return [names[i] for i in idxs]
    return [f"sample_{i}" for i in idxs]


def dataset_sample_ids(loader) -> List[str]:
    """Stable per-sample ids (filename stem if available, else index)."""
    from pathlib import Path
    base, idxs = _base_and_indices(loader.dataset)
    names = getattr(base, "filenames", None)
    if names is not None:
        return [Path(names[i]).stem for i in idxs]
    return [str(i) for i in idxs]


def select_balanced_sample_indices(
    Y: np.ndarray, max_samples: int = 200, min_pos_per_label: int = 10, seed: int = 1234,
) -> Tuple[List[int], np.ndarray]:
    """
    Select a label-balanced subset of sample indices for per-sample attention PDFs.

    Guarantees up to ``min_pos_per_label`` positive samples for **every** label
    (rare labels chosen first, so they aren't starved), then fills the remainder
    with random samples up to ``max_samples``. Because each molecule is multi-label,
    one chosen sample covers several labels at once.

    Returns ``(indices, per_label_positive_counts)`` where ``indices`` indexes rows
    of ``Y`` (i.e. positions in the loader's dataset order).
    """
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y).astype(int)
    N, L = Y.shape
    selected: List[int] = []
    chosen = set()
    prevalence = Y.mean(axis=0)
    # Pass 1: rarest labels first, top up each to min_pos_per_label.
    for li in np.argsort(prevalence):
        pos = np.where(Y[:, li] == 1)[0]
        rng.shuffle(pos)
        have = int(Y[list(chosen), li].sum()) if chosen else 0
        for idx in pos:
            if have >= min_pos_per_label or len(selected) >= max_samples:
                break
            if idx in chosen:
                continue
            chosen.add(int(idx)); selected.append(int(idx))
            have += 1
    # Pass 2: fill remaining slots with random other samples for diversity.
    if len(selected) < max_samples:
        rest = [i for i in range(N) if i not in chosen]
        rng.shuffle(rest)
        for idx in rest[: max_samples - len(selected)]:
            chosen.add(int(idx)); selected.append(int(idx))
    selected = selected[:max_samples]
    counts = Y[selected].sum(axis=0).astype(int) if selected else np.zeros(L, int)
    return selected, counts


def subset_loader(loader, indices, batch_size: Optional[int] = None):
    """Build a deterministic DataLoader over a Subset of ``loader`` (the given
    dataset indices), so attention is only computed for the selected samples."""
    from torch.utils.data import Subset, DataLoader
    bs = batch_size or getattr(loader, "batch_size", None) or 32
    return DataLoader(Subset(loader.dataset, list(indices)), batch_size=bs,
                      shuffle=False, num_workers=0)


@torch.no_grad()
def iter_attention(
    model, loader, device: str = "cpu", max_batches: Optional[int] = None,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
    """
    Yield ``(images_cpu, targets, probs, attn_list)`` per batch, where
    ``attn_list`` is the per-layer list of ``(B, H, S, S)`` tensors (on CPU).
    """
    model.eval()
    for bi, (images, targets) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        logits, attn = model(images.to(device), return_attention_weights=True)
        probs = torch.sigmoid(logits).cpu()
        attn_cpu = [a.cpu() for a in attn]
        yield images.cpu(), torch.as_tensor(targets).cpu(), probs, attn_cpu
