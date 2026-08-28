"""
Label-to-label attention analysis (spec section 7).

Per sample/layer/head we take the 10x10 label-to-label block, remove the
diagonal, and compute, for each query label i:

    M_i      = sum_{j != i} A[i, j]            (absolute off-diagonal mass)
    P[i, j]  = A[i, j] / (M_i + eps)           (conditional label preference)

The headline table is **not** a naive sample average (dominant labels would bias
it). Instead, for each directed relation j -> i (matrix cell A[i, j]) we compute
the imbalance-balanced conditional effect:

    delta[i, j] = E[P[i, j] | y_i=1, y_j=1] - E[P[i, j] | y_i=1, y_j=0]

conditioning on y_i = 1 and giving the y_j=1 and y_j=0 strata equal estimand
weight (each stratum mean is computed separately, then subtracted). Relations
with insufficient support in either stratum are marked unreliable (NaN), never
replaced with zero.

Direction convention (see ``masking``): A[i, j] is query i reading key j, i.e.
information flow j -> i.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-8
DEFAULT_LAYERS = "last"  # final encoder layer only (most interpretable; configurable)


def _resolve_layers(layers, n_layers: int) -> List[int]:
    if layers is None or layers == "all":
        return list(range(n_layers))
    if layers == "last":          # final encoder layer only (default)
        return [n_layers - 1]
    if layers == "last_two":
        return list(range(max(0, n_layers - 2), n_layers))
    if layers == "last_three":
        return list(range(max(0, n_layers - 3), n_layers))
    return [int(l) for l in layers]


def per_sample_label_label(
    attn_list: Sequence,
    n_labels: int,
    layers=DEFAULT_LAYERS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce a batch's attention to per-sample absolute and conditional
    label-to-label matrices.

    Returns ``(A, P)`` each ``(B, L, L)``:
      A : absolute label-to-label attention, mean over heads and selected layers.
          The diagonal (label self-attention) is **preserved** -- it is real
          information now that there is no diagonal mask.
      P : conditional preference P[i,j] = A[i,j] / (M_i + eps), where M_i is the
          *off-diagonal* mass (self-attention excluded from the denominator), so
          the off-diagonal P is the cross-label preference. P[i,i] = self/M_i is
          retained but not used by the directed effect (which skips i==j).
    """
    import torch
    sel = _resolve_layers(layers, len(attn_list))
    # mean over heads, then over selected layers -> (B, L, L)
    blocks = [attn_list[l][:, :, :n_labels, :n_labels].mean(dim=1) for l in sel]
    A = torch.stack(blocks, dim=0).mean(dim=0).clone()  # (B, L, L), diagonal kept
    idx = torch.arange(n_labels)
    diag = A[:, idx, idx].clone()                       # (B, L) self-attention
    M = (A.sum(dim=-1) - diag).unsqueeze(-1)            # (B, L, 1) off-diagonal mass
    P = A / (M + EPS)
    return A.cpu().numpy(), P.cpu().numpy()


def collect_label_label(
    model, loader, n_labels: int, device: str = "cpu", layers=DEFAULT_LAYERS,
) -> Dict[str, np.ndarray]:
    """
    Collect per-sample label-to-label matrices and labels across a loader.

    Returns ``{"A": (N,L,L), "P": (N,L,L), "Y": (N,L)}``.
    """
    from inference import iter_attention
    A_all, P_all, Y_all = [], [], []
    for _, targets, _, attn in iter_attention(model, loader, device):
        A, P = per_sample_label_label(attn, n_labels, layers)
        A_all.append(A); P_all.append(P)
        Y_all.append(np.asarray(targets))
    return {
        "A": np.concatenate(A_all), "P": np.concatenate(P_all),
        "Y": np.concatenate(Y_all).astype(int),
    }


def collect_label_label_multi(
    model, loaders, n_labels: int, device: str = "cpu", layers=DEFAULT_LAYERS,
) -> Dict[str, np.ndarray]:
    """
    Collect per-sample label-to-label matrices across **several** loaders and
    concatenate. Use this to pool train+val+test for the effect estimate when the
    dataset is limited. ``loaders`` is a list or dict of DataLoaders.

    Returns ``{"A": (N,L,L), "P": (N,L,L), "Y": (N,L)}`` over all samples.
    """
    iterable = loaders.values() if isinstance(loaders, dict) else loaders
    A_all, P_all, Y_all = [], [], []
    for loader in iterable:
        d = collect_label_label(model, loader, n_labels, device, layers)
        A_all.append(d["A"]); P_all.append(d["P"]); Y_all.append(d["Y"])
    return {"A": np.concatenate(A_all), "P": np.concatenate(P_all),
            "Y": np.concatenate(Y_all).astype(int)}


def naive_mean_attention(P: np.ndarray) -> np.ndarray:
    """Diagnostic only: naive mean of P over all samples (dominant-label biased)."""
    return P.mean(axis=0)


def mean_label_label_attention(A: np.ndarray) -> np.ndarray:
    """Dataset-mean raw label->label attention matrix (L, L) for the magnitude plot."""
    return A.mean(axis=0)


def top_source_labels(P_mean: np.ndarray, k: int = 2) -> List[int]:
    """
    Identify the top-k attention *hub* source labels: the columns with the highest
    mean conditional preference (i.e. the label tokens that other tokens attend to
    most), excluding the diagonal. Returns label indices, highest first. Use this
    to drive attention-informed ablations instead of hard-coding hub labels.
    """
    P = np.array(P_mean, dtype=float)
    np.fill_diagonal(P, np.nan)
    col_mean = np.nanmean(P, axis=0)          # mean over receivers for each source j
    order = np.argsort(col_mean)[::-1]
    return [int(i) for i in order[:k]]


def generate_split_label_label_pdfs(
    model, loader, split: str, label_names: Sequence[str],
    out_root, device: str = "cpu", layers=DEFAULT_LAYERS,
    max_samples: Optional[int] = None, save_full: bool = True,
    save_full_raw: bool = False,
):
    """
    Per-sample label->label (10x10) and full S x S attention PDFs + NPZ, written
    alongside the label->image outputs. Bounded by ``max_samples`` (None == all).

    For each sample:
      * ``<sid>_label_label.pdf`` : 10x10 label->label (mean heads & selected layers),
      * ``<sid>_full_attention.pdf`` : full S x S map (mean heads of selected layers),
      * ``data/<sid>.npz`` : per-sample A, P, per-layer mean-over-heads maps, and the
        raw per-head tensors when ``save_full_raw`` is set.

    Directory: ``<out_root>/<split>/per_sample/{pdf,data}/``.
    """
    import torch
    from pathlib import Path
    from inference import iter_attention, dataset_sample_ids
    import label_label_plots as llp

    out_root = Path(out_root)
    pdf_dir = out_root / split / "per_sample" / "pdf"
    data_dir = out_root / split / "per_sample" / "data"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    sids = dataset_sample_ids(loader)

    cursor, written = 0, 0
    for _, targets, probs, attn in iter_attention(model, loader, device):
        A, P = per_sample_label_label(attn, len(label_names), layers)  # (B,10,10)
        sel = _resolve_layers(layers, len(attn))
        # mean over heads per layer -> (B, n_layers, S, S)
        per_layer = torch.stack([a.mean(dim=1) for a in attn], dim=1)
        B = A.shape[0]
        for b in range(B):
            gidx = cursor + b
            if max_samples is not None and gidx >= max_samples:
                return {"written": written}
            sid = sids[gidx] if gidx < len(sids) else f"{split}_{gidx}"
            llp.plot_sample_label_label(A[b], label_names, pdf_dir / f"{sid}_label_label", sid)
            if save_full:
                M59 = per_layer[b, sel].mean(dim=0).cpu().numpy()  # mean over selected layers
                llp.plot_full_attention_map(M59, len(label_names), pdf_dir / f"{sid}_full_attention", sid)
            npz = {
                "label_label_A": A[b].astype(np.float32),
                "label_label_P": P[b].astype(np.float32),
                "full_attention_meanheads_per_layer": per_layer[b].cpu().numpy().astype(np.float32),
            }
            if save_full_raw:
                npz["full_attention_per_head"] = torch.stack([a[b] for a in attn]).cpu().numpy().astype(np.float32)
            np.savez_compressed(data_dir / f"{sid}.npz", **npz)
            written += 1
        cursor += B
    return {"written": written}


def balanced_directed_effect(
    P: np.ndarray, Y: np.ndarray, min_support: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Imbalance-balanced directed effect delta[i, j] and its support.

    Returns a dict of (L, L) arrays:
      delta        : E[P|y_i=1,y_j=1] - E[P|y_i=1,y_j=0] (NaN if unsupported)
      mean_present : E[P[i,j] | y_i=1, y_j=1]
      mean_absent  : E[P[i,j] | y_i=1, y_j=0]
      n_present    : sample count of the present stratum
      n_absent     : sample count of the absent stratum
      reliable     : boolean, both strata >= min_support
    """
    N, L, _ = P.shape
    delta = np.full((L, L), np.nan)
    mean_present = np.full((L, L), np.nan)
    mean_absent = np.full((L, L), np.nan)
    n_present = np.zeros((L, L), dtype=int)
    n_absent = np.zeros((L, L), dtype=int)
    reliable = np.zeros((L, L), dtype=bool)

    for i in range(L):
        yi1 = Y[:, i] == 1
        for j in range(L):
            if i == j:
                continue
            present = yi1 & (Y[:, j] == 1)
            absent = yi1 & (Y[:, j] == 0)
            n_p, n_a = int(present.sum()), int(absent.sum())
            n_present[i, j], n_absent[i, j] = n_p, n_a
            if n_p > 0:
                mean_present[i, j] = float(P[present, i, j].mean())
            if n_a > 0:
                mean_absent[i, j] = float(P[absent, i, j].mean())
            if n_p >= min_support and n_a >= min_support:
                reliable[i, j] = True
                delta[i, j] = mean_present[i, j] - mean_absent[i, j]
    return {
        "delta": delta, "mean_present": mean_present, "mean_absent": mean_absent,
        "n_present": n_present, "n_absent": n_absent, "reliable": reliable,
    }


def absolute_label_label_mass(A: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Per-query absolute off-diagonal label-to-label mass M_i, averaged over the
    samples where y_i = 1 (used for node/edge weighting). Returns (L,).
    """
    L = A.shape[1]
    out = np.zeros(L)
    for i in range(L):
        sel = Y[:, i] == 1
        if sel.any():
            out[i] = float(A[sel, i, :].sum(axis=-1).mean())
    return out


def layerwise_sensitivity(
    model, loader, n_labels: int, n_layers: int, device: str = "cpu",
    min_support: int = 10,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Per-layer balanced directed-effect tables (each layer analyzed on its own),
    for the layer-wise sensitivity analysis. Returns ``{layer_idx: effect_dict}``.
    """
    out = {}
    for l in range(n_layers):
        data = collect_label_label(model, loader, n_labels, device, layers=[l])
        out[l] = balanced_directed_effect(data["P"], data["Y"], min_support)
    return out
