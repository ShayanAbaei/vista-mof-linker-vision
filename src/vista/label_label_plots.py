"""
Publication-quality label-to-label attention heatmaps (replaces the removed
Bayesian-graph figures).

Three aggregate 10x10 heatmaps, each a vector PDF + data sidecar:

  * mean label->label attention            (sequential)
  * mean conditional label preference P    (sequential)
  * imbalance-corrected directed effect δ  (diverging, centered at 0)

Direction convention (see ``masking``): cell ``[i, j]`` is query/receiver label
``i`` attending key/source label ``j`` (information flow ``j -> i``). Rows =
receiver i, columns = source j.

The effect heatmap is the corrected version: no credible-star markers, no hatched
"unsupported" cells (cells that are undefined — the diagonal, or a stratum with
zero samples — are simply left blank/neutral), and it is intended to be computed
on the **whole dataset** (train+val+test) because the data is limited.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np


def _format_value(v: float, fmt: str) -> str:
    return "" if not np.isfinite(v) else format(v, fmt)


def plot_matrix_heatmap(
    matrix: np.ndarray,
    label_names: Sequence[str],
    out_base: Union[str, Path],
    cmap: str = "viridis",
    title: str = "",
    cbar_label: str = "",
    center_zero: bool = False,
    mask_diagonal: bool = True,
    annotate: bool = True,
    annot_fmt: str = ".2f",
    annot_min_frac: float = 0.0,
    robust_q: float = 99.0,
):
    """
    Render one 10x10 label-to-label heatmap (rows=receiver i, cols=source j).

    Parameters
    ----------
    center_zero : bool
        If True use a symmetric diverging scale centered at zero (for effects);
        else a sequential scale from 0 (or data min) to a robust max.
    mask_diagonal : bool
        Blank the i==j cells (self-attention) so off-diagonal structure is visible.
    annotate : bool
        Write the numeric value in each cell.
    """
    from plotting import (academic_rc, save_figure, symmetric_robust_limit,
                          DIVERGING_CMAP, SINGLE_COL_W)
    import matplotlib.pyplot as plt

    M = np.array(matrix, dtype=float).copy()
    L = len(label_names)
    if mask_diagonal:
        np.fill_diagonal(M, np.nan)

    out_base = Path(out_base)
    with academic_rc():
        fig, ax = plt.subplots(figsize=(SINGLE_COL_W * 1.75, SINGLE_COL_W * 1.55),
                               constrained_layout=True)
        cmap_obj = plt.get_cmap(DIVERGING_CMAP if center_zero else cmap).copy()
        cmap_obj.set_bad("#F2F2F2")  # neutral light gray for undefined cells (no hatch)
        disp = np.ma.masked_invalid(M)
        if center_zero:
            lim = symmetric_robust_limit(M, robust_q)
            im = ax.imshow(disp, cmap=cmap_obj, vmin=-lim, vmax=lim)
        else:
            finite = M[np.isfinite(M)]
            vmax = float(np.percentile(finite, robust_q)) if finite.size else 1.0
            im = ax.imshow(disp, cmap=cmap_obj, vmin=0.0, vmax=max(vmax, 1e-9))

        if annotate:
            vmax = im.norm.vmax
            annot_min = annot_min_frac * vmax              # below this, leave blank (declutter)
            light_thr = vmax * 0.55
            for i in range(L):
                for j in range(L):
                    v = M[i, j]
                    if not np.isfinite(v) or abs(v) < annot_min:
                        continue
                    if center_zero:
                        txtcolor = "white" if abs(v) > vmax * 0.6 else "black"
                    else:
                        txtcolor = "white" if v > light_thr else "black"
                    s = _format_value(v, annot_fmt)
                    if s in ("-0.00", "+0.00", "-0.0", "+0.0"):
                        s = "0.00"
                    ax.text(j, i, s, ha="center", va="center", fontsize=5.5, color=txtcolor)

        ax.set_xticks(range(L)); ax.set_xticklabels(label_names, rotation=90, fontsize=6)
        ax.set_yticks(range(L)); ax.set_yticklabels(label_names, fontsize=6)
        ax.set_xlabel("Source label $j$ (key)")
        ax.set_ylabel("Receiver label $i$ (query)")
        if title:
            ax.set_title(title)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        if cbar_label:
            cb.set_label(cbar_label)
        written = save_figure(fig, out_base, data={"matrix": np.array(matrix, dtype=float)},
                              metadata={"labels": list(label_names), "center_zero": center_zero,
                                        "mask_diagonal": mask_diagonal})
    return written


def plot_label_label_effect(
    effect: Dict[str, np.ndarray],
    label_names: Sequence[str],
    out_base: Union[str, Path],
    title: str = "Imbalance-corrected directed label-attention effect",
    cbar_label: str = r"$\Delta$ conditional attention preference",
):
    """
    Corrected effect heatmap from ``attention_label_label.balanced_directed_effect``:
    diverging, centered at zero, diagonal blank, no stars, no hatching. Undefined
    cells (zero-support strata) are left neutral. Compute ``effect`` on the whole
    dataset (train+val+test) for maximum support.
    """
    return plot_matrix_heatmap(
        effect["delta"], label_names, out_base, title=title, cbar_label=cbar_label,
        center_zero=True, mask_diagonal=True, annotate=True, annot_fmt="+.2f",
        annot_min_frac=0.12,   # only annotate notable cells (declutter near-zero)
    )


def plot_sample_label_label(A: np.ndarray, label_names, out_base, sample_id,
                            cbar_label="Attention weight"):
    """
    Per-sample 10x10 label->label attention heatmap (mean over heads & selected
    layers, diagonal blank so cross-label structure is visible). Saves PDF + data.
    The unaveraged self-attention diagonal is preserved in the NPZ sidecar.
    """
    return plot_matrix_heatmap(
        A, label_names, out_base, cmap="viridis",
        title=f"Label-to-label attention | sample {sample_id}",
        cbar_label=cbar_label, center_zero=False, mask_diagonal=False,
        annotate=True, annot_fmt=".2f")


def plot_full_attention_map(M: np.ndarray, n_labels: int, out_base, sample_id,
                            cbar_label="Attention weight"):
    """
    Per-sample full S x S attention map (here 59x59 = 10 label + 49 image tokens),
    mean over heads of the selected layers, with the four blocks delineated. Makes
    visible how little of each label token's budget goes to other labels vs. the
    49 image patches. Saves PDF + the matrix.
    """
    from plotting import academic_rc, save_figure, SEQUENTIAL_CMAP, SINGLE_COL_W
    import matplotlib.pyplot as plt

    M = np.asarray(M, dtype=float)
    S = M.shape[-1]
    pos = M[M > 0]
    vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
    out_base = Path(out_base)
    with academic_rc():
        fig, ax = plt.subplots(figsize=(SINGLE_COL_W * 1.9, SINGLE_COL_W * 1.7),
                               constrained_layout=True)
        im = ax.imshow(M, cmap=SEQUENTIAL_CMAP, vmin=0.0, vmax=max(vmax, 1e-9))
        # delineate the label (0..n_labels-1) vs image (n_labels..) blocks
        ax.axhline(n_labels - 0.5, color="white", lw=0.8)
        ax.axvline(n_labels - 0.5, color="white", lw=0.8)
        ax.set_xlabel(f"key token $j$  (0-{n_labels-1}: labels, {n_labels}-{S-1}: image patches)")
        ax.set_ylabel("query token $i$")
        ax.set_title(f"Full {S}x{S} attention | sample {sample_id}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=cbar_label)
        return save_figure(fig, out_base, data={"attention": M},
                           metadata={"sample_id": str(sample_id), "n_labels": int(n_labels),
                                     "seq_len": int(S)})


def plot_mean_attention(A_mean: np.ndarray, label_names, out_base,
                        title="Mean label-to-label attention",
                        cbar_label="Attention weight"):
    """Sequential heatmap of the dataset-mean label->label attention. The diagonal
    (label self-attention) is shown -- it is real information (no diagonal mask)."""
    return plot_matrix_heatmap(A_mean, label_names, out_base, cmap="viridis",
                               title=title, cbar_label=cbar_label, center_zero=False,
                               mask_diagonal=False, annotate=True, annot_fmt=".2f")


def plot_conditional_preference(P_mean: np.ndarray, label_names, out_base,
                                title="Mean conditional label preference",
                                cbar_label=r"$P[i,j]$ (off-diagonal share)"):
    """Sequential heatmap of the dataset-mean conditional preference P[i,j]."""
    return plot_matrix_heatmap(P_mean, label_names, out_base, cmap="magma",
                               title=title, cbar_label=cbar_label, center_zero=False,
                               mask_diagonal=True, annotate=True, annot_fmt=".2f")
