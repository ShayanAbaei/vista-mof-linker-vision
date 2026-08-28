"""
Comparison publication plots (spec section 12, plots 7-9):

  * performance-change matrix for dominant-label removal/masking,
  * forest / point-range plot of rare-label performance changes with uncertainty,
  * baseline-comparison point-range plot.

All center diverging scales at zero, show uncertainty, use colorblind-safe
colors, and save the plotted data alongside the PDF.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd


def plot_change_matrix(
    change_df: pd.DataFrame, out_base: Union[str, Path],
    title: str = "Rare-label test AUPRC change under dominant-source intervention",
    cbar_label: str = "AUPRC change (ablated - full)",
):
    """
    Diverging heatmap of a (rare target x dominant source) change matrix, centered
    at zero with a symmetric robust range. Negative => the rare target benefited
    from that source's supervision/communication path.
    """
    from plotting import academic_rc, save_figure, symmetric_robust_limit, DIVERGING_CMAP, SINGLE_COL_W
    import matplotlib.pyplot as plt

    mat = change_df.values.astype(float)
    lim = symmetric_robust_limit(mat, 99.0)
    out_base = Path(out_base)
    with academic_rc():
        fig, ax = plt.subplots(figsize=(SINGLE_COL_W * 1.7, SINGLE_COL_W * 1.4),
                               constrained_layout=True)
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=DIVERGING_CMAP, vmin=-lim, vmax=lim)
        ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(change_df.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(change_df.index, fontsize=6)
        for (i, j), v in np.ndenumerate(mat):
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=5,
                        color="black" if abs(v) < 0.6 * lim else "white")
        ax.set_xlabel("removed / masked dominant source")
        ax.set_ylabel("rare target label")
        ax.set_title(title, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, label=cbar_label)
        written = save_figure(fig, out_base, data={"matrix": mat},
                              metadata={"rows": list(change_df.index),
                                        "cols": list(change_df.columns), "robust_limit": lim})
    return written


def plot_rare_change_forest(
    changes: pd.DataFrame, out_base: Union[str, Path],
    value_col: str = "delta_mean", low_col: str = "delta_ci_low",
    high_col: str = "delta_ci_high", title: str = "Rare-label AUPRC change (mean +/- 95% CI)",
):
    """
    Forest / point-range plot of per-rare-label performance change with CIs.
    ``changes`` is indexed by label with mean/CI columns.
    """
    from plotting import academic_rc, save_figure, SINGLE_COL_W, PRIMARY
    import matplotlib.pyplot as plt

    labels = list(changes.index)
    y = np.arange(len(labels))
    mean = changes[value_col].values
    lo = changes[low_col].values
    hi = changes[high_col].values
    out_base = Path(out_base)
    with academic_rc():
        fig, ax = plt.subplots(figsize=(SINGLE_COL_W * 1.8, 0.4 * len(labels) + 1.2),
                               constrained_layout=True)
        ax.axvline(0.0, color="0.6", lw=0.8, ls="--")
        ax.errorbar(mean, y, xerr=[mean - lo, hi - mean], fmt="o", color=PRIMARY,
                    ecolor="0.4", capsize=2, ms=4, lw=1)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("AUPRC change vs full model")
        ax.set_title(title, fontsize=8)
        written = save_figure(fig, out_base,
                              data={"label": labels, "mean": mean, "ci_low": lo, "ci_high": hi})
    return written


def plot_baseline_comparison(
    long_df: pd.DataFrame, out_base: Union[str, Path], metric: str = "auprc",
    title: Optional[str] = None,
):
    """
    Point-range comparison of a per-label ``metric`` across models with uncertainty.

    ``long_df`` columns: ``label``, ``model``, ``<metric>_mean``, ``<metric>_ci_low``,
    ``<metric>_ci_high`` (as produced by ``baselines.aggregate_baseline_metrics``).
    """
    from plotting import academic_rc, save_figure, PALETTE, DOUBLE_COL_W
    import matplotlib.pyplot as plt

    labels = list(dict.fromkeys(long_df["label"]))
    models = list(dict.fromkeys(long_df["model"]))
    y = np.arange(len(labels))
    out_base = Path(out_base)
    mean_c = f"{metric}_mean"; lo_c = f"{metric}_ci_low"; hi_c = f"{metric}_ci_high"
    title = title or f"Per-label {metric.upper()} by model (mean ± 95% CI)"

    with academic_rc():
        fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 0.45 * len(labels) + 1.2),
                               constrained_layout=True)
        offsets = np.linspace(-0.28, 0.28, max(1, len(models)))
        for mi, model in enumerate(models):
            sub = long_df[long_df["model"] == model].set_index("label").reindex(labels)
            mean = sub[mean_c].values
            lo = sub.get(lo_c, sub[mean_c]).values
            hi = sub.get(hi_c, sub[mean_c]).values
            ax.errorbar(mean, y + offsets[mi], xerr=[mean - lo, hi - mean], fmt="o",
                        color=PALETTE[mi % len(PALETTE)], ecolor="0.5",
                        capsize=2, ms=4, lw=1, label=model)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(metric.upper())
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=7)
        written = save_figure(fig, out_base, metadata={"metric": metric, "models": models})
    return written
