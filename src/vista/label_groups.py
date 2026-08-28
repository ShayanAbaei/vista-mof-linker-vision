"""
Label prevalence and rare/intermediate/dominant grouping (spec section 4).

Groups are derived **programmatically from training prevalence only**:

    rare         : prevalence < 0.20
    intermediate : 0.20 <= prevalence <= 0.70
    dominant     : prevalence > 0.70

With this project's distribution the dominant group resolves to aromaticity,
H-bond acceptors and conjugation, and the rare group to hydrophobicity,
flexibility, H-bond donors and macrocycles -- but nothing is hard-coded.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

RARE_MAX = 0.20      # prevalence < 0.20 -> rare
DOMINANT_MIN = 0.70  # prevalence > 0.70 -> dominant


def compute_prevalence(labels: np.ndarray) -> np.ndarray:
    """Per-label positive prevalence from a binary (N, L) array."""
    return np.asarray(labels, dtype=float).mean(axis=0)


def assign_group(prevalence: float) -> str:
    if prevalence < RARE_MAX:
        return "rare"
    if prevalence > DOMINANT_MIN:
        return "dominant"
    return "intermediate"


@dataclass
class LabelGroups:
    label_names: List[str]
    prevalence: np.ndarray             # training prevalence per label
    groups: List[str]                  # 'rare'|'intermediate'|'dominant' per label

    @property
    def rare_idx(self) -> List[int]:
        return [i for i, g in enumerate(self.groups) if g == "rare"]

    @property
    def intermediate_idx(self) -> List[int]:
        return [i for i, g in enumerate(self.groups) if g == "intermediate"]

    @property
    def dominant_idx(self) -> List[int]:
        return [i for i, g in enumerate(self.groups) if g == "dominant"]

    def names_for(self, group: str) -> List[str]:
        return [self.label_names[i] for i, g in enumerate(self.groups) if g == group]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "label": self.label_names,
            "train_prevalence": self.prevalence,
            "group": self.groups,
        })


def build_label_groups(train_labels: np.ndarray, label_names: Sequence[str]) -> LabelGroups:
    """Build :class:`LabelGroups` from the training label matrix only."""
    prev = compute_prevalence(train_labels)
    groups = [assign_group(float(p)) for p in prev]
    return LabelGroups(list(label_names), prev, groups)


def split_prevalence_table(
    split_labels: Dict[str, np.ndarray],
    label_names: Sequence[str],
) -> pd.DataFrame:
    """
    Long-form per-split prevalence/count table for every label.
    ``split_labels`` maps split name -> (N, L) binary array.
    """
    rows = []
    for split, arr in split_labels.items():
        arr = np.asarray(arr)
        n = arr.shape[0]
        pos = arr.sum(axis=0).astype(int)
        for i, name in enumerate(label_names):
            rows.append({
                "split": split, "label": name,
                "positive_count": int(pos[i]), "total_count": int(n),
                "negative_count": int(n - pos[i]),
                "prevalence": float(pos[i] / n) if n else 0.0,
                "percentage": float(100.0 * pos[i] / n) if n else 0.0,
            })
    return pd.DataFrame(rows)


def plot_label_distribution(
    split_labels: Dict[str, np.ndarray],
    label_names: Sequence[str],
    groups: LabelGroups,
    out_base: Union[str, Path],
    log_panel: bool = True,
):
    """
    Label-distribution figure: per-split positive prevalence (grouped bars) and an
    optional log-count panel. Professional per-split coloring (no rare/intermediate/
    dominant color encoding). Saves PDF + the long-form table (CSV) + metadata.
    """
    from plotting import academic_rc, save_figure, SPLIT_COLORS, PALETTE, DOUBLE_COL_W
    import matplotlib.pyplot as plt

    table = split_prevalence_table(split_labels, label_names)
    splits = list(split_labels.keys())
    L = len(label_names)
    x = np.arange(L)
    width = 0.8 / max(1, len(splits))
    colors = [SPLIT_COLORS.get(s, PALETTE[i % len(PALETTE)]) for i, s in enumerate(splits)]

    out_base = Path(out_base)
    with academic_rc():
        n_panels = 2 if log_panel else 1
        fig, axes = plt.subplots(n_panels, 1, figsize=(DOUBLE_COL_W, 2.7 * n_panels),
                                 constrained_layout=True, squeeze=False)
        ax = axes[0, 0]
        for s_i, split in enumerate(splits):
            pct = table[table.split == split].set_index("label").loc[list(label_names), "percentage"].values
            ax.bar(x + s_i * width - 0.4 + width / 2, pct, width, label=split.capitalize(),
                   color=colors[s_i], edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(label_names, rotation=35, ha="right")
        ax.set_ylabel("Positive prevalence (%)")
        ax.set_title("Per-label prevalence by split")
        ax.legend(loc="upper right", ncol=len(splits))

        if log_panel:
            ax2 = axes[1, 0]
            for s_i, split in enumerate(splits):
                cnt = table[table.split == split].set_index("label").loc[list(label_names), "positive_count"].values
                ax2.bar(x + s_i * width - 0.4 + width / 2, np.maximum(cnt, 0.1), width,
                        label=split.capitalize(), color=colors[s_i], edgecolor="white", linewidth=0.4)
            ax2.set_yscale("log")
            ax2.set_xticks(x)
            ax2.set_xticklabels(label_names, rotation=35, ha="right")
            ax2.set_ylabel("Positive count")

        written = save_figure(
            fig, out_base,
            data={"table_csv": str(out_base.with_name(out_base.name + "_table.csv"))},
            metadata={"rare_max": RARE_MAX, "dominant_min": DOMINANT_MIN,
                      "splits": splits, "groups": dict(zip(label_names, groups.groups))},
        )
    table.to_csv(out_base.with_name(out_base.name + "_table.csv"), index=False)
    return written, table
