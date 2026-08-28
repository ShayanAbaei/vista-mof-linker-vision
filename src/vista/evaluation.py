"""
Evaluation layer: thresholds, prediction tables, metric tables, confusion
matrices (spec sections 5 and 11).

Threshold protocol (no universal 0.5):
    1. select per-label thresholds on the **validation** split only,
    2. optimizing validation F1 (or a configurable objective),
    3. freeze them,
    4. apply unchanged to the test split,
    5. persist to CSV (and the checkpoint via the trainer).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, brier_score_loss


# --------------------------------------------------------------------------- #
# Threshold selection
# --------------------------------------------------------------------------- #
def _objective_fn(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    if name == "f1":
        return lambda yt, yp: f1_score(yt, yp, zero_division=0)
    if name == "balanced_accuracy":
        return lambda yt, yp: balanced_accuracy_score(yt, yp) if len(np.unique(yt)) > 1 else 0.0
    if name == "youden":  # sensitivity + specificity - 1
        def y(yt, yp):
            tp = np.sum((yt == 1) & (yp == 1)); fn = np.sum((yt == 1) & (yp == 0))
            tn = np.sum((yt == 0) & (yp == 0)); fp = np.sum((yt == 0) & (yp == 1))
            sens = tp / (tp + fn) if (tp + fn) else 0.0
            spec = tn / (tn + fp) if (tn + fp) else 0.0
            return sens + spec - 1.0
        return y
    raise ValueError(f"Unknown threshold objective {name!r}")


def select_thresholds(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    label_names: Sequence[str],
    objective: str = "f1",
    n_grid: int = 200,
) -> Dict[str, np.ndarray]:
    """
    Pick a per-label threshold that maximizes ``objective`` on validation data.

    Returns a dict with ``thresholds`` (L,), ``scores`` (L,) and ``objective``.
    Candidate thresholds are a dense grid in (0, 1); ties resolve to the lower
    threshold (favouring recall on rare labels).
    """
    obj = _objective_fn(objective)
    L = len(label_names)
    thresholds = np.full(L, 0.5, dtype=float)
    scores = np.zeros(L, dtype=float)
    grid = np.linspace(1e-3, 1 - 1e-3, n_grid)

    for i in range(L):
        yt = val_targets[:, i].astype(int)
        ps = val_probs[:, i]
        if yt.sum() == 0:  # no positives -> threshold can't be tuned meaningfully
            thresholds[i] = 0.5
            scores[i] = 0.0
            continue
        best_t, best_s = 0.5, -np.inf
        for t in grid:
            s = obj(yt, (ps > t).astype(int))
            if s > best_s:
                best_s, best_t = s, t
        thresholds[i] = float(best_t)
        scores[i] = float(best_s)
    return {"thresholds": thresholds, "scores": scores, "objective": objective}


def save_thresholds(thresholds: np.ndarray, label_names: Sequence[str],
                    out_csv: Union[str, Path], scores: Optional[np.ndarray] = None,
                    objective: str = "f1") -> pd.DataFrame:
    df = pd.DataFrame({"label": list(label_names), "threshold": np.asarray(thresholds, float)})
    if scores is not None:
        df["val_objective"] = objective
        df["val_score"] = np.asarray(scores, float)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (probs > np.asarray(thresholds).reshape(1, -1)).astype(int)


# --------------------------------------------------------------------------- #
# Prediction tables
# --------------------------------------------------------------------------- #
def build_prediction_table(
    split: str,
    sample_ids: Sequence,
    filenames: Sequence[str],
    targets: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray,
    label_names: Sequence[str],
) -> pd.DataFrame:
    """One row per sample: split, id, filename, gt/prob/pred/threshold per label."""
    preds = apply_thresholds(probs, thresholds)
    data = {"split": split, "sample_id": list(sample_ids), "filename": list(filenames)}
    for i, name in enumerate(label_names):
        data[f"gt_{name}"] = targets[:, i].astype(int)
        data[f"prob_{name}"] = probs[:, i]
        data[f"pred_{name}"] = preds[:, i]
        data[f"thr_{name}"] = float(thresholds[i])
    return pd.DataFrame(data)


def save_prediction_table(df: pd.DataFrame, out_csv: Union[str, Path]) -> None:
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


# --------------------------------------------------------------------------- #
# Full metric table (per-label + overall), with Brier
# --------------------------------------------------------------------------- #
def compute_metric_table(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray,
    label_names: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """Per-label and overall metric DataFrames using frozen per-label thresholds."""
    from train import MultiLabelMetrics
    mm = MultiLabelMetrics(label_names, threshold=np.asarray(thresholds, float))
    res = mm.compute_metrics(probs, targets)

    rows = []
    for i, name in enumerate(label_names):
        m = dict(res["per_label"][name])
        yt = targets[:, i].astype(int)
        m["brier"] = float(brier_score_loss(yt, probs[:, i])) if len(np.unique(yt)) > 1 else float("nan")
        m["label"] = name
        m["threshold"] = float(thresholds[i])
        rows.append(m)
    per_label = pd.DataFrame(rows).set_index("label")
    overall = pd.DataFrame([res["overall"]])
    return {"per_label": per_label, "overall": overall}


def save_metric_table(tables: Dict[str, pd.DataFrame], out_base: Union[str, Path]) -> Dict[str, str]:
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    pl = out_base.with_name(out_base.name + "_per_label.csv")
    ov = out_base.with_name(out_base.name + "_overall.csv")
    tables["per_label"].to_csv(pl)
    tables["overall"].to_csv(ov, index=False)
    return {"per_label": str(pl), "overall": str(ov)}


# --------------------------------------------------------------------------- #
# Confusion matrices (one 2x2 per label)
# --------------------------------------------------------------------------- #
def per_label_confusion(targets: np.ndarray, preds: np.ndarray,
                        label_names: Sequence[str]) -> pd.DataFrame:
    """Per-label TN/FP/FN/TP and row-normalized rates."""
    rows = []
    for i, name in enumerate(label_names):
        yt, yp = targets[:, i].astype(int), preds[:, i].astype(int)
        tn = int(np.sum((yt == 0) & (yp == 0))); fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0))); tp = int(np.sum((yt == 1) & (yp == 1)))
        neg = tn + fp; pos = fn + tp
        rows.append({
            "label": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "TNR": tn / neg if neg else np.nan, "FPR": fp / neg if neg else np.nan,
            "FNR": fn / pos if pos else np.nan, "TPR": tp / pos if pos else np.nan,
        })
    return pd.DataFrame(rows).set_index("label")


def plot_confusion_matrices(
    conf: pd.DataFrame,
    label_names: Sequence[str],
    out_base: Union[str, Path],
    normalized: bool = False,
    ncols: int = 5,
):
    """
    Plot a grid of per-label 2x2 confusion matrices (raw counts or row-normalized
    rates). Saves a vector PDF + the values as CSV via the figure data sidecar.
    """
    from plotting import academic_rc, save_figure, SEQUENTIAL_CMAP, DOUBLE_COL_W
    import matplotlib.pyplot as plt

    L = len(label_names)
    nrows = int(np.ceil(L / ncols))
    out_base = Path(out_base)
    with academic_rc():
        fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE_COL_W, 1.5 * nrows),
                                 constrained_layout=True, squeeze=False)
        for idx, name in enumerate(label_names):
            ax = axes[idx // ncols][idx % ncols]
            r = conf.loc[name]
            if normalized:
                mat = np.array([[r["TNR"], r["FPR"]], [r["FNR"], r["TPR"]]], float)
                fmt, vmax = "{:.2f}", 1.0
            else:
                mat = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]], float)
                fmt, vmax = "{:.0f}", mat.max() if mat.max() > 0 else 1.0
            im = ax.imshow(mat, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=vmax)
            for (yy, xx), val in np.ndenumerate(mat):
                ax.text(xx, yy, fmt.format(val), ha="center", va="center",
                        color="white" if val < 0.6 * vmax else "black", fontsize=7)
            ax.set_title(name, fontsize=7)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=6)
            ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"], fontsize=6)
        for j in range(L, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        kind = "normalized" if normalized else "raw"
        fig.suptitle(f"Per-label confusion matrices ({kind})", fontsize=10)
        written = save_figure(fig, out_base, metadata={"normalized": normalized})
    return written
