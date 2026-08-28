from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

LABEL_COLUMNS = [
    "hydrophobicity", "polarity", "flexibility", "aromaticity",
    "h_bond_donors", "h_bond_acceptors", "planarity_single_ring",
    "macrocycle_gt6_rings", "charge_present", "conjugation_present",
]

LABEL_DISPLAY = {
    "hydrophobicity": "Hydrophobicity",
    "polarity": "Polarity",
    "flexibility": "Flexibility",
    "aromaticity": "Aromaticity",
    "h_bond_donors": "Hydrogen-Bond Donors",
    "h_bond_acceptors": "Hydrogen-Bond Acceptors",
    "planarity_single_ring": "Single-Ring Proxy",
    "macrocycle_gt6_rings": "Ring-Rich (>6 Rings) Proxy",
    "charge_present": "Charge Present",
    "conjugation_present": "Conjugation Present",
}

EXPECTED_N = 9414
EXPECTED_SPLIT = {"train": 6590, "validation": 1412, "test": 1412}
EXPECTED_UNIQUE_CANONICAL = 5727


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("vista_phase_c")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    raise TypeError(type(x).__name__)


def first_linker_from_cell(cell: str) -> str:
    obj = ast.literal_eval(cell)
    if not isinstance(obj, (list, tuple)) or not obj:
        raise ValueError(f"Expected non-empty linker list, got: {cell!r}")
    return str(obj[0])


def normalize_manifest(df: pd.DataFrame, master: pd.DataFrame, scheme: str) -> pd.DataFrame:
    fn_col = "filename" if "filename" in df.columns else "image_name" if "image_name" in df.columns else None
    if fn_col is None:
        raise ValueError(f"{scheme} manifest has no filename/image_name column")
    out = df.copy()
    out["filename"] = out[fn_col].astype(str)
    if out["filename"].duplicated().any():
        raise ValueError(f"{scheme} manifest has duplicate filenames")
    if set(out["split"].astype(str)) != {"train", "validation", "test"}:
        raise ValueError(f"{scheme} manifest split values are not train/validation/test")
    counts = out["split"].value_counts().to_dict()
    if counts != EXPECTED_SPLIT:
        raise ValueError(f"{scheme} split counts {counts} != {EXPECTED_SPLIT}")

    mcols = ["image_name", "qmof_id", "canonical_derived", "first_linker_smiles"] + LABEL_COLUMNS
    meta = master[mcols].rename(columns={"image_name": "filename"})
    # Avoid collisions: frozen manifest values remain authoritative, but master fills missing fields.
    need = [c for c in mcols[1:] if c not in out.columns]
    if need:
        out = out.merge(meta[["filename"] + need], on="filename", how="left", validate="one_to_one")
    else:
        # still attach derived canonical under a private comparison name
        out = out.merge(meta[["filename", "canonical_derived", "first_linker_smiles"]], on="filename", how="left", validate="one_to_one", suffixes=("", "_master"))
    if out["qmof_id"].isna().any() if "qmof_id" in out.columns else False:
        raise ValueError(f"{scheme} manifest could not be fully matched to master metadata")
    if len(out) != EXPECTED_N:
        raise ValueError(f"{scheme} manifest has {len(out)} rows, expected {EXPECTED_N}")

    if "canonical_first_linker_smiles" in out.columns:
        out["canonical_group"] = out["canonical_first_linker_smiles"].astype(str)
        # The frozen manifest is authoritative, but Phase C independently
        # canonicalizes the first linker from the master file. These must agree.
        derived_col = "canonical_derived" if "canonical_derived" in out.columns else "canonical_derived_master"
        if derived_col in out.columns:
            mismatch = (out["canonical_group"].astype(str) != out[derived_col].astype(str))
            if mismatch.any():
                raise ValueError(
                    f"{scheme} manifest canonical key disagrees with independently "
                    f"canonicalized first linker for {int(mismatch.sum())} rows"
                )
    elif "canonical_derived" in out.columns:
        out["canonical_group"] = out["canonical_derived"].astype(str)
    else:
        out["canonical_group"] = out["canonical_derived_master"].astype(str)
    return out


def threshold_by_f1(y: np.ndarray, p: np.ndarray, n_grid: int = 200) -> Tuple[float, float]:
    from sklearn.metrics import f1_score
    y = np.asarray(y, int); p = np.asarray(p, float)
    if y.sum() == 0:
        return 0.5, 0.0
    grid = np.linspace(0.001, 0.999, n_grid)
    best_t, best_s = 0.5, -1.0
    for t in grid:
        s = f1_score(y, (p >= t).astype(int), zero_division=0)
        if s > best_s:  # first max -> lower threshold wins
            best_t, best_s = float(t), float(s)
    return best_t, best_s


def metrics_from_probs(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray, label_names: Sequence[str]):
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, accuracy_score,
        balanced_accuracy_score, average_precision_score, roc_auc_score,
    )
    y = np.asarray(y, int); p = np.asarray(p, float)
    thresholds = np.asarray(thresholds, float)
    pred = (p >= thresholds.reshape(1, -1)).astype(int)
    per = []
    for j, label in enumerate(label_names):
        yt, yp, ps = y[:, j], pred[:, j], p[:, j]
        pos = int(yt.sum()); neg = int(len(yt)-pos)
        auprc = float(average_precision_score(yt, ps)) if pos else float("nan")
        auroc = float(roc_auc_score(yt, ps)) if pos and neg else float("nan")
        ba = float(balanced_accuracy_score(yt, yp)) if pos and neg else float("nan")
        tn = int(((yt==0)&(yp==0)).sum()); fp = int(((yt==0)&(yp==1)).sum())
        spec = tn/(tn+fp) if (tn+fp) else float("nan")
        per.append({
            "label": label,
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "accuracy": float(accuracy_score(yt, yp)),
            "balanced_accuracy": ba,
            "specificity": spec,
            "auprc": auprc,
            "auroc": auroc,
            "pos_support": pos,
            "neg_support": neg,
            "threshold": float(thresholds[j]),
        })
    per_df = pd.DataFrame(per)
    flat_y, flat_p = y.ravel(), pred.ravel()
    overall = {
        "macro_precision": float(per_df["precision"].mean()),
        "macro_recall": float(per_df["recall"].mean()),
        "macro_f1": float(per_df["f1"].mean()),
        "macro_accuracy": float(per_df["accuracy"].mean()),
        "macro_balanced_accuracy": float(per_df["balanced_accuracy"].mean()),
        "macro_specificity": float(per_df["specificity"].mean()),
        "macro_auprc": float(per_df["auprc"].mean()),
        "macro_auroc": float(per_df["auroc"].mean()),
        "micro_precision": float(precision_score(y, pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y, pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "micro_auprc": float(average_precision_score(y, p, average="micro")),
        "subset_accuracy": float(accuracy_score(y, pred)),
        "hamming_loss": float(np.mean(y != pred)),
    }
    return per_df, overall, pred


def macro_auprc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    vals=[]
    for j in range(y.shape[1]):
        if np.asarray(y[:,j]).sum() == 0: continue
        vals.append(average_precision_score(y[:,j], p[:,j]))
    return float(np.mean(vals))
