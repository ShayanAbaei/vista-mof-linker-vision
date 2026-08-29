from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

LABEL_COLUMNS = [
    "hydrophobicity","polarity","flexibility","aromaticity",
    "h_bond_donors","h_bond_acceptors","planarity_single_ring",
    "macrocycle_gt6_rings","charge_present","conjugation_present",
]
# Keep these headers byte-for-byte compatible with the existing VISTA
# prediction/metric tables. Manuscript-safe proxy names are handled at the
# presentation layer, not by changing frozen output-column names.
LABEL_DISPLAY_NAMES = [
    "Hydrophobicity","Polarity","Flexibility","Aromaticity",
    "Hydrogen-Bond Donors","Hydrogen-Bond Acceptors",
    "Planarity (Single-Ring)","Macrocycle (>6 Rings)",
    "Charge Present","Conjugation Present",
]

EXPECTED_N = 9414
EXPECTED_SPLIT_COUNTS = {"train":6590,"validation":1412,"test":1412}


def sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git","-C",str(repo),"rev-parse","HEAD"], text=True
        ).strip()
    except Exception:
        return None


def current_command() -> str:
    return " ".join(shlex.quote(x) for x in sys.argv)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def save_system(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    try:
        nvsmi = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
    except Exception as e:
        nvsmi = f"nvidia-smi unavailable: {e}\n"
    (out/"nvidia_smi.txt").write_text(nvsmi, encoding="utf-8")

    try:
        freeze = subprocess.check_output([sys.executable,"-m","pip","freeze"], text=True)
    except Exception as e:
        freeze = f"pip freeze unavailable: {e}\n"
    (out/"pip_freeze.txt").write_text(freeze, encoding="utf-8")

    env = {
        "python": sys.version,
        "executable": sys.executable,
        "hostname": os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME"),
    }
    try:
        import torch, torchvision
        env.update({
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        })
    except Exception as e:
        env["torch_error"] = repr(e)
    write_json(out/"environment.json", env)
    (out/"environment.txt").write_text(
        json.dumps(env,indent=2,default=str) + "\n\n===== nvidia-smi =====\n" +
        nvsmi + "\n===== pip freeze =====\n" + freeze,
        encoding="utf-8",
    )


def load_frozen_manifest(repo: Path, rel: str, require_disjoint: bool) -> tuple[Path,pd.DataFrame]:
    path = repo / rel
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if len(df) != EXPECTED_N:
        raise ValueError(f"{path}: {len(df)} rows != {EXPECTED_N}")
    if not df["filename"].is_unique:
        raise ValueError(f"{path}: filename is not unique")
    counts = df["split"].value_counts().to_dict()
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"{path}: split counts {counts} != {EXPECTED_SPLIT_COUNTS}")
    for c in LABEL_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"{path}: missing label {c}")
        vals = set(pd.to_numeric(df[c]).astype(int).unique())
        if not vals.issubset({0,1}):
            raise ValueError(f"{path}: non-binary {c}: {vals}")
        df[c] = pd.to_numeric(df[c]).astype(int)
    if "canonical_first_linker_smiles" not in df.columns:
        raise ValueError(f"{path}: canonical_first_linker_smiles missing")
    if require_disjoint:
        crossing = df.groupby("canonical_first_linker_smiles")["split"].nunique()
        n_cross = int((crossing > 1).sum())
        if n_cross:
            raise ValueError(f"{path}: {n_cross} canonical groups cross splits")
    return path, df


def split_csvs(manifest: pd.DataFrame, out: Path) -> Dict[str,Path]:
    d = out/"runtime_split_csvs"
    d.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in ("train","validation","test"):
        x = manifest.loc[manifest["split"]==split, ["filename"]+LABEL_COLUMNS].copy()
        p = d/f"{split}.csv"
        x.to_csv(p, index=False)
        paths[split] = p
    return paths


def enrich_predictions(pred: pd.DataFrame, manifest_split: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [c for c in [
        "filename","qmof_id","selected_linker_smiles",
        "canonical_first_linker_smiles","group_first_linker","row_index"
    ] if c in manifest_split.columns]
    meta = manifest_split[meta_cols].drop_duplicates("filename")
    return pred.merge(meta, on="filename", how="left", validate="one_to_one")


def hardlink_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except Exception:
        shutil = __import__("shutil")
        shutil.copy2(src, dst)
        return "copy"


def run_readme(out: Path, family: str, manifest: Path, manifest_hash: str, seed, extra: dict | None = None):
    extra = extra or {}
    lines = [
        f"# VISTA run: {out.name}",
        "",
        f"- model_family: `{family}`",
        f"- seed: `{seed}`",
        f"- manifest: `{manifest}`",
        f"- manifest_sha256: `{manifest_hash}`",
        f"- git_commit: `{git_commit(Path(__file__).resolve().parents[1])}`",
        "",
        "Threshold policy: validation-only per-label F1 tuning; frozen before test.",
        "Test data must not be used for model/hyperparameter selection.",
        "",
    ]
    for k,v in extra.items():
        lines.append(f"- {k}: `{v}`")
    (out/"README_RUN.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
