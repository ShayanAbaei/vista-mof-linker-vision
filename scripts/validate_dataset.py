#!/usr/bin/env python3
"""
VISTA Phase A strict data validator and freeze utility.

- Does not modify the historical VISTA folder.
- Does not extract qmof_29.zip.
- Verifies Data.csv vs ZIP member names.
- Normalizes current and canonical-first-linker-disjoint manifests.
- Verifies strict split identity disjointness.
- Generates label_definitions.yaml, image_manifest.csv, validation reports and SHA256SUMS.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_SOURCE = str(Path(__file__).resolve().parents[1] / "data" / "raw")
DEFAULT_REPO = str(Path(__file__).resolve().parents[1])

LABELS = [
    "hydrophobicity",
    "polarity",
    "flexibility",
    "aromaticity",
    "h_bond_donors",
    "h_bond_acceptors",
    "planarity_single_ring",
    "macrocycle_gt6_rings",
    "charge_present",
    "conjugation_present",
]

EXPECTED_POSITIVE_COUNTS = {
    "hydrophobicity": 1457,
    "polarity": 3639,
    "flexibility": 1399,
    "aromaticity": 6620,
    "h_bond_donors": 418,
    "h_bond_acceptors": 7659,
    "planarity_single_ring": 2613,
    "macrocycle_gt6_rings": 196,
    "charge_present": 5459,
    "conjugation_present": 7906,
}

EXPECTED_SPLIT_COUNTS = {"train": 6590, "validation": 1412, "test": 1412}

LABEL_YAML = """version: "1.0"
source: "Deterministic RDKit-derived proxy labels used by VISTA"
labels:
  hydrophobicity:
    display_name: "Hydrophobicity proxy"
    rule: "Crippen logP > 2.5"
    limitation: "Thresholded free-linker logP; not framework/solvent/guest hydrophobicity."
  polarity:
    display_name: "Polarity proxy"
    rule: "TPSA > 90"
    limitation: "Global topological polar surface-area threshold; not a local polarity map."
  flexibility:
    display_name: "Flexibility proxy"
    rule: "Rotatable bonds > 5"
    limitation: "Free-linker rotatable bonds do not equal framework flexibility."
  aromaticity:
    display_name: "Aromaticity present"
    rule: "At least one aromatic atom"
    limitation: "Does not quantify aromatic extent."
  h_bond_donors:
    display_name: "H-bond donor proxy"
    rule: "RDKit HBD count > 2"
    limitation: "Does not encode crystal accessibility."
  h_bond_acceptors:
    display_name: "H-bond acceptor proxy"
    rule: "RDKit HBA count > 2"
    limitation: "Does not encode crystal accessibility."
  planarity_single_ring:
    display_name: "Single-ring proxy"
    rule: "RDKit ring count = 1"
    limitation: "This is not a geometric planarity calculation."
  macrocycle_gt6_rings:
    display_name: "Ring-rich (>6 rings) proxy"
    rule: "RDKit ring count > 6"
    limitation: "This is not a universal macrocycle definition."
  charge_present:
    display_name: "Charge present"
    rule: "Net formal charge != 0"
    limitation: "Depends on linker SMILES/protonation representation."
  conjugation_present:
    display_name: "Conjugation present"
    rule: "Any bond order > 1.5"
    limitation: "Simplified proxy; not full orbital-delocalization analysis."
"""


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def first_col(df: pd.DataFrame, candidates: Iterable[str], required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        fail(f"None of the required columns were found: {list(candidates)}")
    return None


def normalize_split_value(v):
    s = str(v).strip().lower()
    mapping = {
        "train": "train",
        "training": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "test": "test",
        "testing": "test",
    }
    return mapping.get(s, s)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def normalize_manifest(raw: pd.DataFrame, labels: pd.DataFrame, strict: bool) -> pd.DataFrame:
    df = raw.copy()

    filename_col = first_col(df, ["filename", "image_name"])
    split_candidates = ["split_group_disjoint", "group_disjoint_split", "split"] if strict else ["split", "current_split"]
    split_col = first_col(df, split_candidates)
    canonical_col = first_col(df, ["canonical_first_linker_smiles", "group_first_linker"])
    group_col = first_col(df, ["group_first_linker", "canonical_first_linker_smiles"])

    rename = {
        filename_col: "filename",
        split_col: "split",
        canonical_col: "canonical_first_linker_smiles",
        group_col: "group_first_linker",
    }
    # Avoid duplicate rename keys when canonical/group are the same column.
    clean_rename = {}
    for old, new in rename.items():
        if old not in clean_rename:
            clean_rename[old] = new
    df = df.rename(columns=clean_rename)

    if "filename" not in df.columns:
        fail("Normalized manifest has no filename column.")

    df["filename"] = df["filename"].astype(str)
    df["split"] = df["split"].map(normalize_split_value)

    # If group_first_linker disappeared because it used the same source as canonical, recreate it.
    if "group_first_linker" not in df.columns and "canonical_first_linker_smiles" in df.columns:
        df["group_first_linker"] = df["canonical_first_linker_smiles"]

    # Merge authoritative compact labels by filename if necessary.
    label_table = labels.rename(columns={"image_name": "filename"})
    label_table["filename"] = label_table["filename"].astype(str)

    missing_label_cols = [c for c in LABELS if c not in df.columns]
    if missing_label_cols:
        df = df.merge(label_table[["filename"] + LABELS], on="filename", how="left", validate="many_to_one")
    else:
        # Verify split-file labels exactly agree with Data.csv.
        check = df[["filename"] + LABELS].merge(
            label_table[["filename"] + LABELS],
            on="filename",
            how="left",
            suffixes=("_manifest", "_data"),
            validate="many_to_one",
        )
        for c in LABELS:
            neq = check[f"{c}_manifest"].astype(int) != check[f"{c}_data"].astype(int)
            if neq.any():
                fail(f"Label mismatch between manifest and Data.csv for {c}: {int(neq.sum())} rows")

    if len(df) != 9414:
        fail(f"Manifest row count is {len(df)}, expected 9414.")

    counts = df["split"].value_counts().to_dict()
    if counts != EXPECTED_SPLIT_COUNTS:
        fail(f"Split counts are {counts}, expected {EXPECTED_SPLIT_COUNTS}")

    for c in LABELS:
        if df[c].isna().any():
            fail(f"Missing values in label {c}")
        vals = set(pd.to_numeric(df[c]).astype(int).unique())
        if not vals.issubset({0, 1}):
            fail(f"Non-binary values in {c}: {vals}")
        df[c] = pd.to_numeric(df[c]).astype(int)

    if "canonical_first_linker_smiles" not in df.columns:
        fail("canonical_first_linker_smiles missing after normalization.")
    if df["canonical_first_linker_smiles"].isna().any():
        fail("Missing canonical_first_linker_smiles values.")

    if strict:
        crossing = df.groupby("canonical_first_linker_smiles")["split"].nunique()
        n_cross = int((crossing > 1).sum())
        if n_cross != 0:
            fail(f"STRICT split has {n_cross} canonical first-linker groups crossing splits; expected 0.")

    preferred = [
        "row_index",
        "sample_id",
        "filename",
        "qmof_id",
        "selected_linker_smiles",
        "canonical_first_linker_smiles",
        "group_first_linker",
        "split",
    ] + LABELS

    df = df.reset_index(drop=True)
    df.insert(0, "row_index", range(len(df)))

    # Prevent duplicate row_index in preferred list.
    seen = set()
    cols = []
    for c in preferred:
        if c in df.columns and c not in seen:
            cols.append(c)
            seen.add(c)

    # Preserve useful extra metadata after the canonical columns.
    extras = [c for c in df.columns if c not in seen]
    return df[cols + extras]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", default=DEFAULT_SOURCE)
    ap.add_argument("--repo-root", default=DEFAULT_REPO)
    args = ap.parse_args()

    src = Path(args.source_root)
    repo = Path(args.repo_root)
    frozen = repo / "data" / "frozen"
    source_data = repo / "data" / "source"
    frozen.mkdir(parents=True, exist_ok=True)

    data_csv = source_data / "Data.csv"
    current_csv = source_data / "split_audit" / "current_split_master_with_linker_keys.csv"
    strict_csv = source_data / "split_audit" / "group_disjoint_master_split.csv"
    image_zip = src / "qmof_29.zip"
    if not image_zip.exists():
        image_zip = src / "vista_images_9414.zip"

    print("=" * 80)
    print("VISTA PHASE A — DATA VALIDATION")
    print("=" * 80)
    print("Repo:", repo)
    print("Image archive:", image_zip)

    labels = read_csv(data_csv)

    # Data.csv exact checks.
    if len(labels) != 9414:
        fail(f"Data.csv has {len(labels)} rows, expected 9414.")
    if "image_name" not in labels.columns:
        fail("Data.csv missing image_name.")
    if labels["image_name"].nunique() != 9414:
        fail("Data.csv image_name values are not unique.")
    if labels.isna().any().any():
        fail("Data.csv contains missing cells.")

    for c in LABELS:
        if c not in labels.columns:
            fail(f"Data.csv missing label: {c}")
        vals = set(pd.to_numeric(labels[c]).astype(int).unique())
        if not vals.issubset({0, 1}):
            fail(f"Data.csv non-binary label {c}: {vals}")
        actual = int(pd.to_numeric(labels[c]).sum())
        expected = EXPECTED_POSITIVE_COUNTS[c]
        if actual != expected:
            fail(f"Positive count mismatch for {c}: {actual} != expected {expected}")

    print("[PASS] Data.csv row/name/label checks")

    # ZIP exact checks, no extraction.
    if not image_zip.exists():
        fail(f"Image ZIP not found: {image_zip}")
    with zipfile.ZipFile(image_zip, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            fail(f"ZIP CRC/integrity failure at member: {bad}")
        infos = [i for i in zf.infolist() if not i.is_dir()]
        png_infos = [i for i in infos if i.filename.lower().endswith(".png")]

    if len(png_infos) != 9414:
        fail(f"Image ZIP has {len(png_infos)} PNG files, expected 9414.")
    zip_names = [i.filename for i in png_infos]
    if len(set(zip_names)) != 9414:
        fail("Image ZIP member names are not unique.")
    if any("/" in n or "\\" in n for n in zip_names):
        fail("Unexpected subdirectory paths found inside image ZIP.")

    data_names = set(labels["image_name"].astype(str))
    zip_name_set = set(zip_names)
    missing = sorted(data_names - zip_name_set)
    extra = sorted(zip_name_set - data_names)
    if missing or extra:
        fail(f"ZIP/Data name mismatch: missing={len(missing)}, extra={len(extra)}")

    print("[PASS] qmof_29.zip integrity and 1:1 Data.csv crosswalk")

    # Image manifest.
    image_manifest = pd.DataFrame({
        "filename": [i.filename for i in png_infos],
        "uncompressed_bytes": [i.file_size for i in png_infos],
        "compressed_bytes": [i.compress_size for i in png_infos],
        "crc32": [f"{i.CRC:08x}" for i in png_infos],
    }).sort_values("filename").reset_index(drop=True)
    image_manifest.to_csv(frozen / "image_manifest.csv", index=False)

    # Current and strict manifests.
    current_raw = read_csv(current_csv)
    strict_raw = read_csv(strict_csv)

    current = normalize_manifest(current_raw, labels, strict=False)
    strict = normalize_manifest(strict_raw, labels, strict=True)

    current.to_csv(frozen / "current_sample_split.csv", index=False)
    strict.to_csv(frozen / "canonical_group_disjoint_split.csv", index=False)

    # Strict split identity audit.
    strict_cross = strict.groupby("canonical_first_linker_smiles")["split"].nunique()
    strict_cross_count = int((strict_cross > 1).sum())
    print(f"[PASS] Strict split canonical identity crossing = {strict_cross_count}")

    current_cross = current.groupby("canonical_first_linker_smiles")["split"].nunique()
    current_cross_count = int((current_cross > 1).sum())

    unique_canonical = int(current["canonical_first_linker_smiles"].nunique())

    # Label definitions.
    (frozen / "label_definitions.yaml").write_text(LABEL_YAML, encoding="utf-8")

    # Hashes.
    hash_targets = [
        image_zip,
        data_csv,
        current_csv,
        strict_csv,
        frozen / "current_sample_split.csv",
        frozen / "canonical_group_disjoint_split.csv",
        frozen / "label_definitions.yaml",
        frozen / "image_manifest.csv",
    ]
    hash_lines = []
    hashes = {}
    for p in hash_targets:
        digest = sha256(p)
        hashes[str(p)] = digest
        hash_lines.append(f"{digest}  {p}")
        print(f"[HASH] {p.name}: {digest}")

    (frozen / "SHA256SUMS.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")

    report = {
        "status": "PASS",
        "data_rows": len(labels),
        "data_unique_image_names": int(labels["image_name"].nunique()),
        "zip_png_members": len(png_infos),
        "zip_unique_png_names": len(set(zip_names)),
        "zip_data_missing": len(missing),
        "zip_data_extra": len(extra),
        "unique_canonical_first_linker_smiles_current": unique_canonical,
        "current_cross_split_canonical_groups": current_cross_count,
        "strict_cross_split_canonical_groups": strict_cross_count,
        "strict_split_counts": strict["split"].value_counts().to_dict(),
        "positive_counts": {c: int(labels[c].sum()) for c in LABELS},
        "hashes": hashes,
    }

    (frozen / "PHASE_A_VALIDATION_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = []
    md.append("# VISTA Phase A Validation Report")
    md.append("")
    md.append("**STATUS: PASS**")
    md.append("")
    md.append(f"- Data rows: {len(labels):,}")
    md.append(f"- Unique image names: {labels['image_name'].nunique():,}")
    md.append(f"- ZIP PNG members: {len(png_infos):,}")
    md.append(f"- ZIP/Data missing: {len(missing)}")
    md.append(f"- ZIP/Data extra: {len(extra)}")
    md.append(f"- Unique canonical first-linker SMILES (current manifest): {unique_canonical:,}")
    md.append(f"- Current split canonical groups crossing splits: {current_cross_count:,}")
    md.append(f"- Strict split canonical groups crossing splits: {strict_cross_count:,}")
    md.append(f"- Strict split counts: {strict['split'].value_counts().to_dict()}")
    md.append("")
    md.append("## Positive counts")
    md.append("")
    for c in LABELS:
        md.append(f"- {c}: {int(labels[c].sum()):,}")
    md.append("")
    md.append("## Frozen outputs")
    md.append("")
    for name in [
        "current_sample_split.csv",
        "canonical_group_disjoint_split.csv",
        "label_definitions.yaml",
        "image_manifest.csv",
        "SHA256SUMS.txt",
    ]:
        md.append(f"- `{name}`")
    md.append("")
    md.append("No image files were extracted during validation.")

    (frozen / "PHASE_A_VALIDATION_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("")
    print("=" * 80)
    print("PASS — PHASE A DATA FREEZE COMPLETED")
    print("=" * 80)
    print("Validation report:")
    print(frozen / "PHASE_A_VALIDATION_REPORT.md")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("")
        print("=" * 80, file=sys.stderr)
        print("FAIL — VISTA PHASE A VALIDATION", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        raise
