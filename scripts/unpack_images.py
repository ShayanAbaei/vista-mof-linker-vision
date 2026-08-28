#!/usr/bin/env python3
"""Safely extract the validated VISTA molecular image archive."""

from __future__ import annotations
import argparse
import zipfile
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, help="Path to vista_images_9414.zip or qmof_29.zip")
    ap.add_argument("--manifest", default="data/frozen/image_manifest.csv")
    ap.add_argument("--output", default="data/images_224")
    args = ap.parse_args()

    archive = Path(args.archive).resolve()
    manifest = Path(args.manifest).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    expected = set(pd.read_csv(manifest)["filename"].astype(str))

    with zipfile.ZipFile(archive, "r") as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        names = [i.filename for i in infos]
        if len(names) != 9414 or set(names) != expected:
            raise RuntimeError("Archive members do not exactly match frozen image manifest.")
        for info in infos:
            # No nested paths are expected; enforce this.
            if "/" in info.filename or "\\" in info.filename:
                raise RuntimeError(f"Unexpected nested path in archive: {info.filename}")
            target = out / info.filename
            with zf.open(info, "r") as src, target.open("wb") as dst:
                while True:
                    b = src.read(1024 * 1024)
                    if not b:
                        break
                    dst.write(b)

    extracted = {p.name for p in out.glob("*.png")}
    if extracted != expected:
        raise RuntimeError(f"Extraction mismatch: extracted={len(extracted)}, expected={len(expected)}")
    print(f"PASS: extracted {len(extracted)} images to {out}")

if __name__ == "__main__":
    main()
