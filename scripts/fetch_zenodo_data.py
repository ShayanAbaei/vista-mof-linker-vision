#!/usr/bin/env python3
"""
Download a VISTA dataset file from a published Zenodo record.

Example after publication:
    python scripts/fetch_zenodo_data.py --record-id 12345678 --filename vista_images_9414.zip
"""

from __future__ import annotations
import argparse
import hashlib
import urllib.request
from pathlib import Path

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-id", required=True)
    ap.add_argument("--filename", default="vista_images_9414.zip")
    ap.add_argument("--output-dir", default="data/raw")
    ap.add_argument("--sha256", default="", help="Optional expected SHA256")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / args.filename
    url = f"https://zenodo.org/records/{args.record_id}/files/{args.filename}?download=1"

    print("Downloading:", url)
    urllib.request.urlretrieve(url, output)
    digest = sha256(output)
    print("Saved:", output)
    print("SHA256:", digest)

    if args.sha256 and digest.lower() != args.sha256.lower():
        output.unlink(missing_ok=True)
        raise RuntimeError("SHA256 mismatch; downloaded file removed.")

if __name__ == "__main__":
    main()
