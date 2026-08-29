#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, shutil, urllib.request, zipfile
from pathlib import Path
import pandas as pd

RECORD_ID = "22141996"
FILE_NAME = "vista_images_9414.zip"
EXPECTED_SHA256 = "244c3d29bea6ed90c7b12c2a48eda2277ad24a3b41a64088e7256b349523c578"
EXPECTED_SIZE = 70112391

def sha256(path, chunk=1024*1024):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(chunk),b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",default=".")
    ap.add_argument("--force-download",action="store_true")
    args=ap.parse_args()
    repo=Path(args.repo_root).resolve()
    raw=repo/"data/raw"; img=repo/"data/images_224"
    raw.mkdir(parents=True,exist_ok=True); img.mkdir(parents=True,exist_ok=True)
    dst=raw/FILE_NAME

    api=f"https://zenodo.org/api/records/{RECORD_ID}"
    with urllib.request.urlopen(api) as r:
        rec=json.load(r)
    files={x["key"]:x for x in rec["files"]}
    if FILE_NAME not in files:
        raise RuntimeError(f"{FILE_NAME} not found in Zenodo record {RECORD_ID}")
    info=files[FILE_NAME]
    url=(info.get("links") or {}).get("content") or (info.get("links") or {}).get("self")
    if not url:
        raise RuntimeError("Zenodo file has no content URL")

    if args.force_download or not dst.exists():
        print("[DOWNLOAD]",url)
        with urllib.request.urlopen(url) as src, open(dst,"wb") as out:
            shutil.copyfileobj(src,out)
    else:
        print("[REUSE]",dst)

    if dst.stat().st_size != EXPECTED_SIZE:
        raise RuntimeError(f"archive size {dst.stat().st_size} != {EXPECTED_SIZE}")
    digest=sha256(dst)
    print("SHA256:",digest)
    if digest != EXPECTED_SHA256:
        raise RuntimeError("Zenodo image archive SHA256 mismatch")

    manifest=pd.read_csv(repo/"data/frozen/image_manifest.csv")
    expected=set(manifest["filename"].astype(str))
    with zipfile.ZipFile(dst) as z:
        names=[x.filename for x in z.infolist() if not x.is_dir()]
        if len(names)!=9414 or set(names)!=expected:
            raise RuntimeError("ZIP member names do not match frozen image manifest")
        for member in z.infolist():
            if member.is_dir(): continue
            name=member.filename
            if "/" in name or "\\" in name or Path(name).name != name:
                raise RuntimeError(f"unsafe archive member: {name}")
            target=img/name
            if not target.exists():
                with z.open(member) as src, open(target,"wb") as out:
                    shutil.copyfileobj(src,out)

    disk={p.name for p in img.glob("*.png")}
    if disk != expected:
        raise RuntimeError(f"extracted image set mismatch: disk={len(disk)}, expected={len(expected)}")
    print("PASS — Zenodo image archive verified and 9,414 images available:",img)

if __name__=="__main__":
    main()
