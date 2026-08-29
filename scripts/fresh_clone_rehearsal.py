#!/usr/bin/env python3
from __future__ import annotations
import argparse, shutil, subprocess, sys, tempfile
from pathlib import Path

def run(cmd,cwd=None):
    print("$"," ".join(map(str,cmd)))
    subprocess.run(list(map(str,cmd)),cwd=cwd,check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-url",required=True)
    ap.add_argument("--ref",required=True)
    ap.add_argument("--workdir",default=None)
    ap.add_argument("--fetch-images",action="store_true")
    args=ap.parse_args()

    base=Path(args.workdir).resolve() if args.workdir else Path(tempfile.mkdtemp(prefix="vista_fresh_clone_"))
    clone=base/"vista-mof-linker-vision"
    if clone.exists(): raise FileExistsError(clone)
    run(["git","clone",args.repo_url,str(clone)])
    run(["git","checkout",args.ref],clone)
    run([sys.executable,"scripts/pre_epfl_rehearsal.py","--repo-root","."],clone)
    if args.fetch_images:
        run([sys.executable,"scripts/fetch_zenodo_images.py","--repo-root","."],clone)
        run([sys.executable,"scripts/pre_epfl_rehearsal.py","--repo-root",".","--require-images"],clone)
    print("PASS — actual fresh-clone repository rehearsal:",clone)
    print("Torch/GPU runtime still belongs to the EPFL runtime gate.")
if __name__=="__main__":
    main()
