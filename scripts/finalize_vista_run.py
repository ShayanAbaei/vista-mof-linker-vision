#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path
from _handover_utils import hardlink_or_copy, run_readme, sha256

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",required=True)
    ap.add_argument("--repo-root",default=".")
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--seed",type=int,required=True)
    args=ap.parse_args()
    run=Path(args.run_dir); repo=Path(args.repo_root).resolve()
    src=run/"final_model_with_thresholds.pth"
    if not src.exists(): raise FileNotFoundError(src)
    mode=hardlink_or_copy(src,run/"best_checkpoint.pt")
    rt=json.loads((run/"runtime.json").read_text())
    (run/"best_epoch.txt").write_text(str(rt.get("best_epoch"))+"\n",encoding="utf-8")
    env=[]
    for n in ("nvidia_smi.txt","pip_freeze.txt","environment.json"):
        p=run/n
        if p.exists():
            env.append(f"===== {n} =====\n{p.read_text(encoding='utf-8',errors='replace')}")
    (run/"environment.txt").write_text("\n\n".join(env)+"\n",encoding="utf-8")
    manifest=repo/args.manifest
    run_readme(run,"VISTA",manifest,sha256(manifest),args.seed,{"checkpoint_alias":mode})
    print("PASS — VISTA output normalized:",run)
if __name__=="__main__":
    main()
