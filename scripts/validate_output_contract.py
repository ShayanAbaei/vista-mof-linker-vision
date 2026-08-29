#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

REQUIRED=[
    "resolved_config.yaml","command.txt","git_commit.txt","nvidia_smi.txt",
    "pip_freeze.txt","environment.txt","history.csv","best_checkpoint.pt","best_epoch.txt",
    "thresholds.csv","validation_predictions.csv","test_predictions.csv",
    "validation_metrics_overall.csv","test_metrics_overall.csv",
    "validation_metrics_per_label.csv","test_metrics_per_label.csv",
    "confusion_matrices.csv","runtime.json","README_RUN.md",
]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",required=True)
    ap.add_argument("--require-unique-canonical",action="store_true")
    args=ap.parse_args()
    run=Path(args.run_dir)
    missing=[x for x in REQUIRED if not (run/x).exists()]
    if args.require_unique_canonical:
        for x in ["unique_canonical_test_predictions.csv","unique_canonical_test_metrics_overall.csv","unique_canonical_test_metrics_per_label.csv"]:
            if not (run/x).exists(): missing.append(x)
    if missing:
        print("FAIL — missing output-contract files:")
        for x in missing: print(" -",x)
        return 2
    t=pd.read_csv(run/"test_predictions.csv")
    v=pd.read_csv(run/"validation_predictions.csv")
    if len(t)!=1412 or len(v)!=1412:
        raise RuntimeError(f"prediction row counts: val={len(v)} test={len(t)}")
    thr=pd.read_csv(run/"thresholds.csv")
    if len(thr)!=10: raise RuntimeError("thresholds.csv must have 10 rows")
    rt=json.loads((run/"runtime.json").read_text())
    print("PASS — output contract:",run)
    print("family:",rt.get("family") or rt.get("mode"))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
