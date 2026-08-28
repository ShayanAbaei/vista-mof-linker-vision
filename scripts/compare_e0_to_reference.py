#!/usr/bin/env python3
"""Compare a new E0 sample-level reproduction run against the frozen historical reference.

This is a diagnostic comparator, not a test-set tuning tool. It reports deltas.
By default it exits non-zero only if required files are missing. Optional
tolerances can be supplied to make it a stricter acceptance gate.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",required=True)
    ap.add_argument("--repo-root",default=".")
    ap.add_argument("--macro-auprc-tolerance",type=float,default=None)
    ap.add_argument("--per-label-auprc-tolerance",type=float,default=None)
    args=ap.parse_args()

    run=Path(args.run_dir)
    repo=Path(args.repo_root).resolve()

    ref_over=repo/"reference_current_results/metrics/test_metrics_overall.csv"
    ref_per=repo/"reference_current_results/metrics/test_metrics_per_label.csv"
    new_over=run/"test_metrics_overall.csv"
    new_per=run/"test_metrics_per_label.csv"
    facts=repo/"reference_current_results/training/HISTORICAL_REFERENCE_FACTS.json"

    required=[ref_over,ref_per,new_over,new_per]
    missing=[str(p) for p in required if not p.exists()]
    if missing:
        print("Missing required files:")
        for p in missing: print(" -",p)
        return 2

    ro=pd.read_csv(ref_over).iloc[0]
    no=pd.read_csv(new_over).iloc[0]
    rp=pd.read_csv(ref_per)
    np_=pd.read_csv(new_per)

    print("E0 HISTORICAL REPRODUCTION COMPARISON")
    print("="*72)
    metrics=["macro_auprc","macro_auroc","macro_f1","macro_balanced_accuracy","micro_auprc","hamming_loss"]
    for m in metrics:
        if m in ro and m in no:
            d=float(no[m])-float(ro[m])
            print(f"{m:28s} ref={float(ro[m]):.9f} new={float(no[m]):.9f} delta={d:+.9f}")

    merged=rp[["label","auprc"]].merge(np_[["label","auprc"]],on="label",suffixes=("_ref","_new"))
    merged["delta"]=merged["auprc_new"]-merged["auprc_ref"]
    print("\nPer-label AUPRC deltas:")
    print(merged.to_string(index=False,float_format=lambda x:f"{x:.6f}"))

    if facts.exists():
        f=json.loads(facts.read_text())
        print("\nHistorical best epoch:",f.get("best_epoch_by_val_macro_auprc"))
        print("Historical best val macro-AUPRC:",f.get("best_val_macro_auprc"))

    failures=[]
    if args.macro_auprc_tolerance is not None:
        d=abs(float(no["macro_auprc"])-float(ro["macro_auprc"]))
        if d>args.macro_auprc_tolerance:
            failures.append(f"macro AUPRC delta {d:.6f} exceeds tolerance {args.macro_auprc_tolerance}")
    if args.per_label_auprc_tolerance is not None:
        bad=merged[merged["delta"].abs()>args.per_label_auprc_tolerance]
        if len(bad):
            failures.append(f"{len(bad)} per-label AUPRC deltas exceed tolerance {args.per_label_auprc_tolerance}")

    if failures:
        print("\nFAIL:")
        for x in failures: print(" -",x)
        return 3
    print("\nPASS — comparison completed.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
