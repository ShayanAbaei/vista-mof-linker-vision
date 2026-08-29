#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd, numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--family",required=True)
    ap.add_argument("--run-dirs",nargs="+",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    rows=[]; urows=[]
    for rd in map(Path,args.run_dirs):
        rt=json.loads((rd/"runtime.json").read_text())
        seed=rt.get("seed")
        x=pd.read_csv(rd/"test_metrics_overall.csv").iloc[0].to_dict()
        x.update({"seed":seed,"run_dir":str(rd)});rows.append(x)
        up=rd/"unique_canonical_test_metrics_overall.csv"
        if up.exists():
            u=pd.read_csv(up).iloc[0].to_dict()
            u.update({"seed":seed,"run_dir":str(rd)});urows.append(u)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows);df.to_csv(out/"per_seed_sample_weighted.csv",index=False)
    metrics=[c for c in ["macro_auprc","macro_auroc","macro_f1","micro_f1","subset_accuracy","hamming_loss"] if c in df]
    summary=pd.DataFrame([{"metric":m,"mean":df[m].mean(),"sd":df[m].std(ddof=1),"n_seeds":len(df)} for m in metrics])
    summary.to_csv(out/"summary_sample_weighted.csv",index=False)
    if urows:
        udf=pd.DataFrame(urows);udf.to_csv(out/"per_seed_unique_canonical.csv",index=False)
        us=pd.DataFrame([{"metric":m,"mean":udf[m].mean(),"sd":udf[m].std(ddof=1),"n_seeds":len(udf)} for m in metrics if m in udf])
        us.to_csv(out/"summary_unique_canonical.csv",index=False)
    print("PASS — repeated-seed family summary:",args.family)
if __name__=="__main__":
    main()
