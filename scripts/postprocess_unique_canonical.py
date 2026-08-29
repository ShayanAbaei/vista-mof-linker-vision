#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd

from _bootstrap import add_vista_legacy_path
add_vista_legacy_path()
import evaluation

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",required=True)
    ap.add_argument("--splits",nargs="+",default=["validation","test"])
    args=ap.parse_args()
    run=Path(args.run_dir)
    thr_df=pd.read_csv(run/"thresholds.csv")
    labels=thr_df["label"].tolist()
    thr=thr_df["threshold"].to_numpy(float)

    for split in args.splits:
        p=run/f"{split}_predictions.csv"
        df=pd.read_csv(p)
        if "canonical_first_linker_smiles" not in df.columns:
            raise ValueError(f"{p}: canonical_first_linker_smiles missing")
        rows=[]
        for smi,g in df.groupby("canonical_first_linker_smiles",sort=False):
            r={"canonical_first_linker_smiles":smi,"n_records":len(g)}
            for lab in labels:
                gt=f"gt_{lab}"; prob=f"prob_{lab}"
                if g[gt].nunique()!=1:
                    raise ValueError(f"inconsistent target {lab} within {smi}")
                r[gt]=int(g[gt].iloc[0]); r[prob]=float(g[prob].mean())
            rows.append(r)
        agg=pd.DataFrame(rows)
        y=np.column_stack([agg[f"gt_{x}"].to_numpy(int) for x in labels])
        pr=np.column_stack([agg[f"prob_{x}"].to_numpy(float) for x in labels])
        pred=evaluation.apply_thresholds(pr,thr)
        for j,lab in enumerate(labels):
            agg[f"pred_{lab}"]=pred[:,j]
            agg[f"thr_{lab}"]=thr[j]
        agg.to_csv(run/f"unique_canonical_{split}_predictions.csv",index=False)
        tables=evaluation.compute_metric_table(pr,y,thr,labels)
        evaluation.save_metric_table(tables,run/f"unique_canonical_{split}_metrics")
        print(split,":",len(agg),"unique canonical identities")
    print("PASS — unique-canonical postprocessing")
if __name__=="__main__":
    main()
