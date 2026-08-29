#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

from _bootstrap import add_vista_legacy_path
add_vista_legacy_path()
import evaluation

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",required=True)
    ap.add_argument("--tanimoto",default="phase_c_results/source_data/tanimoto_strict_test_nearest.csv")
    ap.add_argument("--repo-root",default=".")
    ap.add_argument("--output",required=True)
    ap.add_argument("--min-bin-support",type=int,default=50)
    args=ap.parse_args()

    repo=Path(args.repo_root).resolve()
    run=Path(args.run_dir)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    pred=pd.read_csv(run/"test_predictions.csv")
    tan=pd.read_csv(repo/args.tanimoto)
    thr_df=pd.read_csv(run/"thresholds.csv")
    labels=thr_df["label"].tolist()
    thr=thr_df["threshold"].to_numpy(float)

    key="filename"
    merged=pred.merge(tan[[key,"max_tanimoto"]],on=key,how="left",validate="one_to_one")
    if merged["max_tanimoto"].isna().any():
        raise RuntimeError("Missing Tanimoto values after filename join")

    initial_edges=[-1e-12,0.4,0.6,0.8,1.0000001]
    initial_names=["<0.4","0.4-0.6","0.6-0.8",">=0.8"]
    merged["bin0"]=pd.cut(merged["max_tanimoto"],bins=initial_edges,labels=initial_names,right=False,include_lowest=True)
    support=merged["bin0"].value_counts().reindex(initial_names,fill_value=0)
    use_quantile=bool((support < args.min_bin_support).any())

    if use_quantile:
        # Support-aware fallback: four equal-frequency bins. qcut handles the
        # observed distribution rather than forcing tiny arbitrary intervals.
        q=pd.qcut(merged["max_tanimoto"],q=4,duplicates="drop")
        merged["similarity_bin"]=q.astype(str)
        scheme={"type":"quartile_qcut","reason":f"at least one initial bin had support < {args.min_bin_support}"}
    else:
        merged["similarity_bin"]=merged["bin0"].astype(str)
        scheme={"type":"fixed_support_sufficient","edges":initial_edges,"labels":initial_names}

    rows=[]
    for b,g in merged.groupby("similarity_bin",sort=False):
        y=np.column_stack([g[f"gt_{lab}"].to_numpy(int) for lab in labels])
        p=np.column_stack([g[f"prob_{lab}"].to_numpy(float) for lab in labels])
        tab=evaluation.compute_metric_table(p,y,thr,labels)["overall"].iloc[0].to_dict()
        rows.append({
            "similarity_bin":b,
            "support_rows":len(g),
            "support_unique_canonical":g["canonical_first_linker_smiles"].nunique() if "canonical_first_linker_smiles" in g else np.nan,
            "tanimoto_min":g["max_tanimoto"].min(),
            "tanimoto_median":g["max_tanimoto"].median(),
            "tanimoto_max":g["max_tanimoto"].max(),
            **tab,
        })
    pd.DataFrame(rows).to_csv(out/"strict_performance_by_tanimoto_bin.csv",index=False)
    merged[[c for c in ["filename","qmof_id","canonical_first_linker_smiles","max_tanimoto","similarity_bin"] if c in merged]].to_csv(
        out/"strict_test_tanimoto_with_bins.csv",index=False
    )
    (out/"tanimoto_bin_protocol.json").write_text(json.dumps(scheme,indent=2),encoding="utf-8")
    print("PASS — strict performance-by-nearest-training-similarity exported:",out)

if __name__=="__main__":
    main()
