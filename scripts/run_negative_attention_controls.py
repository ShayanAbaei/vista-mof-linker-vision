#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

LABELS=[
    "Hydrophobicity","Polarity","Flexibility","Aromaticity",
    "Hydrogen-Bond Donors","Hydrogen-Bond Acceptors",
    "Planarity (Single-Ring)","Macrocycle (>6 Rings)",
    "Charge Present","Conjugation Present",
]

def entropy01(m):
    x=np.asarray(m,dtype=float).reshape(-1)
    s=x.sum()
    if s<=0: return np.nan
    p=x/s
    p=p[p>0]
    h=-(p*np.log(p)).sum()
    return float(h/np.log(len(x)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--attention-dir",required=True)
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--n-low",type=int,default=5)
    ap.add_argument("--n-borderline",type=int,default=5)
    args=ap.parse_args()

    import torch
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    root=Path(args.attention_dir)
    z=np.load(root/"test_attention_summary.npz")
    maps=z["label_image_final_meanheads"]
    targets=z["targets"].astype(int)
    probs=z["probs"].astype(float)
    meta=pd.read_csv(root/"test_attention_metadata.csv")
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    thresholds=np.asarray(ck.get("thresholds") or [0.5]*10,dtype=float)

    rows=[]
    for j,lab in enumerate(LABELS):
        pred=(probs[:,j]>thresholds[j]).astype(int)
        tn=np.where((targets[:,j]==0)&(pred==0))[0]
        if not len(tn): continue

        # Very-low probability negatives: strongest negative controls.
        low=tn[np.argsort(probs[tn,j])[:args.n_low]]
        # Borderline negatives: useful sanity cases near the frozen threshold.
        border=tn[np.argsort(np.abs(probs[tn,j]-thresholds[j]))[:args.n_borderline]]

        for case_type,idxs in [("true_negative_low_probability",low),("true_negative_borderline",border)]:
            for idx in idxs:
                m=maps[idx,j]
                rows.append({
                    "label":lab,
                    "case_type":case_type,
                    "test_row":int(idx),
                    "filename":meta.iloc[idx]["filename"],
                    "canonical_first_linker_smiles":meta.iloc[idx].get("canonical_first_linker_smiles"),
                    "ground_truth":0,
                    "prediction":0,
                    "probability":float(probs[idx,j]),
                    "threshold":float(thresholds[j]),
                    "attention_peak":float(np.max(m)),
                    "attention_mean":float(np.mean(m)),
                    "attention_entropy_0_1":entropy01(m),
                })

    df=pd.DataFrame(rows)
    df.to_csv(out/"negative_label_attention_manifest.csv",index=False)
    summary=df.groupby(["label","case_type"]).agg(
        n=("filename","size"),
        mean_probability=("probability","mean"),
        mean_attention_peak=("attention_peak","mean"),
        mean_attention_entropy=("attention_entropy_0_1","mean"),
    ).reset_index()
    summary.to_csv(out/"negative_label_attention_summary.csv",index=False)
    (out/"README_NEGATIVE_CONTROL.md").write_text(
        "# Negative-label attention control\n\n"
        "Candidates are selected deterministically from strict-test true negatives after the seed-42 "
        "model and thresholds are frozen. These are sanity/control examples only. Attention is not "
        "treated as causal evidence, and a negative label is not required to have zero visual attention.\n",
        encoding="utf-8",
    )
    print("PASS — explicit negative-label attention controls exported:",out)

if __name__=="__main__":
    main()
