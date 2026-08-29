#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd, yaml

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",required=True)
    ap.add_argument("--repo-root",default=".")
    ap.add_argument("--policy",default="configs/e0_acceptance.yaml")
    args=ap.parse_args()

    run=Path(args.run_dir)
    repo=Path(args.repo_root).resolve()
    policy=yaml.safe_load((repo/args.policy).read_text(encoding="utf-8"))

    ref_o=pd.read_csv(repo/"reference_current_results/metrics/test_metrics_overall.csv").iloc[0]
    ref_p=pd.read_csv(repo/"reference_current_results/metrics/test_metrics_per_label.csv")
    new_o=pd.read_csv(run/"test_metrics_overall.csv").iloc[0]
    new_p=pd.read_csv(run/"test_metrics_per_label.csv")

    d_auprc=abs(float(new_o["macro_auprc"])-float(ref_o["macro_auprc"]))
    d_f1=abs(float(new_o["macro_f1"])-float(ref_o["macro_f1"]))
    m=ref_p[["label","auprc"]].merge(new_p[["label","auprc"]],on="label",suffixes=("_ref","_new"))
    m["abs_delta"]=(m["auprc_new"]-m["auprc_ref"]).abs()
    max_label=float(m["abs_delta"].max())

    tol=policy["tolerances"]
    checks={
        "macro_auprc_abs_delta":{"value":d_auprc,"limit":float(tol["macro_auprc_abs"]),"pass":d_auprc<=float(tol["macro_auprc_abs"])},
        "macro_f1_abs_delta":{"value":d_f1,"limit":float(tol["macro_f1_abs"]),"pass":d_f1<=float(tol["macro_f1_abs"])},
        "max_per_label_auprc_abs_delta":{"value":max_label,"limit":float(tol["per_label_auprc_max_abs"]),"pass":max_label<=float(tol["per_label_auprc_max_abs"])},
    }
    numeric_pass=all(x["pass"] for x in checks.values())
    result={
        "numeric_gate": "PASS" if numeric_pass else "FAIL",
        "manual_shayan_review_required": True,
        "e1_authorized": False,
        "checks":checks,
        "note":"Even PASS does not automatically authorize E1. Send E0 outputs to Shayan for review.",
    }
    (run/"E0_GATE.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    m.to_csv(run/"E0_per_label_auprc_comparison.csv",index=False)

    print(json.dumps(result,indent=2))
    if not numeric_pass:
        print("STOP — E0 is materially inconsistent; do not start E1.")
        return 3
    print("E0 NUMERIC GATE: PASS")
    print("STOP HERE AND SEND E0 OUTPUTS TO SHAYAN. Manual approval is still required.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
