#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy.stats import mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LABELS=[
    "hydrophobicity","polarity","flexibility","aromaticity",
    "h_bond_donors","h_bond_acceptors","planarity_single_ring",
    "macrocycle_gt6_rings","charge_present","conjugation_present",
]

def norm(s): return str(s).lower().replace(" ","").replace("_","").replace("-","")

def find_metadata(repo: Path, explicit=None):
    if explicit:
        return Path(explicit)
    candidates=[]
    for folder in [repo/"data/source",repo/"data/frozen",repo/"data/raw"]:
        if folder.exists():
            candidates += sorted(folder.glob("*.csv"))
    wanted={"density","pld","lcd","volume","natoms"}
    for p in candidates:
        try:
            cols=pd.read_csv(p,nrows=3).columns
        except Exception:
            continue
        cset={norm(x) for x in cols}
        if any(norm(w) in cset for w in wanted) and any(norm(k) in cset for k in ["qmof_id","filename","sample_id"]):
            return p
    return None

def canonical_column_map(cols):
    aliases={
        "density":["density"],
        "pld":["pld","porelimitingdiameter"],
        "lcd":["lcd","largestcavitydiameter"],
        "volume":["volume","cellvolume"],
        "natoms":["natoms","numberofatoms","natom"],
        "qmof_id":["qmofid","qmof_id"],
        "filename":["filename"],
        "sample_id":["sampleid","sample_id"],
    }
    nmap={norm(c):c for c in cols}
    out={}
    for k,vals in aliases.items():
        for v in vals:
            if norm(v) in nmap:
                out[k]=nmap[norm(v)]
                break
    return out

def cliff_delta(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)==0 or len(y)==0:return np.nan
    # exact pairwise for these modest groups
    return float((np.sum(x[:,None]>y[None,:])-np.sum(x[:,None]<y[None,:]))/(len(x)*len(y)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",default=".")
    ap.add_argument("--config",default="configs/mof_consequence.yaml")
    ap.add_argument("--metadata",default=None)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    repo=Path(args.repo_root).resolve()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load((repo/args.config).read_text())
    meta_path=find_metadata(repo,args.metadata)
    if meta_path is None:
        reason="No traceable local metadata CSV with qmof/sample key plus density/PLD/LCD/volume/natoms was found."
        (out/"NOT_RUN_REASON.md").write_text("# MOF-level consequence not run\n\n"+reason+"\n",encoding="utf-8")
        (out/"mof_level_consequence_status.json").write_text(json.dumps({"status":"NOT_RUN","reason":reason},indent=2),encoding="utf-8")
        print("OPTIONAL MOF-LEVEL CONSEQUENCE: NOT RUN —",reason)
        return 0

    meta=pd.read_csv(meta_path)
    cmap=canonical_column_map(meta.columns)
    manifest=pd.read_csv(repo/"data/frozen/canonical_group_disjoint_split.csv")
    join=None
    for k in ["qmof_id","filename","sample_id"]:
        if k in cmap and k in manifest.columns:
            join=k;break
    if join is None:
        reason=f"Metadata found ({meta_path}) but no shared qmof_id/filename/sample_id join key."
        (out/"NOT_RUN_REASON.md").write_text("# MOF-level consequence not run\n\n"+reason+"\n",encoding="utf-8")
        (out/"mof_level_consequence_status.json").write_text(json.dumps({"status":"NOT_RUN","reason":reason},indent=2),encoding="utf-8")
        print("OPTIONAL MOF-LEVEL CONSEQUENCE: NOT RUN —",reason)
        return 0

    rename={cmap[k]:k for k in cmap}
    meta=meta.rename(columns=rename)
    available=[p for p in cfg["candidate_properties"] if p in meta.columns]
    if not available:
        reason=f"Metadata {meta_path} joined but none of the predeclared candidate properties are present."
        (out/"NOT_RUN_REASON.md").write_text("# MOF-level consequence not run\n\n"+reason+"\n",encoding="utf-8")
        (out/"mof_level_consequence_status.json").write_text(json.dumps({"status":"NOT_RUN","reason":reason},indent=2),encoding="utf-8")
        print("OPTIONAL MOF-LEVEL CONSEQUENCE: NOT RUN —",reason)
        return 0

    use_cols=[join]+available
    d=manifest.merge(meta[use_cols].drop_duplicates(join),on=join,how="left")
    for p in available:
        d[p]=pd.to_numeric(d[p],errors="coerce")

    strat=[]
    for pair in cfg["predeclared_stratification_pairs"]:
        lab=pair["label_column"];prop=pair["property"]
        if prop not in d.columns: continue
        x=d.loc[d[lab].eq(1),prop].dropna().to_numpy(float)
        y=d.loc[d[lab].eq(0),prop].dropna().to_numpy(float)
        if min(len(x),len(y))<20: continue
        u,pv=mannwhitneyu(x,y,alternative="two-sided")
        strat.append({
            "label":lab,"property":prop,
            "n_positive":len(x),"n_negative":len(y),
            "positive_median":float(np.median(x)),"negative_median":float(np.median(y)),
            "positive_mean":float(np.mean(x)),"negative_mean":float(np.mean(y)),
            "median_difference_positive_minus_negative":float(np.median(x)-np.median(y)),
            "cliffs_delta":cliff_delta(x,y),
            "mannwhitney_p":float(pv),
            "n_unique_canonical_positive":int(d.loc[d[lab].eq(1)&d[prop].notna(),"canonical_first_linker_smiles"].nunique()),
            "n_unique_canonical_negative":int(d.loc[d[lab].eq(0)&d[prop].notna(),"canonical_first_linker_smiles"].nunique()),
        })
    pd.DataFrame(strat).to_csv(out/"mof_property_stratification.csv",index=False)

    inc=cfg["incremental_prediction"]
    target=inc["preferred_target"]
    base=[x for x in inc["candidate_baseline_features"] if x in d.columns and x!=target]
    pred_rows=[]
    if target in d.columns and len(base)>=2:
        fit=d.dropna(subset=[target]).copy()
        train=fit[fit.split=="train"];val=fit[fit.split=="validation"];test=fit[fit.split=="test"]
        if min(len(train),len(val),len(test))>=50:
            def fit_eval(features,kind):
                best=None
                for a in inc["validation_alphas"]:
                    pipe=Pipeline([
                        ("imp",SimpleImputer(strategy="median")),
                        ("scale",StandardScaler()),
                        ("ridge",Ridge(alpha=float(a))),
                    ])
                    pipe.fit(train[features],train[target])
                    pv=pipe.predict(val[features])
                    mae=mean_absolute_error(val[target],pv)
                    if best is None or mae<best[0]:best=(mae,float(a),pipe)
                model=best[2]
                pt=model.predict(test[features])
                return {
                    "representation":kind,"target":target,"features":";".join(features),
                    "selected_alpha":best[1],"validation_mae":best[0],
                    "test_mae":mean_absolute_error(test[target],pt),
                    "test_r2":r2_score(test[target],pt),
                    "n_train":len(train),"n_validation":len(val),"n_test":len(test),
                }
            pred_rows.append(fit_eval(base,"baseline_mof_descriptors"))
            pred_rows.append(fit_eval(base+LABELS,"baseline_plus_frozen_linker_proxy_labels"))
    pd.DataFrame(pred_rows).to_csv(out/"mof_incremental_prediction.csv",index=False)

    status={
        "status":"RAN",
        "metadata_file":str(meta_path),
        "join_key":join,
        "available_properties":available,
        "stratification_tests":len(strat),
        "incremental_prediction_rows":len(pred_rows),
        "inclusion_rule":cfg["inclusion_rule"],
        "note":"This is a predeclared screen. Manuscript inclusion requires scientific interpretation and duplicate-aware support review.",
    }
    (out/"mof_level_consequence_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    print("PASS — optional predeclared MOF-level consequence screen completed:",out)

if __name__=="__main__":
    main()
