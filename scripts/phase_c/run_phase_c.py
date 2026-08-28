#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    LABEL_COLUMNS, LABEL_DISPLAY, EXPECTED_N, EXPECTED_SPLIT, EXPECTED_UNIQUE_CANONICAL,
    first_linker_from_cell, macro_auprc, metrics_from_probs, normalize_manifest,
    read_json, setup_logger, sha256, threshold_by_f1, write_json,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "configs" / "phase_c_local_baselines.json"

GENERAL_DESCRIPTORS = [
    "MolWt", "HeavyAtomMolWt", "ExactMolWt", "NumValenceElectrons", "HeavyAtomCount",
    "NHOHCount", "NOCount", "NumHeteroatoms", "FractionCSP3", "LabuteASA", "MolMR",
    "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3",
    "Chi0", "Chi1", "Chi0n", "Chi1n", "Chi2n", "Chi3n", "Chi4n",
    "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3",
    "MolLogP", "TPSA", "NumRotatableBonds", "NumHDonors",
    "RingCount", "NumAromaticRings", "NumAromaticCarbocycles", "NumAromaticHeterocycles",
    "NumAliphaticRings", "NumSaturatedRings", "NumHeterocycles",
]
CUSTOM_DESCRIPTORS = [
    "NumHAcceptors_FrozenV201",
    "AromaticAtomCount",
    "NetFormalCharge",
    "BondOrderGT1p5Count",
    "HasBondOrderGT1p5",
]

# RDKit NumHBA v2.0.1 SMARTS (Release_2025_09_4).
# RDKit 2026.03 changed this definition to v2.0.2 by constraining aromatic n to X2.
# The VISTA frozen labels were generated under the older semantics, so the old
# SMARTS is embedded explicitly for version-stable label auditing.
HBA_FROZEN_V201_SMARTS = (
    "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),"
    "$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]"
)
RING_DIRECT = {
    "RingCount", "NumAromaticRings", "NumAromaticCarbocycles", "NumAromaticHeterocycles",
    "NumAliphaticRings", "NumSaturatedRings", "NumHeterocycles",
}
LRO_EXCLUDE = {
    "hydrophobicity": {"MolLogP"},
    "polarity": {"TPSA"},
    "flexibility": {"NumRotatableBonds"},
    "aromaticity": {"AromaticAtomCount", "NumAromaticRings", "NumAromaticCarbocycles", "NumAromaticHeterocycles"},
    "h_bond_donors": {"NumHDonors", "NHOHCount"},
    "h_bond_acceptors": {"NumHAcceptors_FrozenV201", "NOCount"},
    "planarity_single_ring": RING_DIRECT,
    "macrocycle_gt6_rings": RING_DIRECT,
    "charge_present": {"NetFormalCharge"},
    "conjugation_present": {"BondOrderGT1p5Count", "HasBondOrderGT1p5"},
}


def stage(log, n, total, name):
    log.info("=" * 78)
    log.info(f"[PHASE C] [{n}/{total}] {name}")
    log.info("=" * 78)


def git_commit():
    try: return subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: return None


def load_config():
    return read_json(CONFIG_PATH)


def paths(cfg):
    root=REPO/Path(cfg["outputs"]["root"])
    return {
        "root": root,
        "audit": root/"audit",
        "cache": root/"cache",
        "metrics": root/"metrics",
        "source": root/"source_data",
        "bootstrap": root/"bootstrap",
        "figures": root/"figures",
        "report": root/"report",
        "logs": root/"logs",
    }


def ensure_dirs(pp):
    for p in pp.values(): p.mkdir(parents=True,exist_ok=True)


def environment_info():
    import rdkit, sklearn, matplotlib, scipy
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "rdkit": rdkit.__version__,
        "sklearn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "scipy": scipy.__version__,
        "git_commit": git_commit(),
    }


def load_master_and_molecules(cfg, log):
    from rdkit import Chem
    from rdkit.Chem import Crippen, Lipinski, rdMolDescriptors
    import rdkit

    master_path=REPO/Path(cfg["data"]["master"])
    compact_path=REPO/Path(cfg["data"]["compact_labels"])
    if not master_path.exists(): raise FileNotFoundError(master_path)
    if not compact_path.exists(): raise FileNotFoundError(compact_path)
    master=pd.read_csv(master_path)
    compact=pd.read_csv(compact_path)
    if len(master)!=EXPECTED_N or len(compact)!=EXPECTED_N:
        raise ValueError(f"Expected {EXPECTED_N} rows; master={len(master)}, compact={len(compact)}")
    if master["image_name"].duplicated().any() or compact["image_name"].duplicated().any():
        raise ValueError("Duplicate image_name detected")

    # Exact compact/master agreement.
    mm=master[["image_name"]+LABEL_COLUMNS].copy()
    for c in LABEL_COLUMNS: mm[c]=mm[c].astype(float).astype(int)
    cc=compact[["image_name"]+LABEL_COLUMNS].copy()
    for c in LABEL_COLUMNS: cc[c]=cc[c].astype(float).astype(int)
    if not mm.reset_index(drop=True).equals(cc.reset_index(drop=True)):
        merged=mm.merge(cc,on="image_name",suffixes=("_master","_compact"),validate="one_to_one")
        bad=sum((merged[f"{c}_master"]!=merged[f"{c}_compact"]).sum() for c in LABEL_COLUMNS)
        raise ValueError(f"Master and compact Data.csv do not match exactly ({bad} label mismatches)")

    hba_frozen_query = Chem.MolFromSmarts(HBA_FROZEN_V201_SMARTS)
    if hba_frozen_query is None:
        raise RuntimeError("Could not compile frozen NumHBA v2.0.1 SMARTS")

    mols=[]; first=[]; canonical=[]; oracle_rows=[]; hba_drift_rows=[]
    for i,cell in enumerate(master["info.mofid.smiles_linkers"].astype(str)):
        smi=first_linker_from_cell(cell)
        mol=Chem.MolFromSmiles(smi)
        if mol is None: raise ValueError(f"RDKit failed first linker at row {i}: {smi}")
        can=Chem.MolToSmiles(mol,canonical=True)
        first.append(smi); canonical.append(can); mols.append(mol)

        rings=rdMolDescriptors.CalcNumRings(mol)
        order_count=sum(1 for b in mol.GetBonds() if b.GetBondTypeAsDouble()>1.5)

        # Version-stable frozen HBA rule.
        frozen_hba_count=len(mol.GetSubstructMatches(hba_frozen_query,uniquify=True))
        runtime_hba_count=int(rdMolDescriptors.CalcNumHBA(mol))

        oracle_rows.append({
            "hydrophobicity": int(Crippen.MolLogP(mol)>2.5),
            "polarity": int(rdMolDescriptors.CalcTPSA(mol)>90),
            "flexibility": int(Lipinski.NumRotatableBonds(mol)>5),
            "aromaticity": int(any(a.GetIsAromatic() for a in mol.GetAtoms())),
            "h_bond_donors": int(Lipinski.NumHDonors(mol)>2),
            "h_bond_acceptors": int(frozen_hba_count>2),
            "planarity_single_ring": int(rings==1),
            "macrocycle_gt6_rings": int(rings>6),
            "charge_present": int(sum(a.GetFormalCharge() for a in mol.GetAtoms())!=0),
            "conjugation_present": int(order_count>0),
        })
        if frozen_hba_count != runtime_hba_count:
            hba_drift_rows.append({
                "row_index": i,
                "image_name": master.iloc[i]["image_name"],
                "first_linker_smiles": smi,
                "canonical_first_linker_smiles": can,
                "frozen_hba_count_v2_0_1": int(frozen_hba_count),
                "runtime_hba_count": int(runtime_hba_count),
                "frozen_hba_label": int(master.iloc[i]["h_bond_acceptors"]),
                "runtime_rdkit": rdkit.__version__,
                "runtime_numhba_version": getattr(rdMolDescriptors,"_CalcNumHBA_version",None),
            })

    master=master.copy()
    master["first_linker_smiles"]=first
    master["canonical_derived"]=canonical
    for c in LABEL_COLUMNS: master[c]=master[c].astype(float).astype(int)

    oracle=pd.DataFrame(oracle_rows)
    mismatches={c:int((oracle[c].values!=master[c].values).sum()) for c in LABEL_COLUMNS}
    if any(mismatches.values()):
        raise ValueError(f"Frozen-rule audit mismatch: {mismatches}")

    uniq=master["canonical_derived"].nunique()
    if uniq!=EXPECTED_UNIQUE_CANONICAL:
        raise ValueError(f"Expected {EXPECTED_UNIQUE_CANONICAL} unique canonical first linkers, found {uniq}")

    # Persist the runtime-vs-frozen HBA semantic drift audit. This is expected on
    # RDKit >= 2026.03 because NumHBA changed from descriptor version 2.0.1 to 2.0.2.
    audit_dir=REPO/Path(cfg["outputs"]["root"])/"audit"
    audit_dir.mkdir(parents=True,exist_ok=True)
    drift_df=pd.DataFrame(hba_drift_rows)
    drift_df.to_csv(audit_dir/"rdkit_hba_version_drift_rows.csv",index=False)
    drift_summary={
        "runtime_rdkit": rdkit.__version__,
        "runtime_numhba_version": getattr(rdMolDescriptors,"_CalcNumHBA_version",None),
        "frozen_numhba_semantics": "RDKit NumHBA v2.0.1 SMARTS (Release_2025_09_4)",
        "frozen_hba_rule": "frozen_hba_count > 2",
        "runtime_vs_frozen_hba_count_differences": int(len(drift_df)),
        "frozen_rule_label_mismatches": int(mismatches["h_bond_acceptors"]),
        "interpretation": (
            "The VISTA labels are frozen historical targets. Newer RDKit versions may "
            "produce different HBA counts after the 2026.03 N-heterocycle bug fix. "
            "Phase C therefore audits HBA using the embedded historical v2.0.1 SMARTS "
            "and records runtime descriptor drift separately."
        ),
    }
    write_json(drift_summary,audit_dir/"rdkit_hba_version_drift_summary.json")

    log.info(f"[OK] master + compact labels: {len(master):,} rows, exact agreement")
    log.info(f"[OK] first-linker RDKit parse: {len(mols):,}/{len(master):,}")
    log.info(f"[OK] canonical first-linker identities: {uniq:,}")
    log.info("[OK] frozen deterministic label rules reproduce all 10 labels with zero mismatch")
    if len(drift_df):
        log.info(
            f"[INFO] Runtime RDKit NumHBA differs from frozen v2.0.1 HBA count on "
            f"{len(drift_df):,} rows; this is recorded as version drift, not a label error"
        )
    return master,mols,mismatches,master_path,compact_path


def load_manifests(cfg, master, log):
    current=normalize_manifest(pd.read_csv(REPO/Path(cfg["data"]["current_manifest"])),master,"current")
    strict=normalize_manifest(pd.read_csv(REPO/Path(cfg["data"]["strict_manifest"])),master,"strict")
    # Current overlap is expected; strict overlap must be zero.
    cross_current=int((current.groupby("canonical_group")["split"].nunique()>1).sum())
    cross_strict=int((strict.groupby("canonical_group")["split"].nunique()>1).sum())
    if cross_strict!=0: raise ValueError(f"Strict manifest has {cross_strict} cross-split canonical groups")
    log.info(f"[OK] current split counts: {current['split'].value_counts().to_dict()} | cross-split canonical groups={cross_current}")
    log.info(f"[OK] strict split counts:  {strict['split'].value_counts().to_dict()} | cross-split canonical groups=0")
    if cross_current != 598:
        raise ValueError(f"Frozen current split expected 598 cross-split canonical groups, found {cross_current}")
    return {"current":current,"strict":strict}, cross_current


def build_features(cfg, master, mols, pp, log, force=False):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, rdFingerprintGenerator

    fp_path=pp["cache"]/"morgan_r2_2048.npz"
    desc_path=pp["cache"]/"rdkit_descriptors.npz"
    n_bits=int(cfg["fingerprint"]["n_bits"]); radius=int(cfg["fingerprint"]["radius"])
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=n_bits,includeChirality=bool(cfg["fingerprint"].get("include_chirality",False)),useBondTypes=bool(cfg["fingerprint"].get("use_bond_types",True)))
    fps=[]
    X=np.zeros((len(mols),n_bits),dtype=np.uint8)
    for i,mol in enumerate(mols):
        fp=gen.GetFingerprint(mol); fps.append(fp)
        arr=np.zeros((n_bits,),dtype=np.uint8); DataStructs.ConvertToNumpyArray(fp,arr); X[i]=arr
    np.savez_compressed(fp_path,X=X,radius=radius,n_bits=n_bits)
    log.info(f"[OK] Morgan fingerprints: {X.shape[0]:,} × {X.shape[1]:,} (radius={radius})")

    funcmap=dict(Descriptors.descList)
    missing=[x for x in GENERAL_DESCRIPTORS if x not in funcmap]
    if missing: raise RuntimeError(f"RDKit version lacks required descriptors: {missing}")
    cols=GENERAL_DESCRIPTORS+CUSTOM_DESCRIPTORS
    D=np.full((len(mols),len(cols)),np.nan,dtype=np.float64)
    for i,mol in enumerate(mols):
        for j,name in enumerate(GENERAL_DESCRIPTORS):
            try: D[i,j]=float(funcmap[name](mol))
            except Exception: D[i,j]=np.nan
        j0=len(GENERAL_DESCRIPTORS)
        # Freeze HBA descriptor semantics to the historical NumHBA v2.0.1 SMARTS
        # so Phase-C feature generation is stable across RDKit 2025/2026.
        if i==0:
            _hba_frozen_query=Chem.MolFromSmarts(HBA_FROZEN_V201_SMARTS)
            if _hba_frozen_query is None:
                raise RuntimeError("Could not compile frozen NumHBA v2.0.1 SMARTS")
        D[i,j0+0]=len(mol.GetSubstructMatches(_hba_frozen_query,uniquify=True))
        D[i,j0+1]=sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        D[i,j0+2]=sum(a.GetFormalCharge() for a in mol.GetAtoms())
        D[i,j0+3]=sum(1 for b in mol.GetBonds() if b.GetBondTypeAsDouble()>1.5)
        D[i,j0+4]=float(D[i,j0+3]>0)
        if (i+1)%1000==0:
            log.info(f"      descriptors: {i+1:,}/{len(mols):,} molecules")
    D[~np.isfinite(D)]=np.nan
    np.savez_compressed(desc_path,X=D,columns=np.array(cols,dtype=object))
    log.info(f"[OK] RDKit descriptor matrix: {D.shape[0]:,} × {D.shape[1]:,}")
    return X,fps,D,cols


def split_indices(manifest, master):
    idxmap=pd.Series(np.arange(len(master)),index=master["image_name"].astype(str)).to_dict()
    out={}
    for s in ["train","validation","test"]:
        fns=manifest.loc[manifest["split"]==s,"filename"].astype(str).tolist()
        out[s]=np.array([idxmap[x] for x in fns],dtype=int)
    return out


def tune_thresholds(yv,pv,n_grid):
    th=[]; scores=[]
    for j in range(yv.shape[1]):
        t,s=threshold_by_f1(yv[:,j],pv[:,j],n_grid=n_grid); th.append(t); scores.append(s)
    return np.array(th,float),np.array(scores,float)


def save_model_outputs(model_name, scheme, manifest, master, idxs, yv, pv, yt, pt, thresholds, val_f1, pp, params, val_macro, grid_rows):
    # Validation/test metrics.
    per_v,ov_v,pred_v=metrics_from_probs(yv,pv,thresholds,LABEL_COLUMNS)
    per_t,ov_t,pred_t=metrics_from_probs(yt,pt,thresholds,LABEL_COLUMNS)
    rows=[]
    for eval_split, per, ov in [("validation",per_v,ov_v),("test",per_t,ov_t)]:
        per=per.copy(); per.insert(0,"model",model_name); per.insert(0,"split_scheme",scheme); per.insert(2,"evaluation_split",eval_split)
        rows.append((eval_split,per,ov))
    # Prediction tables.
    for eval_split, arr_idx, y, p, pred in [("validation",idxs["validation"],yv,pv,pred_v),("test",idxs["test"],yt,pt,pred_t)]:
        df=pd.DataFrame({
            "filename":master.iloc[arr_idx]["image_name"].values,
            "qmof_id":master.iloc[arr_idx]["qmof_id"].values,
            "canonical_first_linker_smiles":master.iloc[arr_idx]["canonical_derived"].values,
        })
        for j,c in enumerate(LABEL_COLUMNS):
            df[f"gt_{c}"]=y[:,j].astype(int); df[f"prob_{c}"]=p[:,j]; df[f"pred_{c}"]=pred[:,j].astype(int); df[f"thr_{c}"]=thresholds[j]
        df.to_csv(pp["source"]/f"{scheme}__{model_name}__{eval_split}_predictions.csv",index=False)
    # Unique-canonical-linker metrics (duplicate-aware second view).
    # Probabilities are averaged within canonical identity; deterministic labels
    # must be identical within each canonical group.
    for eval_split, arr_idx, y, p in [
        ("validation",idxs["validation"],yv,pv),
        ("test",idxs["test"],yt,pt),
    ]:
        tmp=pd.DataFrame({
            "canonical_first_linker_smiles":master.iloc[arr_idx]["canonical_derived"].values,
        })
        for j,c in enumerate(LABEL_COLUMNS):
            tmp[f"gt_{c}"]=y[:,j].astype(int)
            tmp[f"prob_{c}"]=p[:,j]
        agg_rows=[]
        for smi,g in tmp.groupby("canonical_first_linker_smiles",sort=False):
            row={"canonical_first_linker_smiles":smi,"n_records":len(g)}
            for j,c in enumerate(LABEL_COLUMNS):
                if g[f"gt_{c}"].nunique()!=1:
                    raise ValueError(f"Inconsistent {c} labels inside canonical group {smi}")
                row[f"gt_{c}"]=int(g[f"gt_{c}"].iloc[0])
                row[f"prob_{c}"]=float(g[f"prob_{c}"].mean())
            agg_rows.append(row)
        agg=pd.DataFrame(agg_rows)
        ay=np.column_stack([agg[f"gt_{c}"].values.astype(int) for c in LABEL_COLUMNS])
        ap=np.column_stack([agg[f"prob_{c}"].values.astype(float) for c in LABEL_COLUMNS])
        aper,aov,apred=metrics_from_probs(ay,ap,thresholds,LABEL_COLUMNS)
        aov_df=pd.DataFrame([{
            "split_scheme":scheme,"model":model_name,"evaluation_split":eval_split,
            "weighting":"unique_canonical",**aov
        }])
        aper=aper.copy()
        aper.insert(0,"model",model_name)
        aper.insert(0,"split_scheme",scheme)
        aper.insert(2,"evaluation_split",eval_split)
        aper.insert(3,"weighting","unique_canonical")
        aov_df.to_csv(pp["metrics"]/f"unique_canonical__{scheme}__{model_name}__{eval_split}_overall.csv",index=False)
        aper.to_csv(pp["metrics"]/f"unique_canonical__{scheme}__{model_name}__{eval_split}_per_label.csv",index=False)
        for j,c in enumerate(LABEL_COLUMNS):
            agg[f"pred_{c}"]=apred[:,j].astype(int)
            agg[f"thr_{c}"]=thresholds[j]
        agg.to_csv(pp["source"]/f"unique_canonical__{scheme}__{model_name}__{eval_split}_predictions.csv",index=False)

    # Metrics append-ready return.
    overall=[]; perall=[]
    for eval_split,per,ov in rows:
        rr={"split_scheme":scheme,"model":model_name,"evaluation_split":eval_split,**ov}
        overall.append(rr); perall.append(per)
    hp={"split_scheme":scheme,"model":model_name,"selection_split":"validation","selection_metric":"macro_auprc","selected_params":json.dumps(params,sort_keys=True),"selected_val_macro_auprc":val_macro}
    thr=pd.DataFrame({"split_scheme":scheme,"model":model_name,"label":LABEL_COLUMNS,"threshold":thresholds,"validation_f1":val_f1})
    return overall,perall,hp,thr,grid_rows


def morgan_logreg(X, master, manifest, idxs, profile, n_jobs, seed, n_grid, log):
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    train,val,test=idxs["train"],idxs["validation"],idxs["test"]
    y=master[LABEL_COLUMNS].values.astype(int)
    candidates = ([{"C":1.0,"class_weight":None}] if profile=="smoke" else
                  [{"C":c,"class_weight":cw} for c in [0.1,1.0,10.0] for cw in [None,"balanced"]])
    grid=[]; best=None
    for k,p in enumerate(candidates,1):
        log.info(f"      LR candidate {k}/{len(candidates)}: C={p['C']}, class_weight={p['class_weight']}")
        est=LogisticRegression(C=p["C"],class_weight=p["class_weight"],solver="liblinear",max_iter=2000,tol=1e-4,random_state=seed)
        clf=OneVsRestClassifier(est,n_jobs=n_jobs)
        clf.fit(X[train],y[train]); pv=clf.predict_proba(X[val]); score=macro_auprc(y[val],pv)
        grid.append({**p,"val_macro_auprc":score})
        if best is None or score>best[0]: best=(score,p,clf,pv)
    score,p,clf,pv=best; pt=clf.predict_proba(X[test]); th,vf=tune_thresholds(y[val],pv,n_grid)
    return y[val],pv,y[test],pt,th,vf,p,score,grid


def morgan_rf(X, master, manifest, idxs, profile, n_jobs, seed, n_grid, log):
    """
    Independent per-label random forests.

    We intentionally avoid a multi-output RandomForestClassifier with a single
    string class_weight because scikit-learn combines class weights across
    outputs. For this imbalanced multi-label benchmark, independent binary RFs
    make the weighting policy explicit and easier to audit.
    """
    from joblib import Parallel, delayed
    from sklearn.ensemble import RandomForestClassifier

    train,val,test=idxs["train"],idxs["validation"],idxs["test"]
    y=master[LABEL_COLUMNS].values.astype(int)
    if profile=="smoke":
        candidates=[{"n_estimators":40,"max_depth":16,"min_samples_leaf":1,"max_features":"sqrt"}]
    else:
        candidates=[
          {"n_estimators":200,"max_depth":None,"min_samples_leaf":1,"max_features":"sqrt"},
          {"n_estimators":200,"max_depth":24,"min_samples_leaf":1,"max_features":"sqrt"},
          {"n_estimators":200,"max_depth":None,"min_samples_leaf":2,"max_features":"sqrt"},
        ]

    def fit_one(label_idx, params):
        clf=RandomForestClassifier(
            **params,
            class_weight="balanced_subsample",
            random_state=seed+101*label_idx,
            n_jobs=1,
        )
        clf.fit(X[train],y[train,label_idx])
        pv=clf.predict_proba(X[val])[:,1]
        return clf,pv

    grid=[]; best_score=-1; best_params=None
    for k,p in enumerate(candidates,1):
        log.info(f"      RF candidate {k}/{len(candidates)}: {p}")
        fitted=Parallel(n_jobs=n_jobs,prefer="threads")(
            delayed(fit_one)(j,p) for j in range(len(LABEL_COLUMNS))
        )
        pv=np.column_stack([x[1] for x in fitted])
        score=macro_auprc(y[val],pv)
        grid.append({**p,"class_weight":"balanced_subsample","fit_mode":"independent_per_label","val_macro_auprc":score})
        if score>best_score:
            best_score=score; best_params=dict(p)

    log.info("      RF refit selected independent per-label configuration")
    fitted=Parallel(n_jobs=n_jobs,prefer="threads")(
        delayed(fit_one)(j,best_params) for j in range(len(LABEL_COLUMNS))
    )
    models=[x[0] for x in fitted]
    pv=np.column_stack([x[1] for x in fitted])
    pt=np.column_stack([models[j].predict_proba(X[test])[:,1] for j in range(len(LABEL_COLUMNS))])
    th,vf=tune_thresholds(y[val],pv,n_grid)
    params={**best_params,"class_weight":"balanced_subsample","fit_mode":"independent_per_label"}
    return y[val],pv,y[test],pt,th,vf,params,best_score,grid


def descriptor_lro(D, desc_cols, master, idxs, profile, seed, n_grid, log):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    train,val,test=idxs["train"],idxs["validation"],idxs["test"]
    y=master[LABEL_COLUMNS].values.astype(int)
    candidates = ([{"C":1.0,"class_weight":None}] if profile=="smoke" else
                  [{"C":c,"class_weight":cw} for c in [0.1,1.0,10.0] for cw in [None,"balanced"]])
    col_index={c:i for i,c in enumerate(desc_cols)}
    masks={label:np.array([i for i,c in enumerate(desc_cols) if c not in LRO_EXCLUDE[label]],dtype=int) for label in LABEL_COLUMNS}
    grid=[]; best_score=-1; best_params=None
    for k,p in enumerate(candidates,1):
        log.info(f"      Descriptor LR candidate {k}/{len(candidates)}: C={p['C']}, class_weight={p['class_weight']}")
        pv=np.zeros((len(val),len(LABEL_COLUMNS)),float)
        for j,label in enumerate(LABEL_COLUMNS):
            use=masks[label]
            clf=Pipeline([
                ("impute",SimpleImputer(strategy="median")),
                ("scale",StandardScaler()),
                ("lr",LogisticRegression(C=p["C"],class_weight=p["class_weight"],solver="lbfgs",max_iter=1000,tol=1e-4,random_state=seed+37*j)),
            ])
            clf.fit(D[train][:,use],y[train,j]); pv[:,j]=clf.predict_proba(D[val][:,use])[:,1]
        score=macro_auprc(y[val],pv); grid.append({**p,"val_macro_auprc":score})
        if score>best_score: best_score=score; best_params=p
    pv=np.zeros((len(val),len(LABEL_COLUMNS)),float); pt=np.zeros((len(test),len(LABEL_COLUMNS)),float)
    for j,label in enumerate(LABEL_COLUMNS):
        use=masks[label]
        log.info(f"      Descriptor selected fit {j+1}/10: {label} | excluded={sorted(LRO_EXCLUDE[label])}")
        clf=Pipeline([
            ("impute",SimpleImputer(strategy="median")),
            ("scale",StandardScaler()),
            ("lr",LogisticRegression(C=best_params["C"],class_weight=best_params["class_weight"],solver="lbfgs",max_iter=1000,tol=1e-4,random_state=seed+37*j)),
        ])
        clf.fit(D[train][:,use],y[train,j]); pv[:,j]=clf.predict_proba(D[val][:,use])[:,1]; pt[:,j]=clf.predict_proba(D[test][:,use])[:,1]
    th,vf=tune_thresholds(y[val],pv,n_grid)
    return y[val],pv,y[test],pt,th,vf,best_params,best_score,grid,masks


def tanimoto_audit(fps, master, manifest, idxs, scheme, pp, log):
    from rdkit import DataStructs
    tr=idxs["train"]; te=idxs["test"]
    train_fps=[fps[i] for i in tr]
    rows=[]
    for pos,i in enumerate(te):
        sims=DataStructs.BulkTanimotoSimilarity(fps[i],train_fps)
        j=int(np.argmax(sims)); train_i=int(tr[j]); sim=float(sims[j])
        rows.append({
            "filename":master.iloc[i]["image_name"],
            "qmof_id":master.iloc[i]["qmof_id"],
            "canonical_first_linker_smiles":master.iloc[i]["canonical_derived"],
            "nearest_train_filename":master.iloc[train_i]["image_name"],
            "nearest_train_qmof_id":master.iloc[train_i]["qmof_id"],
            "nearest_train_canonical_first_linker_smiles":master.iloc[train_i]["canonical_derived"],
            "max_tanimoto":sim,
            "exact_canonical_match":bool(master.iloc[i]["canonical_derived"]==master.iloc[train_i]["canonical_derived"]),
        })
        if (pos+1)%400==0: log.info(f"      Tanimoto {scheme}: {pos+1}/{len(te)} test samples")
    df=pd.DataFrame(rows)
    df.to_csv(pp["source"]/f"tanimoto_{scheme}_test_nearest.csv",index=False)
    # Canonical-level view prevents duplicate records inside a split from
    # over-weighting the similarity distribution.
    cdf=(df.sort_values("filename")
           .drop_duplicates("canonical_first_linker_smiles")
           .reset_index(drop=True))
    cdf.to_csv(pp["source"]/f"tanimoto_{scheme}_canonical_test_nearest.csv",index=False)
    s=df["max_tanimoto"]; cs=cdf["max_tanimoto"]
    summary={
        "split_scheme":scheme,"n_test":len(df),"n_unique_test_canonical":len(cdf),
        "mean":float(s.mean()),"median":float(s.median()),
        "canonical_mean":float(cs.mean()),"canonical_median":float(cs.median()),
        "p10":float(s.quantile(.10)),"p25":float(s.quantile(.25)),"p75":float(s.quantile(.75)),"p90":float(s.quantile(.90)),
        "fraction_eq_1":float((s>=1-1e-12).mean()),"fraction_ge_0p9":float((s>=.9).mean()),
        "fraction_ge_0p8":float((s>=.8).mean()),"fraction_ge_0p7":float((s>=.7).mean()),
        "exact_canonical_matches":int(df["exact_canonical_match"].sum()),
    }
    if scheme=="strict" and summary["exact_canonical_matches"]!=0:
        raise ValueError("Strict Tanimoto audit found exact canonical identity in training")
    return df,summary


def bootstrap_historical(cfg, current_manifest, pp, n_boot, log):
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score
    pred_path=REPO/Path(cfg["data"]["historical_test_predictions"])
    if not pred_path.exists(): raise FileNotFoundError(pred_path)
    pred=pd.read_csv(pred_path)
    meta=current_manifest[["filename","canonical_group"]].drop_duplicates("filename")
    pred=pred.merge(meta,on="filename",how="left",validate="one_to_one")
    if pred["canonical_group"].isna().any(): raise ValueError("Historical predictions could not all be assigned canonical groups")
    # Historical display-name columns.
    display=["Hydrophobicity","Polarity","Flexibility","Aromaticity","Hydrogen-Bond Donors","Hydrogen-Bond Acceptors","Planarity (Single-Ring)","Macrocycle (>6 Rings)","Charge Present","Conjugation Present"]
    gt=np.column_stack([pred[f"gt_{x}"].values.astype(int) for x in display])
    prob=np.column_stack([pred[f"prob_{x}"].values.astype(float) for x in display])
    binary=np.column_stack([pred[f"pred_{x}"].values.astype(int) for x in display])
    groups=pred["canonical_group"].astype(str).values
    uniq=np.array(pd.unique(groups),dtype=object)
    idx_by={g:np.where(groups==g)[0] for g in uniq}
    rng=np.random.default_rng(int(cfg["random_seeds"]["bootstrap"]))
    draws=[]
    for b in range(n_boot):
        picked=rng.choice(uniq,size=len(uniq),replace=True)
        ix=np.concatenate([idx_by[g] for g in picked])
        y=gt[ix]; p=prob[ix]; pr=binary[ix]
        aps=[]; aucs=[]
        for j in range(y.shape[1]):
            aps.append(average_precision_score(y[:,j],p[:,j]) if y[:,j].sum() else np.nan)
            aucs.append(roc_auc_score(y[:,j],p[:,j]) if len(np.unique(y[:,j]))>1 else np.nan)
        draws.append({
            "replicate":b,
            "macro_auprc":float(np.nanmean(aps)),
            "macro_auroc":float(np.nanmean(aucs)),
            "macro_f1":float(np.mean([f1_score(y[:,j],pr[:,j],zero_division=0) for j in range(y.shape[1])])),
            "micro_f1":float(f1_score(y,pr,average="micro",zero_division=0)),
            "subset_accuracy":float(accuracy_score(y,pr)),
            "hamming_loss":float(np.mean(y!=pr)),
        })
        if (b+1)%500==0: log.info(f"      Cluster bootstrap: {b+1}/{n_boot}")
    d=pd.DataFrame(draws); d.to_csv(pp["bootstrap"]/"historical_vista_current_cluster_bootstrap_draws.csv",index=False)
    alpha=.025; rows=[]
    # Point estimate from original test set.
    point={
        "macro_auprc":float(np.mean([average_precision_score(gt[:,j],prob[:,j]) for j in range(gt.shape[1])])),
        "macro_auroc":float(np.mean([roc_auc_score(gt[:,j],prob[:,j]) for j in range(gt.shape[1])])),
        "macro_f1":float(np.mean([f1_score(gt[:,j],binary[:,j],zero_division=0) for j in range(gt.shape[1])])),
        "micro_f1":float(f1_score(gt,binary,average="micro",zero_division=0)),
        "subset_accuracy":float(accuracy_score(gt,binary)),
        "hamming_loss":float(np.mean(gt!=binary)),
    }
    for m,v in point.items():
        rows.append({"metric":m,"point_estimate":v,"bootstrap_mean":float(d[m].mean()),"ci_low":float(d[m].quantile(alpha)),"ci_high":float(d[m].quantile(1-alpha)),"n_resamples":n_boot,"n_unique_test_canonical_groups":len(uniq)})
    s=pd.DataFrame(rows); s.to_csv(pp["bootstrap"]/"historical_vista_current_cluster_bootstrap_summary.csv",index=False)
    return s


def plot_results(overall_df, tanimoto_frames, bootstrap_summary, pp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    try:
        font_manager.findfont("Arial",fallback_to_default=False); _font="Arial"
    except Exception:
        _font="DejaVu Sans"
    plt.rcParams.update({"font.family":_font,"font.size":10,"axes.titlesize":11,"axes.labelsize":10,"pdf.fonttype":42,"ps.fonttype":42})
    # 1: CPU baseline macro AUPRC current vs strict.
    xmodels=["morgan_logreg","morgan_rf","rdkit_descriptor_lro"]
    fig,ax=plt.subplots(figsize=(7.2,4.0),constrained_layout=True)
    x=np.arange(len(xmodels)); w=.36
    cur=[float(overall_df[(overall_df.split_scheme=="current")&(overall_df.model==m)&(overall_df.evaluation_split=="test")].macro_auprc.iloc[0]) for m in xmodels]
    st=[float(overall_df[(overall_df.split_scheme=="strict")&(overall_df.model==m)&(overall_df.evaluation_split=="test")].macro_auprc.iloc[0]) for m in xmodels]
    ax.bar(x-w/2,cur,width=w,label="Current sample split"); ax.bar(x+w/2,st,width=w,label="Canonical-disjoint split")
    ax.set_xticks(x); ax.set_xticklabels(["Morgan LR","Morgan RF","RDKit LRO"]); ax.set_ylabel("Macro AUPRC"); ax.set_ylim(0,1.02); ax.legend(frameon=False); ax.set_title("CPU baselines under current vs canonical-disjoint split")
    fig.savefig(pp["figures"]/"cpu_baselines_macro_auprc.pdf"); fig.savefig(pp["figures"]/"cpu_baselines_macro_auprc.png",dpi=300); plt.close(fig)
    pd.DataFrame({"model":xmodels,"current_macro_auprc":cur,"strict_macro_auprc":st}).to_csv(pp["source"]/"figure_cpu_baselines_macro_auprc.csv",index=False)
    # 2: Tanimoto distributions.
    fig,ax=plt.subplots(figsize=(7.2,4.0),constrained_layout=True)
    bins=np.linspace(0,1,31)
    for scheme,df in tanimoto_frames.items(): ax.hist(df["max_tanimoto"],bins=bins,density=True,histtype="step",linewidth=1.8,label=scheme.capitalize())
    ax.set_xlabel("Maximum training-set ECFP4 Tanimoto similarity"); ax.set_ylabel("Density"); ax.set_xlim(0,1); ax.legend(frameon=False); ax.set_title("Nearest-training similarity audit")
    fig.savefig(pp["figures"]/"tanimoto_current_vs_strict.pdf"); fig.savefig(pp["figures"]/"tanimoto_current_vs_strict.png",dpi=300); plt.close(fig)
    # 3: Current historical VISTA bootstrap interval.
    fig,ax=plt.subplots(figsize=(6.4,3.6),constrained_layout=True)
    b=bootstrap_summary.set_index("metric").loc[["macro_auprc","macro_auroc","macro_f1","micro_f1"]]
    y=np.arange(len(b)); vals=b["point_estimate"].values; lo=vals-b["ci_low"].values; hi=b["ci_high"].values-vals
    ax.errorbar(vals,y,xerr=np.vstack([lo,hi]),fmt="o",capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(["Macro AUPRC","Macro AUROC","Macro F1","Micro F1"]); ax.set_xlim(0.85,1.005); ax.set_xlabel("Metric value"); ax.set_title("Historical VISTA: canonical-linker cluster bootstrap")
    fig.savefig(pp["figures"]/"historical_vista_cluster_bootstrap.pdf"); fig.savefig(pp["figures"]/"historical_vista_cluster_bootstrap.png",dpi=300); plt.close(fig)


def build_summary(overall_df, per_df, hp_df, tan_summary, boot, audit, pp):
    test=overall_df[overall_df.evaluation_split=="test"].copy()
    lines=["# VISTA Phase C Results Summary","",f"**Status:** PASS","", "## Data and rule audit", "", f"- Samples: {audit['n_samples']:,}", f"- Unique canonical first linkers: {audit['unique_canonical_first_linkers']:,}", "- Master metadata and compact `Data.csv` agree exactly.", "- All first-linker SMILES parse with RDKit.", "- All ten deterministic label rules regenerate the frozen labels with zero mismatch.", "", "## CPU baseline test results", "", "| Split | Model | Macro AUPRC | Macro AUROC | Macro F1 | Micro F1 |", "|---|---|---:|---:|---:|---:|"]
    for _,r in test.sort_values(["split_scheme","model"]).iterrows():
        lines.append(f"| {r['split_scheme']} | {r['model']} | {r['macro_auprc']:.4f} | {r['macro_auroc']:.4f} | {r['macro_f1']:.4f} | {r['micro_f1']:.4f} |")
    lines += ["", "The RDKit rule-access reconstruction is an upper-bound/label-definition audit and is **not** treated as an independent ML baseline.", "", "## Split sensitivity", ""]
    for model in ["morgan_logreg","morgan_rf","rdkit_descriptor_lro"]:
        c=float(test[(test.split_scheme=="current")&(test.model==model)].macro_auprc.iloc[0]); s=float(test[(test.split_scheme=="strict")&(test.model==model)].macro_auprc.iloc[0])
        lines.append(f"- `{model}`: strict minus current macro-AUPRC = {s-c:+.4f}.")
    lines += ["", "## Nearest-training ECFP4 similarity", ""]
    for r in tan_summary:
        lines.append(f"- **{r['split_scheme']}**: canonical-level median max Tanimoto {r['canonical_median']:.3f}; sample-weighted median {r['median']:.3f}; exact canonical matches {r['exact_canonical_matches']}; fraction similarity=1 {r['fraction_eq_1']:.3%}.")
    lines += ["", "## Historical VISTA canonical-linker cluster bootstrap", ""]
    for _,r in boot.iterrows():
        lines.append(f"- {r['metric']}: {r['point_estimate']:.4f} (95% cluster-bootstrap CI {r['ci_low']:.4f}–{r['ci_high']:.4f}).")
    lines += ["", "## Scientific interpretation boundary", "", "Phase C measures how conventional 2D molecular representations behave under the historical and canonical-disjoint splits. It does not replace the pending strict VISTA GPU reruns. Any superiority claim between VISTA and the new CPU baselines should wait until the strict VISTA seeds are returned from EPFL.", ""]
    (pp["report"]/"PHASE_C_RESULTS_SUMMARY.md").write_text("\n".join(lines),encoding="utf-8")


def final_gate(cfg, pp, audit, tan_summary, log):
    required=[
        pp["metrics"]/"baseline_metrics_overall.csv", pp["metrics"]/"baseline_metrics_per_label.csv",
        pp["metrics"]/"selected_hyperparameters.csv", pp["metrics"]/"validation_thresholds.csv",
        pp["metrics"]/"validation_grid_scores.csv", pp["metrics"]/"tanimoto_summary.csv",
        pp["bootstrap"]/"historical_vista_current_cluster_bootstrap_summary.csv",
        pp["metrics"]/"unique_canonical__strict__morgan_logreg__test_overall.csv",
        pp["metrics"]/"unique_canonical__strict__morgan_rf__test_overall.csv",
        pp["metrics"]/"unique_canonical__strict__rdkit_descriptor_lro__test_overall.csv",
        pp["report"]/"PHASE_C_RESULTS_SUMMARY.md",
    ]
    missing=[str(p) for p in required if not p.exists()]
    errors=[]
    if missing: errors.append("missing files: "+", ".join(missing))
    if any(audit["rule_mismatches"].values()): errors.append("rule mismatch nonzero")
    strict=[r for r in tan_summary if r["split_scheme"]=="strict"][0]
    if strict["exact_canonical_matches"]!=0: errors.append("strict Tanimoto exact canonical match nonzero")
    status={"status":"PASS" if not errors else "FAIL","errors":errors,"gpu_required":False,"torch_required":False}
    write_json(status,pp["report"]/"PHASE_C_STATUS.json")
    if errors:
        log.info("PHASE C FINAL GATE: FAIL"); [log.info("  - "+e) for e in errors]; raise RuntimeError("Phase C final gate failed")
    log.info("PHASE C FINAL GATE: PASS")
    log.info("CPU scientific strengthening completed; no PyTorch/GPU work was performed.")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--profile",choices=["smoke","standard"],default="standard")
    ap.add_argument("--stage",choices=["preflight","all"],default="all")
    ap.add_argument("--n-jobs",type=int,default=0,help="0 = automatic, capped at 8 workers")
    args=ap.parse_args()
    cfg=load_config(); pp=paths(cfg); ensure_dirs(pp); log=setup_logger(pp["logs"]/"phase_c.log")
    total=8
    stage(log,1,total,"Environment and frozen-input preflight")
    env=environment_info(); write_json(env,pp["audit"]/"environment.json")
    if env.get("conda_default_env") not in (None,"",cfg["environment_name"]):
        log.info(f"[WARN] CONDA_DEFAULT_ENV={env.get('conda_default_env')} (expected {cfg['environment_name']})")
    log.info(f"[OK] Python: {sys.version.split()[0]} | RDKit: {env['rdkit']} | sklearn: {env['sklearn']}")
    master,mols,mismatches,master_path,compact_path=load_master_and_molecules(cfg,log)
    manifests,cross_current=load_manifests(cfg,master,log)
    audit={
        "n_samples":len(master),"unique_image_names":int(master["image_name"].nunique()),
        "unique_canonical_first_linkers":int(master["canonical_derived"].nunique()),
        "rule_mismatches":mismatches,"current_cross_split_canonical_groups":cross_current,
        "strict_cross_split_canonical_groups":0,
        "input_sha256":{"master":sha256(master_path),"compact_labels":sha256(compact_path),"current_manifest":sha256(REPO/Path(cfg['data']['current_manifest'])),"strict_manifest":sha256(REPO/Path(cfg['data']['strict_manifest']))},
    }
    write_json(audit,pp["audit"]/"input_data_audit.json")
    pd.DataFrame([{"label":c,"mismatches":mismatches[c]} for c in LABEL_COLUMNS]).to_csv(pp["audit"]/"rule_oracle_mismatch_summary.csv",index=False)
    if args.stage=="preflight":
        log.info("PHASE C PREFLIGHT: PASS")
        return

    n_jobs=args.n_jobs if args.n_jobs>0 else min(8,max(1,(os.cpu_count() or 2)-2))
    log.info(f"[INFO] CPU parallelism: n_jobs={n_jobs}")
    stage(log,2,total,"ECFP4/Morgan and RDKit descriptor feature construction")
    X,fps,D,desc_cols=build_features(cfg,master,mols,pp,log)

    all_over=[]; all_per=[]; all_hp=[]; all_thr=[]; all_grid=[]
    n_grid=int(cfg["model_selection"]["threshold_grid_points"]); seed=int(cfg["random_seeds"]["models"])
    for scheme,manifest in manifests.items():
        idxs=split_indices(manifest,master)
        stage(log,3 if scheme=="current" else 4,total,f"CPU baselines — {scheme} split")
        log.info("   [A] Morgan Logistic Regression")
        res=morgan_logreg(X,master,manifest,idxs,args.profile,n_jobs,seed,n_grid,log)
        yv,pv,yt,pt,th,vf,params,val_macro,grid=res
        o,p,h,t,g=save_model_outputs("morgan_logreg",scheme,manifest,master,idxs,yv,pv,yt,pt,th,vf,pp,params,val_macro,grid)
        all_over+=o; all_per+=p; all_hp.append(h); all_thr.append(t); all_grid += [{"split_scheme":scheme,"model":"morgan_logreg",**x} for x in g]
        log.info(f"      [OK] selected validation macro-AUPRC={val_macro:.4f} | params={params}")

        log.info("   [B] Morgan Random Forest")
        res=morgan_rf(X,master,manifest,idxs,args.profile,n_jobs,seed,n_grid,log)
        yv,pv,yt,pt,th,vf,params,val_macro,grid=res
        o,p,h,t,g=save_model_outputs("morgan_rf",scheme,manifest,master,idxs,yv,pv,yt,pt,th,vf,pp,params,val_macro,grid)
        all_over+=o; all_per+=p; all_hp.append(h); all_thr.append(t); all_grid += [{"split_scheme":scheme,"model":"morgan_rf",**x} for x in g]
        log.info(f"      [OK] selected validation macro-AUPRC={val_macro:.4f} | params={params}")

        log.info("   [C] RDKit leave-rule-out descriptor Logistic Regression")
        res=descriptor_lro(D,desc_cols,master,idxs,args.profile,seed,n_grid,log)
        yv,pv,yt,pt,th,vf,params,val_macro,grid,masks=res
        o,p,h,t,g=save_model_outputs("rdkit_descriptor_lro",scheme,manifest,master,idxs,yv,pv,yt,pt,th,vf,pp,params,val_macro,grid)
        all_over+=o; all_per+=p; all_hp.append(h); all_thr.append(t); all_grid += [{"split_scheme":scheme,"model":"rdkit_descriptor_lro",**x} for x in g]
        log.info(f"      [OK] selected validation macro-AUPRC={val_macro:.4f} | params={params}")

    overall_df=pd.DataFrame(all_over); per_df=pd.concat(all_per,ignore_index=True); hp_df=pd.DataFrame(all_hp); thr_df=pd.concat(all_thr,ignore_index=True); grid_df=pd.DataFrame(all_grid)
    # Append the frozen historical VISTA current-split test reference for direct context.
    ref_over=REPO/"reference_current_results/metrics/test_metrics_overall.csv"
    ref_per=REPO/"reference_current_results/metrics/test_metrics_per_label.csv"
    if ref_over.exists():
        rr=pd.read_csv(ref_over).iloc[0].to_dict(); rr={"split_scheme":"current","model":"historical_vista_reference","evaluation_split":"test",**rr}
        overall_df=pd.concat([overall_df,pd.DataFrame([rr])],ignore_index=True,sort=False)
    if ref_per.exists():
        rp=pd.read_csv(ref_per).copy(); display_to_machine={v:k for k,v in {
            "hydrophobicity":"Hydrophobicity","polarity":"Polarity","flexibility":"Flexibility","aromaticity":"Aromaticity",
            "h_bond_donors":"Hydrogen-Bond Donors","h_bond_acceptors":"Hydrogen-Bond Acceptors",
            "planarity_single_ring":"Planarity (Single-Ring)","macrocycle_gt6_rings":"Macrocycle (>6 Rings)",
            "charge_present":"Charge Present","conjugation_present":"Conjugation Present"}.items()}
        rp["label"]=rp["label"].map(display_to_machine).fillna(rp["label"]); rp.insert(0,"model","historical_vista_reference"); rp.insert(0,"split_scheme","current"); rp.insert(2,"evaluation_split","test")
        per_df=pd.concat([per_df,rp],ignore_index=True,sort=False)
    overall_df.to_csv(pp["metrics"]/"baseline_metrics_overall.csv",index=False); per_df.to_csv(pp["metrics"]/"baseline_metrics_per_label.csv",index=False); hp_df.to_csv(pp["metrics"]/"selected_hyperparameters.csv",index=False); thr_df.to_csv(pp["metrics"]/"validation_thresholds.csv",index=False); grid_df.to_csv(pp["metrics"]/"validation_grid_scores.csv",index=False)
    exclusions=pd.DataFrame([{"label":k,"excluded_descriptors":";".join(sorted(v))} for k,v in LRO_EXCLUDE.items()]); exclusions.to_csv(pp["metrics"]/"descriptor_lro_exclusions.csv",index=False)

    stage(log,5,total,"Nearest-training ECFP4 Tanimoto audit")
    tan_frames={}; tan_summary=[]
    for scheme,manifest in manifests.items():
        idxs=split_indices(manifest,master); df,s=tanimoto_audit(fps,master,manifest,idxs,scheme,pp,log); tan_frames[scheme]=df; tan_summary.append(s); log.info(f"      [OK] {scheme}: median={s['median']:.3f}, exact canonical matches={s['exact_canonical_matches']}")
    pd.DataFrame(tan_summary).to_csv(pp["metrics"]/"tanimoto_summary.csv",index=False)

    stage(log,6,total,"Canonical-linker cluster bootstrap of historical VISTA test predictions")
    n_boot=100 if args.profile=="smoke" else int(cfg["bootstrap"]["n_resamples_standard"])
    boot=bootstrap_historical(cfg,manifests["current"],pp,n_boot,log)
    log.info(f"      [OK] bootstrap replicates={n_boot}")

    stage(log,7,total,"Figures and machine-readable summary")
    plot_results(overall_df,tan_frames,boot,pp)
    build_summary(overall_df,per_df,hp_df,tan_summary,boot,audit,pp)
    manifest={"phase":"C","profile":args.profile,"git_commit":git_commit(),"environment":environment_info(),"config":cfg,"input_audit":audit,"generated_at_unix":time.time()}
    write_json(manifest,pp["report"]/"PHASE_C_RUN_MANIFEST.json")
    # Compact checksums of committed scientific outputs (not caches/logs).
    files=[]
    for sub in [pp["audit"],pp["metrics"],pp["source"],pp["bootstrap"],pp["figures"],pp["report"]]:
        for p in sorted(sub.rglob("*")):
            if p.is_file(): files.append({"path":str(p.relative_to(REPO)).replace("\\","/"),"sha256":sha256(p),"bytes":p.stat().st_size})
    pd.DataFrame(files).to_csv(pp["report"]/"PHASE_C_OUTPUT_MANIFEST.csv",index=False)

    stage(log,8,total,"Phase-C final gate")
    final_gate(cfg,pp,audit,tan_summary,log)
    log.info(f"Results summary: {pp['report']/ 'PHASE_C_RESULTS_SUMMARY.md'}")

if __name__=="__main__":
    main()
