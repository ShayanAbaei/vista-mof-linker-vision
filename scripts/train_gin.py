#!/usr/bin/env python3
from __future__ import annotations

import argparse, itertools, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch, yaml
from sklearn.metrics import average_precision_score
from torch_geometric.loader import DataLoader

from _bootstrap import add_vista_legacy_path
add_vista_legacy_path()

import evaluation, seeding, experiments
from gin_graph import ManifestGraphDataset, GINMultiLabel
from _handover_utils import (
    LABEL_COLUMNS, LABEL_DISPLAY_NAMES, git_commit, load_frozen_manifest,
    save_system, sha256, write_json, run_readme
)

def macro_auprc(y,p):
    vals=[]
    for j in range(y.shape[1]):
        vals.append(average_precision_score(y[:,j],p[:,j]) if y[:,j].sum()>0 else np.nan)
    return float(np.nanmean(vals))

@torch.no_grad()
def collect(model,loader,device):
    model.eval(); probs=[]; ys=[]; idx=[]
    for batch in loader:
        batch=batch.to(device)
        logits=model(batch)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(batch.y.view(-1,10).cpu().numpy())
        idx.append(batch.row_index.view(-1).cpu().numpy())
    return np.concatenate(probs),np.concatenate(ys),np.concatenate(idx).astype(int)

def fit_one(train_frame,val_frame,params,cfg,seed,device,out_ck=None):
    seeding.set_all_seeds(seed,deterministic_torch=True)
    train_ds=ManifestGraphDataset(train_frame,LABEL_COLUMNS)
    val_ds=ManifestGraphDataset(val_frame,LABEL_COLUMNS)
    bs=int(cfg["training"]["batch_size"]); nw=int(cfg["training"]["num_workers"])
    tr=DataLoader(train_ds,batch_size=bs,shuffle=True,num_workers=nw)
    va=DataLoader(val_ds,batch_size=bs,shuffle=False,num_workers=nw)
    in_dim=train_ds[0].x.shape[1]
    model=GINMultiLabel(in_dim,int(params["hidden_dim"]),int(params["n_layers"]),float(params["dropout"]),10).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=float(params["learning_rate"]),weight_decay=float(cfg["training"]["weight_decay"]))
    freqs=train_frame[LABEL_COLUMNS].mean(axis=0).tolist()
    criterion=experiments.build_criterion(freqs,loss_type="asl").to(device)
    best=-np.inf; best_epoch=0; patience=0; history=[]; best_state=None
    for epoch in range(1,int(cfg["training"]["epochs"])+1):
        model.train(); losses=[]
        for batch in tr:
            batch=batch.to(device); opt.zero_grad()
            logits=model(batch); y=batch.y.view(-1,10)
            loss=criterion(logits,y)
            loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        pv,yv,_=collect(model,va,device)
        score=macro_auprc(yv,pv)
        history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"val_macro_auprc":score})
        if score>best+1e-12:
            best=score;best_epoch=epoch;patience=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            patience+=1
        if patience>=int(cfg["training"]["patience"]): break
    model.load_state_dict(best_state)
    if out_ck:
        torch.save({"model_state_dict":best_state,"params":params,"best_epoch":best_epoch,"val_macro_auprc":best},out_ck)
    return model,best,best_epoch,pd.DataFrame(history)

def pred_table(frame,probs,targets,thresholds):
    d={
        "split":frame["split"].tolist(),
        "sample_id":[Path(x).stem for x in frame["filename"]],
        "filename":frame["filename"].tolist(),
    }
    for c in ["qmof_id","canonical_first_linker_smiles","group_first_linker","row_index"]:
        if c in frame.columns: d[c]=frame[c].tolist()
    pred=evaluation.apply_thresholds(probs,thresholds)
    for j,n in enumerate(LABEL_DISPLAY_NAMES):
        d[f"gt_{n}"]=targets[:,j].astype(int)
        d[f"prob_{n}"]=probs[:,j]
        d[f"pred_{n}"]=pred[:,j]
        d[f"thr_{n}"]=float(thresholds[j])
    return pd.DataFrame(d)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--mode",choices=["tune","final"],required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--seed",type=int,default=None)
    ap.add_argument("--selected-config",default=None)
    args=ap.parse_args()

    repo=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load(Path(args.config).read_text())
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    save_system(out)
    (out/"resolved_config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
    (out/"command.txt").write_text(" ".join(__import__("shlex").quote(x) for x in __import__("sys").argv)+"\n",encoding="utf-8")
    (out/"git_commit.txt").write_text(str(git_commit(repo))+"\n",encoding="utf-8")
    manifest_path,df=load_frozen_manifest(repo,cfg["data"]["manifest"],True)
    train=df[df.split=="train"].reset_index(drop=True)
    val=df[df.split=="validation"].reset_index(drop=True)
    test=df[df.split=="test"].reset_index(drop=True)

    if args.mode=="tune":
        seed=int(cfg["run"]["tuning_seed"])
        grid=cfg["tuning_grid"]; rows=[]; best=None
        combos=list(itertools.product(grid["n_layers"],grid["hidden_dim"],grid["dropout"],grid["learning_rate"]))
        print(f"GIN validation grid: {len(combos)} candidates; TEST IS NOT EVALUATED.")
        for k,(nl,hd,dr,lr) in enumerate(combos,1):
            params={"n_layers":nl,"hidden_dim":hd,"dropout":dr,"learning_rate":lr}
            print(f"[{k}/{len(combos)}] {params}")
            model,score,ep,hist=fit_one(train,val,params,cfg,seed,args.device)
            rows.append({**params,"best_val_macro_auprc":score,"best_epoch":ep})
            if best is None or score>best[0]:
                best=(score,params,ep)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(out/"validation_grid_scores.csv",index=False)
        selected={"selected_by":"validation_macro_auprc","tuning_seed":seed,
                  "params":best[1],"best_val_macro_auprc":best[0],"best_epoch":best[2]}
        write_json(out/"selected_gin_config.json",selected)
        write_json(out/"runtime.json",{"mode":"tune","manifest_sha256":sha256(manifest_path),
                                      "git_commit":git_commit(repo),"test_evaluated":False})
        print("PASS — GIN validation-only tuning complete:",selected)
        return

    if not args.selected_config:
        raise RuntimeError("--selected-config is required in final mode")
    selected=json.loads(Path(args.selected_config).read_text())
    params=selected["params"]
    seed=int(args.seed if args.seed is not None else 42)
    t0=time.time()
    model,score,ep,hist=fit_one(train,val,params,cfg,seed,args.device,out/"best_checkpoint.pt")
    hist.to_csv(out/"history.csv",index=False)
    (out/"best_epoch.txt").write_text(str(ep)+"\n",encoding="utf-8")

    bs=int(cfg["training"]["batch_size"]); nw=int(cfg["training"]["num_workers"])
    val_loader=DataLoader(ManifestGraphDataset(val,LABEL_COLUMNS),batch_size=bs,shuffle=False,num_workers=nw)
    test_loader=DataLoader(ManifestGraphDataset(test,LABEL_COLUMNS),batch_size=bs,shuffle=False,num_workers=nw)
    pv,yv,iv=collect(model,val_loader,args.device)
    pt,yt,it=collect(model,test_loader,args.device)

    sel=evaluation.select_thresholds(
        pv,yv,LABEL_DISPLAY_NAMES,
        objective=cfg["evaluation"]["threshold_objective"],
        n_grid=int(cfg["evaluation"]["threshold_grid_points"])
    )
    thr=sel["thresholds"]
    evaluation.save_thresholds(thr,LABEL_DISPLAY_NAMES,out/"thresholds.csv",sel["scores"],cfg["evaluation"]["threshold_objective"])

    for split,frame,probs,tgts in [("validation",val,pv,yv),("test",test,pt,yt)]:
        pred_table(frame,probs,tgts,thr).to_csv(out/f"{split}_predictions.csv",index=False)
        tables=evaluation.compute_metric_table(probs,tgts,thr,LABEL_DISPLAY_NAMES)
        evaluation.save_metric_table(tables,out/f"{split}_metrics")
        if split=="test":
            evaluation.per_label_confusion(tgts,evaluation.apply_thresholds(probs,thr),LABEL_DISPLAY_NAMES).to_csv(out/"confusion_matrices.csv")

    ck=torch.load(out/"best_checkpoint.pt",map_location="cpu",weights_only=False)
    ck.update({"thresholds":thr.tolist(),"manifest_sha256":sha256(manifest_path),"seed":seed})
    torch.save(ck,out/"best_checkpoint.pt")
    write_json(out/"runtime.json",{
        "mode":"final","family":"GIN","seed":seed,"runtime_seconds":time.time()-t0,
        "best_epoch":ep,"best_val_macro_auprc":score,
        "manifest_sha256":sha256(manifest_path),"git_commit":git_commit(repo),
        "selected_config":selected
    })
    run_readme(out,"GIN",manifest_path,sha256(manifest_path),seed,{"selected_config":args.selected_config})
    print("PASS — GIN final run complete:",out)

if __name__=="__main__":
    main()
