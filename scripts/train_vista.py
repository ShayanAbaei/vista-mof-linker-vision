#!/usr/bin/env python3
"""Reproducible VISTA training entry point over a frozen manifest.

This wraps the existing historical VISTA modules without changing their model
implementation. It is intended for E0 sample-level reproduction and E1 strict
canonical-first-linker-disjoint training.

Full GPU execution should only begin after Phase-B smoke tests are green.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import torch
from torch.utils.data import DataLoader

from _bootstrap import add_vista_legacy_path
add_vista_legacy_path()

import data as vista_data
import evaluation
import experiments
import inference
import optim
import seeding
from train import MultiLabelTrainer

LABEL_COLUMNS = [
    "hydrophobicity","polarity","flexibility","aromaticity",
    "h_bond_donors","h_bond_acceptors","planarity_single_ring",
    "macrocycle_gt6_rings","charge_present","conjugation_present",
]
LABEL_DISPLAY_NAMES = [
    "Hydrophobicity","Polarity","Flexibility","Aromaticity",
    "Hydrogen-Bond Donors","Hydrogen-Bond Acceptors",
    "Planarity (Single-Ring)","Macrocycle (>6 Rings)",
    "Charge Present","Conjugation Present",
]

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def git_commit(repo:Path):
    try:
        return subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    except Exception:
        return None

def save_system(out:Path):
    try:
        txt=subprocess.check_output(["nvidia-smi"],text=True,stderr=subprocess.STDOUT)
    except Exception as e:
        txt=f"nvidia-smi unavailable: {e}\n"
    (out/"nvidia_smi.txt").write_text(txt,encoding="utf-8")
    try:
        pf=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
    except Exception as e:
        pf=f"pip freeze unavailable: {e}\n"
    (out/"pip_freeze.txt").write_text(pf,encoding="utf-8")

def make_loader(ds,batch_size,shuffle,num_workers,seed):
    return DataLoader(
        ds,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,
        worker_init_fn=seeding.seed_worker,
        generator=seeding.make_generator(seed),
        pin_memory=torch.cuda.is_available(),
    )

def split_csvs(manifest:pd.DataFrame,out:Path):
    d=out/"runtime_split_csvs";d.mkdir(parents=True,exist_ok=True)
    paths={}
    for split in ("train","validation","test"):
        x=manifest.loc[manifest["split"]==split,["filename"]+LABEL_COLUMNS].copy()
        p=d/f"{split}.csv";x.to_csv(p,index=False);paths[split]=p
    return paths

def enrich_predictions(pred:pd.DataFrame,manifest:pd.DataFrame):
    meta_cols=[c for c in [
        "filename","qmof_id","selected_linker_smiles","canonical_first_linker_smiles",
        "group_first_linker","row_index"
    ] if c in manifest.columns]
    meta=manifest[meta_cols].drop_duplicates("filename")
    return pred.merge(meta,on="filename",how="left",validate="one_to_one")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--seed",type=int,default=None)
    ap.add_argument("--output",required=True)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--image-dir",default=None)
    args=ap.parse_args()

    repo=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.seed is not None:
        cfg["run"]["seed"]=args.seed
    seed=int(cfg["run"]["seed"])
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    t0=time.time()

    # Freeze command/config provenance before training.
    (out/"command.txt").write_text(" ".join(shlex.quote(x) for x in sys.argv)+"\n",encoding="utf-8")
    (out/"git_commit.txt").write_text(str(git_commit(repo))+"\n",encoding="utf-8")
    (out/"resolved_config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
    save_system(out)

    manifest_path=repo/Path(cfg["data"]["manifest"])
    manifest=pd.read_csv(manifest_path)
    assert len(manifest)==9414
    assert set(manifest["split"].unique())=={"train","validation","test"}
    assert manifest["filename"].is_unique
    for c in LABEL_COLUMNS:
        assert c in manifest.columns
        assert set(manifest[c].unique()).issubset({0,1})
    if cfg["data"].get("require_canonical_disjoint",False):
        assert manifest.groupby("canonical_first_linker_smiles")["split"].nunique().max()==1

    image_dir=Path(args.image_dir) if args.image_dir else repo/Path(cfg["data"]["image_dir"])
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    missing=[fn for fn in manifest["filename"] if not (image_dir/fn).exists()]
    if missing:
        raise RuntimeError(f"Missing images: {len(missing)}; first={missing[:5]}")

    split_paths=split_csvs(manifest,out)

    seed_info=seeding.set_all_seeds(seed,deterministic_torch=True)

    # Primary run intentionally uses historical augmentation when chemistry_safe=false.
    train_tf,eval_tf=vista_data.image_augmentation(
        image_size=int(cfg["data"]["image_size"]),
        normalize=True,
        chemistry_safe=bool(cfg["augmentation"]["chemistry_safe"]),
    )

    train_ds=vista_data.MultiLabelMoleculeDataset(
        image_dir,split_paths["train"],LABEL_COLUMNS,transform=train_tf,filename_column="filename"
    )
    val_ds=vista_data.MultiLabelMoleculeDataset(
        image_dir,split_paths["validation"],LABEL_COLUMNS,transform=eval_tf,filename_column="filename"
    )
    test_ds=vista_data.MultiLabelMoleculeDataset(
        image_dir,split_paths["test"],LABEL_COLUMNS,transform=eval_tf,filename_column="filename"
    )

    bs=int(cfg["training"]["batch_size"])
    nw=int(cfg["training"]["num_workers"])
    train_loader=make_loader(train_ds,bs,True,nw,seed)
    val_loader=make_loader(val_ds,bs,False,nw,seed+1)
    test_loader=make_loader(test_ds,bs,False,nw,seed+2)

    freqs=train_ds.labels.mean(axis=0).tolist()
    (out/"train_label_frequencies.json").write_text(json.dumps(dict(zip(LABEL_COLUMNS,freqs)),indent=2),encoding="utf-8")

    ec=experiments.ExperimentConfig(
        name=cfg["run"]["experiment_name"],
        backbone=cfg["model"]["backbone"],
        embed_dim=int(cfg["model"]["embed_dim"]),
        n_encoder_layers=int(cfg["model"]["n_encoder_layers"]),
        n_heads=int(cfg["model"]["n_heads"]),
        n_labels=10,
        attention_mask_mode="none",
        loss_type="asl",
        base_lr=float(cfg["training"]["base_lr"]),
        backbone_lr_ratio=float(cfg["training"]["backbone_lr_ratio"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        epochs=int(cfg["training"]["epochs"]),
        patience=int(cfg["training"]["patience"]),
        unfreeze_epoch=int(cfg["training"]["frozen_epochs"]),
        selection_metric="macro_auprc",
        seeds=(seed,),
    )
    weights="default" if cfg["model"]["pretrained"] else None
    model=experiments.build_label_token_model(ec,weights=weights)
    criterion=experiments.build_criterion(freqs,loss_type="asl")
    optimizer=optim.build_optimizer(
        model,base_lr=ec.base_lr,backbone_lr_ratio=ec.backbone_lr_ratio,
        weight_decay=ec.weight_decay
    )
    sched_cfg=optim.LRScheduleConfig(
        warmup_start_factor=float(cfg["training"]["warmup_start_factor"]),
        warmup_epochs=int(cfg["training"]["warmup_epochs"]),
        total_epochs=int(cfg["training"]["epochs"]),
        min_lr_ratio=float(cfg["training"]["min_lr_ratio"]),
    )
    scheduler=optim.build_scheduler(optimizer,sched_cfg)
    optim.save_lr_schedule(out/"lr_schedule",ec.epochs,ec.base_lr,ec.backbone_lr_ratio,sched_cfg)

    device=args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    trainer=MultiLabelTrainer(
        model,train_loader,val_loader,criterion,LABEL_DISPLAY_NAMES,
        optimizer,scheduler,device=device,log_dir=str(out/"tensorboard"),
        checkpoint_dir=str(out/"checkpoints"),experiment_name="model",
        selection_metric="macro_auprc",experiment_config=ec.to_dict(),seeds=seed_info,
    )
    trainer.train(
        num_epochs=ec.epochs,patience=ec.patience,
        unfreeze_epoch=ec.unfreeze_epoch,unfreeze_lr_ratio=ec.backbone_lr_ratio
    )

    best_path=out/"checkpoints"/"model"/"best_model.pth"
    ck=torch.load(best_path,map_location=device,weights_only=False)
    model.load_state_dict(ck["model_state_dict"])

    # Deterministic eval-only loaders after training.
    val_probs,val_tgts=inference.collect_predictions(model,val_loader,device)
    test_probs,test_tgts=inference.collect_predictions(model,test_loader,device)

    sel=evaluation.select_thresholds(
        val_probs,val_tgts,LABEL_DISPLAY_NAMES,
        objective=cfg["evaluation"]["threshold_objective"],
        n_grid=int(cfg["evaluation"]["threshold_grid_points"]),
    )
    thresholds=sel["thresholds"]
    evaluation.save_thresholds(
        thresholds,LABEL_DISPLAY_NAMES,out/"thresholds.csv",
        scores=sel["scores"],objective=cfg["evaluation"]["threshold_objective"]
    )

    for split,loader,probs,tgts in [
        ("validation",val_loader,val_probs,val_tgts),
        ("test",test_loader,test_probs,test_tgts),
    ]:
        ids=inference.dataset_sample_ids(loader)
        fns=inference.dataset_filenames(loader)
        pred=evaluation.build_prediction_table(split,ids,fns,tgts,probs,thresholds,LABEL_DISPLAY_NAMES)
        pred=enrich_predictions(pred,manifest.loc[manifest["split"]==split])
        pred.to_csv(out/f"{split}_predictions.csv",index=False)
        tables=evaluation.compute_metric_table(probs,tgts,thresholds,LABEL_DISPLAY_NAMES)
        evaluation.save_metric_table(tables,out/f"{split}_metrics")
        if split=="test":
            binary=evaluation.apply_thresholds(probs,thresholds)
            conf=evaluation.per_label_confusion(tgts,binary,LABEL_DISPLAY_NAMES)
            conf.to_csv(out/"confusion_matrices.csv")

    hist_path=out/"checkpoints"/"model"/"history.json"
    if hist_path.exists():
        hist=json.loads(hist_path.read_text())
        rows=[]
        for h in hist:
            row={"epoch":h.get("epoch"),"train_loss":h.get("train_loss"),"val_loss":h.get("val_loss"),"lr":h.get("lr")}
            row.update({f"train_{k}":v for k,v in (h.get("train") or {}).items()})
            row.update({f"val_{k}":v for k,v in (h.get("val") or {}).items()})
            rows.append(row)
        pd.DataFrame(rows).to_csv(out/"history.csv",index=False)

    ck["thresholds"]=thresholds.tolist()
    ck["threshold_objective"]=cfg["evaluation"]["threshold_objective"]
    ck["manifest_sha256"]=sha256(manifest_path)
    ck["run_git_commit"]=git_commit(repo)
    torch.save(ck,out/"final_model_with_thresholds.pth")

    runtime={
        "runtime_seconds":time.time()-t0,
        "seed":seed,
        "best_epoch":ck.get("epoch"),
        "manifest":str(manifest_path),
        "manifest_sha256":sha256(manifest_path),
        "git_commit":git_commit(repo),
        "device":device,
    }
    (out/"runtime.json").write_text(json.dumps(runtime,indent=2),encoding="utf-8")
    print("PASS — run completed:",out)

if __name__=="__main__":
    main()
