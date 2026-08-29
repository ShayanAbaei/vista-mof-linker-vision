#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, shlex, shutil, time
from pathlib import Path

import pandas as pd
import torch, yaml
from torch.utils.data import DataLoader

from _bootstrap import add_vista_legacy_path
add_vista_legacy_path()

import baselines
import data as vista_data
import evaluation, experiments, inference, optim, seeding
from train import MultiLabelTrainer
from _handover_utils import (
    LABEL_COLUMNS, LABEL_DISPLAY_NAMES, enrich_predictions, git_commit,
    load_frozen_manifest, save_system, sha256, split_csvs, run_readme
)

def make_loader(ds, bs, shuffle, nw, seed):
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle, num_workers=nw,
        worker_init_fn=seeding.seed_worker,
        generator=seeding.make_generator(seed),
        pin_memory=torch.cuda.is_available(),
    )

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
        cfg["run"]["seed"]=int(args.seed)
    seed=int(cfg["run"]["seed"])
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    t0=time.time()

    (out/"command.txt").write_text(" ".join(shlex.quote(x) for x in __import__("sys").argv)+"\n",encoding="utf-8")
    (out/"git_commit.txt").write_text(str(git_commit(repo))+"\n",encoding="utf-8")
    (out/"resolved_config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
    save_system(out)

    manifest_path,manifest=load_frozen_manifest(
        repo,cfg["data"]["manifest"],bool(cfg["data"].get("require_canonical_disjoint",False))
    )
    image_dir=Path(args.image_dir) if args.image_dir else repo/Path(cfg["data"]["image_dir"])
    missing=[fn for fn in manifest["filename"] if not (image_dir/fn).exists()]
    if missing:
        raise RuntimeError(f"Missing images: {len(missing)}; first={missing[:5]}")
    split_paths=split_csvs(manifest,out)
    seed_info=seeding.set_all_seeds(seed,deterministic_torch=True)

    train_tf,eval_tf=vista_data.image_augmentation(
        image_size=int(cfg["data"]["image_size"]),
        normalize=True,
        chemistry_safe=bool(cfg["augmentation"]["chemistry_safe"]),
    )
    train_ds=vista_data.MultiLabelMoleculeDataset(image_dir,split_paths["train"],LABEL_COLUMNS,train_tf,"filename")
    val_ds=vista_data.MultiLabelMoleculeDataset(image_dir,split_paths["validation"],LABEL_COLUMNS,eval_tf,"filename")
    test_ds=vista_data.MultiLabelMoleculeDataset(image_dir,split_paths["test"],LABEL_COLUMNS,eval_tf,"filename")

    bs=int(cfg["training"]["batch_size"])
    nw=int(cfg["training"]["num_workers"])
    train_loader=make_loader(train_ds,bs,True,nw,seed)
    val_loader=make_loader(val_ds,bs,False,nw,seed+1)
    test_loader=make_loader(test_ds,bs,False,nw,seed+2)

    # CRITICAL: exact frozen shared architecture from src/vista/baselines.py:
    # Swin-B spatial backbone + GAP + ten independent Linear(1024,1) heads.
    model=baselines.build_shared_cnn(
        backbone_name=cfg["model"]["backbone"],
        n_labels=int(cfg["model"]["n_labels"]),
        weights=("default" if cfg["model"]["pretrained"] else None),
    )
    model.freeze_backbone()

    freqs=train_ds.labels.mean(axis=0).tolist()
    criterion=experiments.build_criterion(freqs,loss_type="asl")
    optimizer=optim.build_optimizer(
        model,
        base_lr=float(cfg["training"]["base_lr"]),
        backbone_lr_ratio=float(cfg["training"]["backbone_lr_ratio"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    sched_cfg=optim.LRScheduleConfig(
        warmup_start_factor=float(cfg["training"]["warmup_start_factor"]),
        warmup_epochs=int(cfg["training"]["warmup_epochs"]),
        total_epochs=int(cfg["training"]["epochs"]),
        min_lr_ratio=float(cfg["training"]["min_lr_ratio"]),
    )
    scheduler=optim.build_scheduler(optimizer,sched_cfg)
    optim.save_lr_schedule(
        out/"lr_schedule",
        int(cfg["training"]["epochs"]),
        float(cfg["training"]["base_lr"]),
        float(cfg["training"]["backbone_lr_ratio"]),
        sched_cfg,
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    trainer=MultiLabelTrainer(
        model,train_loader,val_loader,criterion,LABEL_DISPLAY_NAMES,
        optimizer,scheduler,device=args.device,
        log_dir=str(out/"tensorboard"),
        checkpoint_dir=str(out/"checkpoints"),
        experiment_name="model",
        selection_metric="macro_auprc",
        experiment_config={
            "family":"shared_swin",
            "architecture_source":"src/vista/baselines.py::SharedCNNMultiLabel",
            "config":cfg,
        },
        seeds=seed_info,
    )
    trainer.train(
        num_epochs=int(cfg["training"]["epochs"]),
        patience=int(cfg["training"]["patience"]),
        unfreeze_epoch=int(cfg["training"]["frozen_epochs"]),
        unfreeze_lr_ratio=float(cfg["training"]["backbone_lr_ratio"]),
    )

    best_path=out/"checkpoints/model/best_model.pth"
    ck=torch.load(best_path,map_location=args.device,weights_only=False)
    model.load_state_dict(ck["model_state_dict"])

    val_probs,val_tgts=inference.collect_predictions(model,val_loader,args.device)
    test_probs,test_tgts=inference.collect_predictions(model,test_loader,args.device)
    sel=evaluation.select_thresholds(
        val_probs,val_tgts,LABEL_DISPLAY_NAMES,
        objective=cfg["evaluation"]["threshold_objective"],
        n_grid=int(cfg["evaluation"]["threshold_grid_points"]),
    )
    thr=sel["thresholds"]
    evaluation.save_thresholds(
        thr,LABEL_DISPLAY_NAMES,out/"thresholds.csv",
        sel["scores"],cfg["evaluation"]["threshold_objective"],
    )

    for split,loader,probs,tgts in [
        ("validation",val_loader,val_probs,val_tgts),
        ("test",test_loader,test_probs,test_tgts),
    ]:
        pred=evaluation.build_prediction_table(
            split,inference.dataset_sample_ids(loader),inference.dataset_filenames(loader),
            tgts,probs,thr,LABEL_DISPLAY_NAMES,
        )
        pred=enrich_predictions(pred,manifest.loc[manifest["split"]==split])
        pred.to_csv(out/f"{split}_predictions.csv",index=False)
        tables=evaluation.compute_metric_table(probs,tgts,thr,LABEL_DISPLAY_NAMES)
        evaluation.save_metric_table(tables,out/f"{split}_metrics")
        if split=="test":
            evaluation.per_label_confusion(
                tgts,evaluation.apply_thresholds(probs,thr),LABEL_DISPLAY_NAMES
            ).to_csv(out/"confusion_matrices.csv",index=False)

    hist=out/"checkpoints/model/history.json"
    if hist.exists():
        rows=[]
        for h in json.loads(hist.read_text()):
            row={"epoch":h.get("epoch"),"train_loss":h.get("train_loss"),"val_loss":h.get("val_loss"),"lr":h.get("lr")}
            row.update({f"train_{k}":v for k,v in (h.get("train") or {}).items()})
            row.update({f"val_{k}":v for k,v in (h.get("val") or {}).items()})
            rows.append(row)
        pd.DataFrame(rows).to_csv(out/"history.csv",index=False)

    ck["thresholds"]=thr.tolist()
    ck["manifest_sha256"]=sha256(manifest_path)
    ck["run_git_commit"]=git_commit(repo)
    ck["architecture_source"]="src/vista/baselines.py::SharedCNNMultiLabel"
    torch.save(ck,out/"final_model_with_thresholds.pth")
    (out/"best_epoch.txt").write_text(str(ck.get("epoch"))+"\n",encoding="utf-8")
    try:
        os.link(out/"final_model_with_thresholds.pth",out/"best_checkpoint.pt")
    except Exception:
        shutil.copy2(out/"final_model_with_thresholds.pth",out/"best_checkpoint.pt")

    runtime={
        "runtime_seconds":time.time()-t0,
        "seed":seed,
        "best_epoch":ck.get("epoch"),
        "manifest":str(manifest_path),
        "manifest_sha256":sha256(manifest_path),
        "git_commit":git_commit(repo),
        "device":args.device,
        "family":"shared_swin",
        "architecture_source":"src/vista/baselines.py::SharedCNNMultiLabel",
        "n_params":int(sum(p.numel() for p in model.parameters())),
    }
    (out/"runtime.json").write_text(json.dumps(runtime,indent=2),encoding="utf-8")
    run_readme(
        out,"shared_swin",manifest_path,sha256(manifest_path),seed,
        {"architecture_source":"src/vista/baselines.py::SharedCNNMultiLabel"},
    )
    print("PASS — exact frozen shared-Swin architecture run completed:",out)

if __name__=="__main__":
    main()
