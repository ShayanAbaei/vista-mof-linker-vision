#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from torch.utils.data import DataLoader

from _bootstrap import add_vista_legacy_path
add_vista_legacy_path()
import data as vista_data, experiments, inference
from attention import label_to_image_maps, attention_rollout
from attention_label_label import per_sample_label_label, top_source_labels
from _handover_utils import LABEL_COLUMNS, LABEL_DISPLAY_NAMES, load_frozen_manifest

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--repo-root",default=".")
    ap.add_argument("--image-dir",default=None)
    ap.add_argument("--output",required=True)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--batch-size",type=int,default=64)
    args=ap.parse_args()
    repo=Path(args.repo_root).resolve()
    cfg=yaml.safe_load(Path(args.config).read_text())
    _,manifest=load_frozen_manifest(repo,cfg["data"]["manifest"],True)
    image_dir=Path(args.image_dir) if args.image_dir else repo/cfg["data"]["image_dir"]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)

    ec=experiments.ExperimentConfig(
        name="attention_export",backbone=cfg["model"]["backbone"],
        embed_dim=int(cfg["model"]["embed_dim"]),n_encoder_layers=int(cfg["model"]["n_encoder_layers"]),
        n_heads=int(cfg["model"]["n_heads"]),n_labels=10,seeds=(42,)
    )
    model=experiments.build_label_token_model(ec,weights=("default" if cfg["model"]["pretrained"] else None)).to(args.device)
    ck=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    model.load_state_dict(ck["model_state_dict"]);model.eval()

    _,tf=vista_data.image_augmentation(image_size=int(cfg["data"]["image_size"]),normalize=True,chemistry_safe=False)
    train_P=[]
    for split in ("train","validation","test"):
        frame=manifest[manifest.split==split].reset_index(drop=True)
        csv=out/f"_{split}.csv";frame[["filename"]+LABEL_COLUMNS].to_csv(csv,index=False)
        ds=vista_data.MultiLabelMoleculeDataset(image_dir,csv,LABEL_COLUMNS,tf,"filename")
        loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0)
        li=[];roll=[];A=[];P=[];Y=[];PROB=[];FULL=[]
        for images,targets,probs,attn in inference.iter_attention(model,loader,args.device):
            last=attn[-1]
            maps=label_to_image_maps(last,10,(7,7)).numpy().astype("float32")
            li.append(maps)
            R=attention_rollout(attn).numpy()
            roll.append(R[:,:10,10:].reshape(-1,10,7,7).astype("float32"))
            a,p=per_sample_label_label(attn,10,layers="last")
            A.append(a.astype("float32"));P.append(p.astype("float32"))
            if split=="test":
                FULL.append(last.mean(dim=1).numpy().astype("float32"))
            Y.append(targets.numpy().astype("int8"));PROB.append(probs.numpy().astype("float32"))
        arrays={
            "label_image_final_meanheads":np.concatenate(li),
            "label_image_rollout":np.concatenate(roll),
            "label_label_A":np.concatenate(A),
            "label_label_P":np.concatenate(P),
            "targets":np.concatenate(Y),
            "probs":np.concatenate(PROB),
        }
        if split=="test":
            arrays["full_attention_final_meanheads_59x59"]=np.concatenate(FULL)
        np.savez_compressed(out/f"{split}_attention_summary.npz",**arrays)
        meta_cols=[c for c in ["row_index","filename","qmof_id","canonical_first_linker_smiles","split"] if c in frame]
        frame[meta_cols].to_csv(out/f"{split}_attention_metadata.csv",index=False)
        if split=="train": train_P.append(arrays["label_label_P"])
        csv.unlink(missing_ok=True)

    Pmean=np.concatenate(train_P).mean(axis=0)
    hubs=top_source_labels(Pmean,k=3)
    hub_payload={
        "selection_source":"TRAIN ONLY",
        "method":"columns with highest mean off-diagonal conditional label preference P",
        "hub_indices":hubs,
        "hub_labels":[LABEL_DISPLAY_NAMES[i] for i in hubs]
    }
    (out/"train_only_attention_hubs.json").write_text(json.dumps(hub_payload,indent=2),encoding="utf-8")

    # Candidate list for the eventual two-case manuscript design. This is
    # post-hoc descriptive selection only, after the strict seed-42 model is
    # frozen; it must never be used to change the model.
    test_npz=np.load(out/"test_attention_summary.npz")
    tprob=test_npz["probs"]; ty=test_npz["targets"]
    tmeta=pd.read_csv(out/"test_attention_metadata.csv")
    thresholds=np.asarray(ck.get("thresholds") or [0.5]*10,dtype=float)
    cand=[]
    for j,lab in enumerate(LABEL_DISPLAY_NAMES):
        pred=(tprob[:,j] > thresholds[j]).astype(int)
        correct_pos=np.where((ty[:,j]==1)&(pred==1))[0]
        for idx in correct_pos[np.argsort(tprob[correct_pos,j])[-5:]]:
            cand.append({"label":lab,"case_type":"correct_high_confidence","test_row":int(idx),
                         "filename":tmeta.iloc[idx]["filename"],"gt":1,
                         "prob":float(tprob[idx,j]),"pred":1,"threshold":float(thresholds[j])})
        # Borderline examples nearest the frozen decision threshold.
        order=np.argsort(np.abs(tprob[:,j]-thresholds[j]))[:5]
        for idx in order:
            cand.append({"label":lab,"case_type":"borderline","test_row":int(idx),
                         "filename":tmeta.iloc[idx]["filename"],"gt":int(ty[idx,j]),
                         "prob":float(tprob[idx,j]),"pred":int(pred[idx]),"threshold":float(thresholds[j])})
        errors=np.where(pred!=ty[:,j])[0]
        if len(errors):
            # Deterministically list the five largest-confidence errors.
            conf=np.abs(tprob[errors,j]-thresholds[j])
            for idx in errors[np.argsort(conf)[-5:]]:
                cand.append({"label":lab,"case_type":"error","test_row":int(idx),
                             "filename":tmeta.iloc[idx]["filename"],"gt":int(ty[idx,j]),
                             "prob":float(tprob[idx,j]),"pred":int(pred[idx]),"threshold":float(thresholds[j])})
    pd.DataFrame(cand).to_csv(out/"test_attention_case_candidates.csv",index=False)

    print("PASS — strict seed-42 attention summaries exported; hubs derived from TRAIN ONLY")

if __name__=="__main__":
    main()
