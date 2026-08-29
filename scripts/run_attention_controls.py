#!/usr/bin/env python3
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from _bootstrap import add_vista_legacy_path
add_vista_legacy_path()
import data as vista_data, experiments
from _handover_utils import LABEL_COLUMNS, LABEL_DISPLAY_NAMES, load_frozen_manifest

def mask_cells(img, cells, fill):
    im=img.copy(); d=ImageDraw.Draw(im); W,H=im.size
    for cell in cells:
        r,c=divmod(int(cell),7)
        x0=round(c*W/7); x1=round((c+1)*W/7)
        y0=round(r*H/7); y1=round((r+1)*H/7)
        d.rectangle([x0,y0,x1-1,y1-1],fill=tuple(fill))
    return im

@torch.no_grad()
def predict(model, tf, images, device):
    x=torch.stack([tf(im) for im in images]).to(device)
    logits,_=model(x)
    return torch.sigmoid(logits).cpu().numpy()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--controls-config",required=True)
    ap.add_argument("--attention-dir",required=True)
    ap.add_argument("--repo-root",default=".")
    ap.add_argument("--image-dir",default=None)
    ap.add_argument("--output",required=True)
    ap.add_argument("--device",default="cuda")
    args=ap.parse_args()
    repo=Path(args.repo_root).resolve()
    cfg=yaml.safe_load(Path(args.config).read_text())
    cc=yaml.safe_load(Path(args.controls_config).read_text())["occlusion"]
    _,manifest=load_frozen_manifest(repo,cfg["data"]["manifest"],True)
    test=manifest[manifest.split=="test"].reset_index(drop=True)
    image_dir=Path(args.image_dir) if args.image_dir else repo/cfg["data"]["image_dir"]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)

    ec=experiments.ExperimentConfig(name="occlusion",backbone=cfg["model"]["backbone"],
        embed_dim=int(cfg["model"]["embed_dim"]),n_encoder_layers=int(cfg["model"]["n_encoder_layers"]),
        n_heads=int(cfg["model"]["n_heads"]),n_labels=10,seeds=(42,))
    model=experiments.build_label_token_model(ec,weights=("default" if cfg["model"]["pretrained"] else None)).to(args.device)
    ck=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    model.load_state_dict(ck["model_state_dict"]);model.eval()
    _,tf=vista_data.image_augmentation(image_size=int(cfg["data"]["image_size"]),normalize=True,chemistry_safe=False)

    a=np.load(Path(args.attention_dir)/"test_attention_summary.npz")
    maps=a["label_image_final_meanheads"]; probs=a["probs"]; targets=a["targets"]
    if len(test)!=len(maps): raise ValueError("attention/test metadata mismatch")

    frac=float(cc["fraction_7x7_cells"]);k=max(1,int(round(49*frac)))
    cap=int(cc["max_positive_pairs_per_label"]);rr=int(cc["random_repeats"]);fill=cc["fill_rgb"]
    rng=np.random.default_rng(42); rows=[]
    for lab in range(10):
        candidates=np.where(targets[:,lab]==1)[0]
        if len(candidates)>cap:
            candidates=np.sort(rng.choice(candidates,size=cap,replace=False))
        print(f"[{lab+1}/10] {LABEL_DISPLAY_NAMES[lab]}: {len(candidates)} positive pairs")
        for idx in candidates:
            raw=Image.open(image_dir/test.iloc[idx]["filename"]).convert("RGB")
            flat=maps[idx,lab].reshape(-1)
            hi=np.argsort(flat)[-k:]; lo=np.argsort(flat)[:k]
            p0=float(probs[idx,lab])
            perturb=[mask_cells(raw,hi,fill),mask_cells(raw,lo,fill)]
            for _ in range(rr):
                rc=rng.choice(49,size=k,replace=False)
                perturb.append(mask_cells(raw,rc,fill))
            pp=predict(model,tf,perturb,args.device)[:,lab]
            ph=float(pp[0]); pl=float(pp[1]); prs=[float(x) for x in pp[2:]]
            rows.append({
                "label":LABEL_DISPLAY_NAMES[lab],"test_row":int(idx),"filename":test.iloc[idx]["filename"],
                "canonical_first_linker_smiles":test.iloc[idx]["canonical_first_linker_smiles"],
                "n_cells":k,"p_original":p0,"p_high_mask":ph,"p_low_mask":pl,
                "p_random_mask_mean":float(np.mean(prs)),
                "delta_high":p0-ph,"delta_low":p0-pl,"delta_random":p0-float(np.mean(prs))
            })
    df=pd.DataFrame(rows);df.to_csv(out/"attention_occlusion_pairs.csv",index=False)
    s=df.groupby("label").agg(
        n=("delta_high","size"),
        mean_delta_high=("delta_high","mean"),
        mean_delta_low=("delta_low","mean"),
        mean_delta_random=("delta_random","mean")
    ).reset_index()
    s["high_minus_random"]=s["mean_delta_high"]-s["mean_delta_random"]
    s["high_minus_low"]=s["mean_delta_high"]-s["mean_delta_low"]
    s.to_csv(out/"attention_occlusion_summary.csv",index=False)
    print("PASS — coarse attention-guided perturbation complete")

if __name__=="__main__":
    main()
