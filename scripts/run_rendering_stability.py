#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from PIL import Image, ImageFilter
from torchvision.transforms import functional as TF

from _bootstrap import add_vista_legacy_path
add_vista_legacy_path()
import data as vista_data, experiments, evaluation
from _handover_utils import LABEL_COLUMNS, LABEL_DISPLAY_NAMES, load_frozen_manifest

def variants(im,cfg):
    out=[]
    for a in cfg["rotations_deg"]:
        out.append((f"rotate_{a}",TF.rotate(im,float(a),fill=255)))
    for xy in cfg["translations_px"]:
        out.append((f"translate_{xy[0]}_{xy[1]}",TF.affine(im,0,tuple(xy),1.0,[0.0,0.0],fill=255)))
    for s in cfg["scales"]:
        out.append((f"scale_{s}",TF.affine(im,0,[0,0],float(s),[0.0,0.0],fill=255)))

    # Explicit padding/centering perturbation: shrink depiction and center it
    # on a same-size white canvas without changing connectivity.
    W,H=im.size
    for pad in cfg.get("padding_px",[]):
        pad=int(pad)
        if 0 < pad < min(W,H)//3:
            small=im.resize((W-2*pad,H-2*pad),resample=Image.Resampling.BILINEAR)
            canvas=Image.new("RGB",(W,H),(255,255,255))
            canvas.paste(small,(pad,pad))
            out.append((f"padding_{pad}px",canvas))

    # Deterministic raster line-thickness sensitivity. MinFilter expands dark
    # strokes; MaxFilter contracts them. This changes depiction style only.
    for mode in cfg.get("line_width_filters",[]):
        if mode=="thicker":
            out.append(("line_thicker_3px",im.filter(ImageFilter.MinFilter(3))))
        elif mode=="thinner":
            out.append(("line_thinner_3px",im.filter(ImageFilter.MaxFilter(3))))
    return out

@torch.no_grad()
def batch_prob(model,tf,ims,device):
    x=torch.stack([tf(z) for z in ims]).to(device)
    logits,_=model(x)
    return torch.sigmoid(logits).cpu().numpy()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--controls-config",required=True)
    ap.add_argument("--repo-root",default=".")
    ap.add_argument("--image-dir",default=None)
    ap.add_argument("--output",required=True)
    ap.add_argument("--device",default="cuda")
    args=ap.parse_args()
    repo=Path(args.repo_root).resolve()
    cfg=yaml.safe_load(Path(args.config).read_text())
    rc=yaml.safe_load(Path(args.controls_config).read_text())["rendering_stability"]
    _,manifest=load_frozen_manifest(repo,cfg["data"]["manifest"],True)
    test=manifest[manifest.split=="test"].reset_index(drop=True)
    image_dir=Path(args.image_dir) if args.image_dir else repo/cfg["data"]["image_dir"]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    thresholds=pd.read_csv(Path(args.checkpoint).parent/"thresholds.csv")["threshold"].to_numpy(float) \
        if (Path(args.checkpoint).parent/"thresholds.csv").exists() else np.asarray(
            torch.load(args.checkpoint,map_location="cpu",weights_only=False).get("thresholds"),float)

    ec=experiments.ExperimentConfig(name="render_stability",backbone=cfg["model"]["backbone"],
        embed_dim=int(cfg["model"]["embed_dim"]),n_encoder_layers=int(cfg["model"]["n_encoder_layers"]),
        n_heads=int(cfg["model"]["n_heads"]),n_labels=10,seeds=(42,))
    model=experiments.build_label_token_model(ec,weights=("default" if cfg["model"]["pretrained"] else None)).to(args.device)
    ck=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    model.load_state_dict(ck["model_state_dict"]);model.eval()
    _,tf=vista_data.image_augmentation(image_size=int(cfg["data"]["image_size"]),normalize=True,chemistry_safe=False)

    rows=[]
    for i,row in test.iterrows():
        raw=Image.open(image_dir/row["filename"]).convert("RGB")
        vs=variants(raw,rc); names=["original"]+[x[0] for x in vs]; ims=[raw]+[x[1] for x in vs]
        pr=batch_prob(model,tf,ims,args.device)
        p0=pr[0]; b0=(p0>thresholds).astype(int)
        for k,name in enumerate(names[1:],1):
            pv=pr[k];bv=(pv>thresholds).astype(int)
            for j,lab in enumerate(LABEL_DISPLAY_NAMES):
                rows.append({
                    "filename":row["filename"],"canonical_first_linker_smiles":row["canonical_first_linker_smiles"],
                    "variant":name,"label":lab,"ground_truth":int(row[LABEL_COLUMNS[j]]),
                    "p_original":p0[j],"p_variant":pv[j],
                    "abs_delta":abs(p0[j]-pv[j]),"prediction_consistent":int(b0[j]==bv[j])
                })
        if (i+1)%200==0: print("processed",i+1,"/",len(test))
    df=pd.DataFrame(rows);df.to_csv(out/"rendering_stability_rows.csv",index=False)
    df["brier_variant"]=(df["p_variant"]-df["ground_truth"])**2
    summary=df.groupby(["variant","label"]).agg(
        n=("abs_delta","size"),
        mean_abs_probability_change=("abs_delta","mean"),
        binary_prediction_consistency=("prediction_consistent","mean"),
        brier_score=("brier_variant","mean"),
    ).reset_index()
    cors=[]
    for (variant,lab),g in df.groupby(["variant","label"]):
        cors.append({"variant":variant,"label":lab,"probability_correlation":g["p_original"].corr(g["p_variant"])})
    summary=summary.merge(pd.DataFrame(cors),on=["variant","label"])
    summary.to_csv(out/"rendering_stability_summary.csv",index=False)
    print("PASS — deterministic rendering-stability control complete")

if __name__=="__main__":
    main()
