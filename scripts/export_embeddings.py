#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from torch.utils.data import DataLoader

from _bootstrap import add_vista_legacy_path
add_vista_legacy_path()
import data as vista_data, experiments
from _handover_utils import LABEL_COLUMNS, load_frozen_manifest, write_json

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--repo-root",default=".")
    ap.add_argument("--image-dir",default=None)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--batch-size",type=int,default=64)
    args=ap.parse_args()

    repo=Path(args.repo_root).resolve()
    cfg=yaml.safe_load(Path(args.config).read_text())
    manifest_path,manifest=load_frozen_manifest(repo,cfg["data"]["manifest"],True)
    image_dir=Path(args.image_dir) if args.image_dir else repo/cfg["data"]["image_dir"]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)

    all_csv=out/"_all_eval.csv"
    manifest[["filename"]+LABEL_COLUMNS].to_csv(all_csv,index=False)
    _,tf=vista_data.image_augmentation(image_size=int(cfg["data"]["image_size"]),normalize=True,chemistry_safe=False)
    ds=vista_data.MultiLabelMoleculeDataset(image_dir,all_csv,LABEL_COLUMNS,tf,"filename")
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=True)

    ec=experiments.ExperimentConfig(
        name="embedding_export",backbone=cfg["model"]["backbone"],
        embed_dim=int(cfg["model"]["embed_dim"]),n_encoder_layers=int(cfg["model"]["n_encoder_layers"]),
        n_heads=int(cfg["model"]["n_heads"]),n_labels=10,seeds=(42,)
    )
    model=experiments.build_label_token_model(ec,weights=("default" if cfg["model"]["pretrained"] else None)).to(args.device)
    ck=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    model.load_state_dict(ck["model_state_dict"]);model.eval()
    inner=model.model;L=inner.n_labels

    bb=[];proj=[];post=[];lab=[];labmean=[]
    for images,_ in loader:
        f=model.backbone(images.to(args.device))
        bb.append(f.mean(dim=(-2,-1)).cpu().numpy().astype("float32"))
        flat=f.flatten(2).transpose(1,2)
        p=inner.feature_embed.projection(flat)
        proj.append(p.mean(dim=1).cpu().numpy().astype("float32"))

        x=inner.feature_embed(f)
        block_mask=None
        for block in inner.encoder_blocks:
            x,_=block(x,return_attention_weights=False,attn_block_mask=block_mask)
        x=inner.norm_final(x)
        post.append(x[:,L:].mean(dim=1).cpu().numpy().astype("float32"))
        lt=x[:,:L]
        lab.append(lt.cpu().numpy().astype("float32"))
        labmean.append(lt.mean(dim=1).cpu().numpy().astype("float32"))

    arrays={
        "backbone_mean_1024":np.concatenate(bb),
        "projected_image_mean_384":np.concatenate(proj),
        "post_transformer_image_mean_384":np.concatenate(post),
        "final_label_tokens_10x384":np.concatenate(lab),
        "final_label_mean_384":np.concatenate(labmean),
    }
    for k,a in arrays.items(): np.save(out/f"embedding_{k}.npy",a)

    meta_cols=[c for c in [
        "row_index","sample_id","filename","qmof_id","canonical_first_linker_smiles","split"
    ]+LABEL_COLUMNS if c in manifest.columns]
    meta=manifest[meta_cols].copy().reset_index(drop=True)
    meta.to_csv(out/"embedding_metadata.csv",index=False)

    # Canonical aggregates for all 2D representations.
    groups=meta["canonical_first_linker_smiles"].astype(str)
    can_meta=meta.groupby("canonical_first_linker_smiles",sort=False).first().reset_index()
    can_meta["n_records"]=meta.groupby("canonical_first_linker_smiles",sort=False).size().values
    can_meta.to_csv(out/"embedding_metadata_canonical.csv",index=False)
    tmp=pd.DataFrame({"group":groups})
    index_lists=list(tmp.groupby("group",sort=False).indices.values())
    for k,a in arrays.items():
        # Mean over repeated QMOF-derived rows, preserving every non-row axis.
        ca=np.stack([a[np.asarray(ix)].mean(axis=0) for ix in index_lists]).astype("float32")
        np.save(out/f"embedding_canonical_{k}.npy",ca)

    write_json(out/"embedding_contract.json",{
        "primary_umap_representation":"post_transformer_image_mean_384",
        "row_count":len(meta),"canonical_count":len(can_meta),
        "split_source":str(manifest_path),
        "arrays":{k:list(a.shape) for k,a in arrays.items()}
    })
    all_csv.unlink(missing_ok=True)
    print("PASS — VISTA embeddings exported:",out)

if __name__=="__main__":
    main()
