#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--embeddings",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    root=Path(args.embeddings);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load(Path(args.config).read_text())
    rep=cfg["input_representation"]
    X=np.load(root/f"embedding_canonical_{rep}.npy")
    meta=pd.read_csv(root/"embedding_metadata_canonical.csv")
    if len(X)!=len(meta): raise ValueError("canonical embedding/metadata row mismatch")
    Xn=normalize(X,norm="l2")
    train=meta["split"].eq("train").to_numpy()
    if not train.any(): raise ValueError("no training canonical identities")

    pca=PCA(n_components=int(cfg["pca"]["n_components"]))
    pca.fit(Xn[train]); pc=pca.transform(Xn)

    import umap
    uc=cfg["umap"]
    reducer=umap.UMAP(
        n_components=int(uc["n_components"]),n_neighbors=int(uc["n_neighbors"]),
        min_dist=float(uc["min_dist"]),metric=uc["metric"],
        random_state=int(uc["random_state"]),transform_seed=int(uc["random_state"])
    )
    reducer.fit(Xn[train]); um=reducer.transform(Xn)

    coords=meta.copy()
    coords["PCA1"]=pc[:,0];coords["PCA2"]=pc[:,1]
    coords["UMAP1"]=um[:,0];coords["UMAP2"]=um[:,1]
    coords.to_csv(out/"canonical_dimred_coordinates.csv",index=False)

    # Row-level coordinates are transformed through the SAME reducers; they are
    # never used to fit PCA/UMAP.
    row_X=np.load(root/f"embedding_{rep}.npy")
    row_meta=pd.read_csv(root/"embedding_metadata.csv")
    if len(row_X)!=len(row_meta): raise ValueError("row embedding/metadata mismatch")
    row_Xn=normalize(row_X,norm="l2")
    row_pc=pca.transform(row_Xn)
    row_um=reducer.transform(row_Xn)
    row_coords=row_meta.copy()
    row_coords["PCA1"]=row_pc[:,0];row_coords["PCA2"]=row_pc[:,1]
    row_coords["UMAP1"]=row_um[:,0];row_coords["UMAP2"]=row_um[:,1]
    row_coords.to_csv(out/"row_dimred_coordinates.csv",index=False)

    (out/"dimred_protocol.json").write_text(json.dumps({
        "representation":rep,"l2_normalized":True,"fit_split":"train",
        "n_fit_train_canonical":int(train.sum()),
        "pca_explained_variance_ratio":pca.explained_variance_ratio_.tolist(),
        "umap":uc
    },indent=2),encoding="utf-8")
    print("PASS — train-fit PCA/UMAP source coordinates:",out)
if __name__=="__main__":
    main()
