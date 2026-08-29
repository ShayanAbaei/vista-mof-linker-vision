#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil, tarfile
from pathlib import Path

SOURCES={
    "e0_sample_seed42":"runs/epfl/e0_vista_sample_seed42",
    "e1_vista_strict_seed17":"runs/epfl/e1_vista_strict_seed17",
    "e1_vista_strict_seed42":"runs/epfl/e1_vista_strict_seed42",
    "e1_vista_strict_seed123":"runs/epfl/e1_vista_strict_seed123",
    "e2_shared_swin_seed17":"runs/epfl/e2_shared_swin_strict_seed17",
    "e2_shared_swin_seed42":"runs/epfl/e2_shared_swin_strict_seed42",
    "e2_shared_swin_seed123":"runs/epfl/e2_shared_swin_strict_seed123",
    "e3_gin_tuning":"runs/epfl/e3_gin_tuning",
    "e3_gin_seed17":"runs/epfl/e3_gin_strict_seed17",
    "e3_gin_seed42":"runs/epfl/e3_gin_strict_seed42",
    "e3_gin_seed123":"runs/epfl/e3_gin_strict_seed123",
    "embeddings_seed42":"outputs/epfl_exports/embeddings_seed42",
    "dimred_seed42":"outputs/epfl_exports/dimred_seed42",
    "attention_seed42":"outputs/epfl_exports/attention_seed42",
    "negative_attention_controls":"outputs/epfl_exports/negative_attention_seed42",
    "occlusion_seed42":"outputs/epfl_exports/occlusion_seed42",
    "rendering_stability_seed42":"outputs/epfl_exports/rendering_stability_seed42",
    "similarity_stratification":"outputs/epfl_exports/similarity_seed42",
    "mof_level_consequence":"outputs/epfl_exports/mof_consequence",
    "summary_tables":"outputs/epfl_summaries",
}

def sha256(path,chunk=1024*1024):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(chunk),b""):h.update(b)
    return h.hexdigest()

def link_or_copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    try:
        os.link(src,dst);return "hardlink"
    except Exception:
        shutil.copy2(src,dst);return "copy"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",default=".")
    ap.add_argument("--output",default="VISTA_EPFL_RETURN_v1")
    ap.add_argument("--archive",action="store_true")
    args=ap.parse_args()
    repo=Path(args.repo_root).resolve()
    out=repo/args.output
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)

    missing=[]
    manifest=[]
    for logical,rel in SOURCES.items():
        src=repo/rel
        dest=out/logical
        if not src.exists():
            if logical=="mof_level_consequence":
                dest.mkdir(parents=True)
                (dest/"NOT_RUN_REASON.md").write_text(
                    "# Optional MOF-level consequence not present\n\n"
                    "The predeclared optional screen was not available in this return build.\n",
                    encoding="utf-8",
                )
                continue
            missing.append(f"{logical}: {src}")
            continue
        for f in src.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts:
                d=dest/f.relative_to(src)
                mode=link_or_copy(f,d)
                manifest.append({"logical_group":logical,"source":str(f.relative_to(repo)),"return_path":str(d.relative_to(out)),"transfer":mode})

    if missing:
        (out/"MISSING_REQUIRED_OUTPUTS.txt").write_text("\n".join(missing)+"\n",encoding="utf-8")
        print("FAIL — required run outputs are missing:")
        for x in missing: print(" -",x)
        return 2

    # Add frozen configs/docs needed to interpret the returned science.
    context=out/"frozen_context"
    for rel in [
        "configs/vista_sample_repro.yaml","configs/vista_group_disjoint.yaml",
        "configs/shared_swin_group_disjoint.yaml","configs/gin_group_disjoint.yaml",
        "configs/dimred.yaml","configs/attention_controls.yaml","configs/e0_acceptance.yaml",
        "data/frozen/canonical_group_disjoint_split.csv","data/frozen/current_sample_split.csv",
        "docs/EPFL_HANDOVER.md","docs/OUTPUT_CONTRACT.md",
    ]:
        src=repo/rel
        if src.exists(): link_or_copy(src,context/rel)

    # Hash every returned file except hash table itself.
    rows=[]
    for f in sorted(out.rglob("*")):
        if f.is_file() and f.name!="SHA256SUMS.txt":
            rows.append({"path":str(f.relative_to(out)).replace("\\","/"),"bytes":f.stat().st_size,"sha256":sha256(f)})
    with open(out/"MASTER_RUN_MANIFEST.csv","w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=["path","bytes","sha256"]);w.writeheader();w.writerows(rows)
    (out/"SHA256SUMS.txt").write_text("\n".join(f'{r["sha256"]}  {r["path"]}' for r in rows)+"\n",encoding="utf-8")
    (out/"README_RETURN.md").write_text(
        "# VISTA EPFL return bundle\n\n"
        "This bundle is generated automatically from the frozen run/output locations. "
        "It contains raw predictions, thresholds, checkpoints, environment/provenance, "
        "strict repeated-seed results, embeddings, dimensional-reduction source data, "
        "attention/perturbation controls, similarity stratification, and summary tables.\n",
        encoding="utf-8",
    )

    if args.archive:
        tar=out.with_suffix(".tar.gz")
        with tarfile.open(tar,"w:gz") as tf:
            tf.add(out,arcname=out.name)
        print("Archive:",tar)
    print("PASS — EPFL return bundle built:",out)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
