#!/usr/bin/env python3
from __future__ import annotations

import argparse, compileall, json, os, re, subprocess, sys
from pathlib import Path
import pandas as pd, yaml

REQUIRED=[
    "configs/vista_sample_repro.yaml","configs/vista_group_disjoint.yaml",
    "configs/shared_swin_group_disjoint.yaml","configs/gin_group_disjoint.yaml",
    "configs/dimred.yaml","configs/attention_controls.yaml","configs/e0_acceptance.yaml",
    "configs/mof_consequence.yaml",
    "scripts/train_vista.py","scripts/train_shared_swin.py","scripts/train_gin.py",
    "scripts/export_embeddings.py","scripts/make_dimred_source.py",
    "scripts/export_attention_cases.py","scripts/run_attention_controls.py",
    "scripts/run_negative_attention_controls.py","scripts/run_rendering_stability.py",
    "scripts/performance_by_tanimoto.py","scripts/run_mof_level_consequence.py",
    "scripts/build_epfl_return_bundle.py","scripts/e0_acceptance_gate.py",
    "scripts/epfl_runtime_gate.py","scripts/fetch_zenodo_images.py",
    "scripts/validate_output_contract.py",
    "docs/EPFL_HANDOVER.md","docs/OUTPUT_CONTRACT.md","docs/GPU_RUNTIME_GATE.md",
    "docs/EPFL_ENVIRONMENT.md","docs/FRESH_CLONE_REHEARSAL.md",
    "environment-epfl-core.yml","requirements-gin.txt",
]

def validate_shell_text(path: Path):
    failures=[]
    raw=path.read_bytes()
    txt=raw.decode("utf-8")
    if "\r\n" in txt:
        failures.append(f"CRLF line endings in {path}")
    if not txt.startswith("#!/usr/bin/env bash"):
        failures.append(f"missing bash shebang in {path}")
    if re.search(r"[A-Za-z]:\\",txt):
        failures.append(f"Windows path embedded in {path}")
    if "<REPLACE" in txt or "<PRIVATE_REPO" in txt:
        failures.append(f"unresolved placeholder in {path}")
    return failures

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",default=".")
    ap.add_argument("--require-images",action="store_true")
    args=ap.parse_args()
    repo=Path(args.repo_root).resolve()
    failures=[]

    print("="*78)
    print("VISTA PRE-EPFL REHEARSAL")
    print("Host:",sys.platform)
    print("Repo:",repo)

    for rel in REQUIRED:
        if not (repo/rel).exists():
            failures.append("missing "+rel)

    for rel in [
        "configs/shared_swin_group_disjoint.yaml","configs/gin_group_disjoint.yaml",
        "configs/dimred.yaml","configs/attention_controls.yaml",
        "configs/e0_acceptance.yaml","configs/mof_consequence.yaml",
    ]:
        try:
            yaml.safe_load((repo/rel).read_text(encoding="utf-8"))
        except Exception as e:
            failures.append(f"yaml {rel}: {e}")

    if not compileall.compile_dir(repo/"scripts",quiet=1):
        failures.append("scripts compileall failed")
    if not compileall.compile_dir(repo/"src/vista",quiet=1):
        failures.append("src/vista compileall failed")

    pa=repo/"data/frozen/PHASE_A_VALIDATION_REPORT.json"
    if not pa.exists() or json.loads(pa.read_text()).get("status")!="PASS":
        failures.append("Phase A validation not PASS")

    pc=repo/"phase_c_results/report/PHASE_C_STATUS.json"
    if not pc.exists() or json.loads(pc.read_text()).get("status")!="PASS":
        failures.append("Phase C status not PASS")

    strict_path=repo/"data/frozen/canonical_group_disjoint_split.csv"
    strict=pd.read_csv(strict_path)
    if len(strict)!=9414:
        failures.append(f"strict rows {len(strict)} != 9414")
    counts=strict["split"].value_counts().to_dict()
    if counts!={"train":6590,"validation":1412,"test":1412}:
        failures.append(f"strict split counts wrong: {counts}")
    if strict.groupby("canonical_first_linker_smiles")["split"].nunique().max()!=1:
        failures.append("strict split is not canonical-first-linker disjoint")

    shell_files=sorted((repo/"launch").glob("*.sh"))
    for sh in shell_files:
        failures.extend(validate_shell_text(sh))

    # IMPORTANT:
    # On Windows, "bash" usually resolves to WSL. Passing a native D:\... path
    # directly to WSL bash mangles the path. Therefore we DO NOT invoke
    # `bash -n D:\...` from Windows. The release package itself is bash-syntax
    # checked on Linux, and on EPFL/Linux we repeat bash -n using POSIX paths.
    if os.name=="nt":
        print(f"[OK] Windows host: validated {len(shell_files)} launch scripts textually.")
        print("[SKIP] bash -n on Windows (prevents WSL D:\\ path translation failure).")
    else:
        for sh in shell_files:
            p=subprocess.run(["bash","-n",str(sh)],capture_output=True,text=True)
            if p.returncode:
                failures.append(f"bash syntax {sh}: {p.stderr.strip()}")
        print(f"[OK] POSIX host: bash -n validated {len(shell_files)} launch scripts.")

    # Frozen shared architecture contract.
    base=(repo/"src/vista/baselines.py").read_text(encoding="utf-8")
    train_shared=(repo/"scripts/train_shared_swin.py").read_text(encoding="utf-8")
    if "class SharedCNNMultiLabel" not in base or "nn.ModuleList([nn.Linear(backbone.feature_dim, 1)" not in base:
        failures.append("frozen SharedCNNMultiLabel architecture not found in baselines.py")
    if "baselines.build_shared_cnn" not in train_shared:
        failures.append("train_shared_swin.py is not using frozen baselines.build_shared_cnn")
    if (repo/"src/vista/shared_swin.py").exists():
        failures.append("stale v1 src/vista/shared_swin.py still exists")

    if args.require_images:
        imgs=list((repo/"data/images_224").glob("*.png"))
        if len(imgs)!=9414:
            failures.append(f"image count {len(imgs)} != 9414")

    if failures:
        print("\nPRE-EPFL REHEARSAL: FAIL")
        for x in failures:
            print(" -",x)
        return 2

    print("\nPRE-EPFL REHEARSAL: PASS")
    print("GPU/Torch execution remains gated by scripts/epfl_runtime_gate.py on EPFL.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
