#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, platform, subprocess, sys
from pathlib import Path

def ver(mod):
    try:
        m=__import__(mod)
        return getattr(m,"__version__","unknown")
    except Exception as e:
        return f"NOT_AVAILABLE: {type(e).__name__}: {e}"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default="docs/environment_snapshot.json")
    args=ap.parse_args()
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    payload={
        "python":sys.version,
        "platform":platform.platform(),
        "numpy":ver("numpy"),
        "pandas":ver("pandas"),
        "sklearn":ver("sklearn"),
        "matplotlib":ver("matplotlib"),
        "PIL":ver("PIL"),
        "torch":ver("torch"),
        "torchvision":ver("torchvision"),
    }
    try:
        import torch
        payload["torch_cuda_runtime"]=torch.version.cuda
        payload["cuda_available"]=torch.cuda.is_available()
        payload["cuda_device_count"]=torch.cuda.device_count()
        payload["cuda_devices"]=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as e:
        payload["torch_cuda_error"]=repr(e)
    try:
        payload["git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception:
        payload["git_commit"]=None
    out.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))
    print("WROTE:",out)
if __name__=="__main__":
    main()
