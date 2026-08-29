#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

from _bootstrap import add_vista_legacy_path
add_vista_legacy_path()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default="outputs/epfl_runtime_gate")
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--batch-size",type=int,default=32)
    ap.add_argument("--require-gpu-count",type=int,default=2)
    args=ap.parse_args()

    import torch, torchvision
    import experiments

    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    try:
        nvsmi=subprocess.check_output(["nvidia-smi"],text=True)
    except Exception as e:
        nvsmi=f"nvidia-smi unavailable: {e}\n"
    (out/"nvidia_smi.txt").write_text(nvsmi,encoding="utf-8")

    info={
        "python":sys.version,
        "torch":torch.__version__,
        "torchvision":torchvision.__version__,
        "torch_cuda_runtime":torch.version.cuda,
        "cuda_available":torch.cuda.is_available(),
        "gpu_count":torch.cuda.device_count(),
        "gpus":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    # CUDA_VISIBLE_DEVICES may intentionally expose one physical GPU to a job.
    physical_visible = torch.cuda.device_count()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None and physical_visible < args.require_gpu_count:
        raise RuntimeError(f"Expected >= {args.require_gpu_count} visible GPUs, found {physical_visible}")

    cfg=experiments.ExperimentConfig(name="runtime_gate",seeds=(42,))
    model=experiments.build_label_token_model(cfg,weights="default").to(args.device)
    model.train()
    model.unfreeze_backbone()

    x=torch.randn(args.batch_size,3,224,224,device=args.device)
    logits,attn=model(x,return_attention_weights=True)
    assert tuple(logits.shape)==(args.batch_size,10)
    assert len(attn)==6 and tuple(attn[0].shape)==(args.batch_size,12,59,59)
    loss=logits.square().mean()
    loss.backward()
    if not any(p.grad is not None for p in model.backbone.parameters() if p.requires_grad):
        raise RuntimeError("No backbone gradients after full-model backward")

    # Checkpoint round-trip.
    ck=out/"runtime_gate_checkpoint.pt"
    torch.save({"model_state_dict":model.state_dict(),"shape":list(logits.shape)},ck)
    loaded=torch.load(ck,map_location="cpu",weights_only=False)
    assert "model_state_dict" in loaded
    ck.unlink(missing_ok=True)

    info.update({
        "batch_size":args.batch_size,
        "logits_shape":list(logits.shape),
        "attention_layers":len(attn),
        "attention_shape":list(attn[0].shape),
        "full_model_backward":"PASS",
        "checkpoint_roundtrip":"PASS",
        "status":"PASS",
    })
    (out/"runtime_gate.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    print(json.dumps(info,indent=2))
    print("PASS — EPFL GPU runtime gate")

if __name__=="__main__":
    main()
