# EPFL GPU runtime gate

Run before **any** scientific job:

```bash
bash launch/00_runtime_gate.sh
```

Required:
- CUDA available;
- pretrained torchvision Swin-B loads;
- VISTA forward shape `(32,10)`;
- six attention tensors with `(32,12,59,59)`;
- full-model backward after backbone unfreeze;
- checkpoint save/load roundtrip.

The workstation previously showed two RTX A6000 GPUs (~49 GB each), driver 595.71.05.
`nvidia-smi` CUDA compatibility is not the PyTorch CUDA runtime.
