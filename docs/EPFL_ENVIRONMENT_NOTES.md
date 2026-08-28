# EPFL environment notes

Observed GPU workstation:
- 2 × NVIDIA RTX A6000
- ~48 GB VRAM each
- NVIDIA driver 595.71.05
- `nvidia-smi` advertises CUDA compatibility up to 13.2

`nvidia-smi`'s CUDA line is not the installed PyTorch runtime.

Historical VISTA target:
- Python 3.12.13
- PyTorch 2.11.0 + cu128
- torchvision 0.26.0 + cu128
- NumPy 2.0.2
- scikit-learn 1.6.1
- matplotlib 3.10.0

On EPFL, record before training:

```bash
nvidia-smi
python --version
python - <<'PY'
import torch, torchvision
print(torch.__version__)
print(torchvision.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

Do not use distributed two-GPU training for one seed unless a concrete need appears. Use one independent run per GPU.
