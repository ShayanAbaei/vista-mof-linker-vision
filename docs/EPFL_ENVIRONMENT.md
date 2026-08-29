# EPFL environment — frozen setup

The historical VISTA run recorded approximately:

```text
Python       3.12.13
PyTorch      2.11.0 + cu128
torchvision  0.26.0 + cu128
NumPy        2.0.2
scikit-learn 1.6.1
matplotlib   3.10.0
```

## Core VISTA/shared-Swin environment

```bash
conda env create -f environment-epfl-core.yml
conda activate vista-epfl

python -m pip install \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```bash
python - <<'PY'
import torch, torchvision
print(torch.__version__)
print(torchvision.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY
```

The NVIDIA driver's `nvidia-smi` "CUDA Version" line is driver compatibility, not the
PyTorch CUDA runtime.

## GIN-family environment

Do not destabilize the core reproduction environment.

```bash
conda create -n vista-epfl-gin --clone vista-epfl
conda activate vista-epfl-gin
python -m pip install -r requirements-gin.txt
```

PyG >=2.3 can run a basic model from a minimal `torch_geometric` install. If optional
compiled extensions are needed, follow PyG's wheel matrix for torch 2.11 + cu128 and
record every installed package in the run output.

## Pretrained Swin-B

The first runtime gate loads torchvision's frozen Swin-B ImageNet weights. If the
workstation cannot reach the download host, stop and arrange a cached torchvision
weight file rather than silently switching weights.
