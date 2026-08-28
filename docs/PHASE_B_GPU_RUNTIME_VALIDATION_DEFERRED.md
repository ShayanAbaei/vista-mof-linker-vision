# Phase B GPU Runtime Validation — Deferred to EPFL

The local Windows machine intentionally does not have PyTorch/torchvision installed.

This is **not** a scientific-code failure. Phase B freezes data, source, configuration,
historical provenance, and evaluation policy locally. Tensor/GPU runtime validation is
deferred to the actual EPFL execution environment.

No E0/E1 scientific training may begin until the EPFL runtime gate verifies:
- torch/torchvision import;
- CUDA visibility;
- two RTX A6000 devices;
- 59-token model forward/attention contract;
- pretrained Swin-B availability;
- batch-32 forward/backward;
- checkpoint save/load;
- test-suite execution.

The local environment snapshot is retained as provenance and is not the target
training environment.
