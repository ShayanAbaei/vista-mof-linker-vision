# Historical VISTA Reference Contract

The supplied historical archive resolves several provenance questions.

## Executed sample-level run
- global seed: 42
- executed epochs: 50
- best validation macro-AUPRC epoch: 44
- best validation macro-AUPRC: 0.9777503420277581
- historical test macro-AUPRC: 0.9879242367586224
- historical test macro-F1: 0.9604032151282652
- historical test micro-AUPRC: 0.997934841258488
- epoch-50 head LR: 1e-6

## Historical software snapshot
From the archived execution manifest:
- Python 3.12.13
- PyTorch 2.11.0+cu128
- torchvision 0.26.0+cu128
- scikit-learn 1.6.1
- NumPy 2.0.2
- matplotlib 3.10.0
- historical GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition

## 50-vs-100 LR artifact
The archived LR CSV contains 100 plotted rows, but the scheduler config has
`total_epochs=50`, the LR reaches its minimum at epoch 50, and the actual model
history contains exactly 50 epochs. Therefore the 100-row plot is not the training
length.

## Baseline uncertainty warning
The archived `baseline_auprc.csv` has one seed per label (`n_seeds=1`), zero
standard deviation, and CI endpoints equal to the mean. These are reference
single-seed values, not multi-seed uncertainty estimates.

## E0 purpose
E0 exists to verify that the hardened execution path reproduces the historical
sample-level experiment closely enough before launching the strict split.
It is not a new headline result.
