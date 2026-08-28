"""
Deterministic seeding for full reproducibility (spec section 4).

Covers Python, NumPy, PyTorch (CPU + CUDA), DataLoader workers, graph layouts
and Bayesian-bootstrap sampling. A single ``set_all_seeds`` call plus the
``seed_worker`` / ``make_generator`` helpers make every stochastic step
reproducible.
"""
from __future__ import annotations

import os
import random
from typing import Dict, Optional

import numpy as np


def set_all_seeds(seed: int = 1234, deterministic_torch: bool = True) -> Dict[str, int]:
    """
    Seed Python, NumPy and PyTorch (CPU + CUDA) and request deterministic algos.

    Returns the dict of seeds used so it can be stored in checkpoints / manifests.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # cuBLAS workspace config is required for deterministic matmul on CUDA.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except Exception:
        pass

    return {
        "global": seed,
        "python": seed,
        "numpy": seed,
        "torch": seed,
        "cuda": seed,
    }


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for DataLoader workers (derives from torch initial seed)."""
    import torch
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 1234):
    """A seeded ``torch.Generator`` for DataLoader shuffling reproducibility."""
    import torch
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def make_rng(seed: int = 1234) -> np.random.Generator:
    """A seeded NumPy ``Generator`` (use for Bayesian bootstrap, etc.)."""
    return np.random.default_rng(seed)
