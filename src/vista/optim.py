"""
Optimizer and learning-rate-schedule construction for the label-token model.

Design goals (see project spec section 3E):

* Build the head/Transformer **and** backbone parameter groups from the very
  beginning, so no new optimizer group is ever added after a scheduler has been
  constructed. The backbone starts frozen (``requires_grad=False``) and simply
  begins receiving updates once unfrozen — the LR ratio is preserved because
  both groups are scaled by the same schedule multiplier.
* A **warmup + cosine** schedule (continuous, explicitly documented):
    - epochs 1..warmup_epochs        : linear warmup from ``warmup_start_factor``
                                       (default 0.1x) up to 1.0x base LR
    - epochs warmup_epochs..total     : single half-cosine decay from 1.0x down to
                                       ``min_lr_ratio`` x base LR
    - epoch > total_epochs            : held at ``min_lr_ratio`` (cosine complete)
  Set ``warmup_epochs=0`` for a pure cosine schedule with no warmup. Both param
  groups share the same multiplier, so the head/backbone LR ratio is preserved.
* Save the planned schedule (head + backbone LRs) as CSV and PDF before training.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple, Union

import torch


# --------------------------------------------------------------------------- #
# Parameter groups / optimizer
# --------------------------------------------------------------------------- #
def split_parameters(model: torch.nn.Module) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """
    Split a ``BackboneWithModel`` (or any module) into (backbone, head) params by
    name. Anything whose qualified name contains ``"backbone"`` is a backbone
    parameter; everything else (label tokens, Transformer, heads) is a head
    parameter.
    """
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        (backbone_params if 'backbone' in name else head_params).append(p)
    return backbone_params, head_params


def build_optimizer(
    model: torch.nn.Module,
    base_lr: float = 1e-4,
    backbone_lr_ratio: float = 0.1,
    weight_decay: float = 1e-2,
    optimizer_cls: type = torch.optim.AdamW,
    **optimizer_kwargs,
) -> torch.optim.Optimizer:
    """
    Build an optimizer with both head and backbone groups present from the start.

    The backbone group is included even while the backbone is frozen; frozen
    parameters receive no updates (their grads stay ``None``) until unfrozen.
    Because both groups exist up front, the scheduler can be constructed
    afterwards without ever needing ``add_param_group`` — which would otherwise
    desynchronise a ``LambdaLR``'s ``base_lrs``.

    Group 0 = head/Transformer (lr = ``base_lr``).
    Group 1 = backbone (lr = ``base_lr * backbone_lr_ratio``).
    """
    backbone_params, head_params = split_parameters(model)
    param_groups = [
        {'params': head_params, 'lr': base_lr, 'name': 'head'},
        {'params': backbone_params, 'lr': base_lr * backbone_lr_ratio, 'name': 'backbone'},
    ]
    return optimizer_cls(param_groups, weight_decay=weight_decay, **optimizer_kwargs)


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #
@dataclass
class LRScheduleConfig:
    """Parameters of the warmup + cosine schedule.

    ``warmup_epochs=0`` gives a pure cosine schedule (no warmup). The cosine
    decays from 1.0x base LR at the end of warmup to ``min_lr_ratio`` x base LR
    at ``total_epochs`` and is held at the minimum afterwards.
    """
    warmup_start_factor: float = 0.1   # 0.1x base LR at epoch 1
    warmup_epochs: int = 10            # end of linear warmup (reaches 1.0x); 0 = no warmup
    total_epochs: int = 50             # epoch at which cosine reaches the minimum
    min_lr_ratio: float = 0.01         # minimum LR as a fraction of base LR

    def to_dict(self) -> Dict:
        return asdict(self)


def schedule_factor(epoch: int, cfg: LRScheduleConfig) -> float:
    """
    Multiplicative LR factor for a **1-based** epoch number.

    Returns the factor that should multiply the base LR *during* ``epoch``:
    linear warmup up to 1.0x, then a single half-cosine decay to
    ``min_lr_ratio``, held at the minimum beyond ``total_epochs``. Continuous at
    the warmup/cosine boundary.
    """
    m = cfg.min_lr_ratio
    if cfg.warmup_epochs > 0 and epoch <= cfg.warmup_epochs:
        # Linear from warmup_start_factor (epoch 1) to 1.0 (epoch warmup_epochs).
        if cfg.warmup_epochs == 1:
            return 1.0
        denom = cfg.warmup_epochs - 1
        return cfg.warmup_start_factor + (1.0 - cfg.warmup_start_factor) * (epoch - 1) / denom
    # Cosine decay from the end of warmup to total_epochs (held at min after).
    start = cfg.warmup_epochs
    span = max(1, cfg.total_epochs - start)
    progress = min(1.0, max(0.0, (epoch - start) / span))
    return m + (1.0 - m) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_lr_lambda(cfg: LRScheduleConfig) -> Callable[[int], float]:
    """
    Build a ``LambdaLR`` factor function.

    ``torch``'s ``LambdaLR`` passes a 0-based step counter (0 during epoch 1,
    after the in-``__init__`` step). We map ``step -> schedule_factor(step + 1)``
    so the factor applied *during* epoch ``e`` equals ``schedule_factor(e)`` and
    matches the saved schedule plot exactly.
    """
    return lambda step: schedule_factor(step + 1, cfg)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: LRScheduleConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Construct the LambdaLR. Both param groups share the same factor, so the
    head/backbone LR ratio is preserved throughout training."""
    fn = make_lr_lambda(cfg)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[fn] * len(optimizer.param_groups))


def compute_lr_schedule(
    n_epochs: int,
    base_lr: float,
    backbone_lr_ratio: float,
    cfg: LRScheduleConfig,
) -> Dict[str, List[float]]:
    """Return per-epoch head and backbone LRs for epochs 1..n_epochs."""
    epochs = list(range(1, n_epochs + 1))
    factors = [schedule_factor(e, cfg) for e in epochs]
    return {
        'epoch': epochs,
        'factor': factors,
        'head_lr': [base_lr * f for f in factors],
        'backbone_lr': [base_lr * backbone_lr_ratio * f for f in factors],
    }


# Academic plotting rcParams (embedded TrueType fonts, serif).
_ACADEMIC_RC = {
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'STIXGeneral'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
}


def save_lr_schedule(
    out_dir: Union[str, Path],
    n_epochs: int,
    base_lr: float,
    backbone_lr_ratio: float = 0.1,
    cfg: LRScheduleConfig = None,
    filename: str = 'lr_schedule',
) -> Dict[str, Path]:
    """
    Save the planned LR schedule as CSV + PDF (head and backbone plotted
    separately) plus a companion JSON of the configuration. Call this **before**
    training so the intended schedule is recorded.

    Returns a dict of the written paths.
    """
    cfg = cfg or LRScheduleConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sched = compute_lr_schedule(n_epochs, base_lr, backbone_lr_ratio, cfg)

    csv_path = out_dir / f'{filename}.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'factor', 'head_lr', 'backbone_lr'])
        for i in range(n_epochs):
            w.writerow([sched['epoch'][i], sched['factor'][i],
                        sched['head_lr'][i], sched['backbone_lr'][i]])

    json_path = out_dir / f'{filename}.json'
    with open(json_path, 'w') as f:
        json.dump({'config': cfg.to_dict(), 'base_lr': base_lr,
                   'backbone_lr_ratio': backbone_lr_ratio, 'n_epochs': n_epochs}, f, indent=2)

    pdf_path = out_dir / f'{filename}.pdf'
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        with plt.rc_context(_ACADEMIC_RC):
            fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
            ax.plot(sched['epoch'], sched['head_lr'], label='Head / Transformer LR', lw=1.6)
            ax.plot(sched['epoch'], sched['backbone_lr'], label='Backbone LR', lw=1.6, ls='--')
            for boundary in (cfg.warmup_epochs, cfg.total_epochs):
                if 0 < boundary <= n_epochs:
                    ax.axvline(boundary, color='0.7', lw=0.8, ls=':')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning rate')
            ax.set_yscale('log')
            ax.set_title('Planned learning-rate schedule (warmup + cosine)')
            ax.legend(frameon=False)
            fig.savefig(pdf_path)
            plt.close(fig)
    except Exception as e:  # pragma: no cover - plotting is best-effort
        pdf_path = None
        print(f"[optim] LR schedule PDF not written ({e}); CSV/JSON saved.")

    return {'csv': csv_path, 'json': json_path, 'pdf': pdf_path}
