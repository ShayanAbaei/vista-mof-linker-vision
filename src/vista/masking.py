"""
Attention masking utilities for the label-token Transformer.

The default model is **unmasked** (the label self-attention diagonal mask was
found to be harmful and has been removed). The only masking that remains is the
*directed label-edge block*, used exclusively by the directed-communication
ablation (block specific label->label edges during training/evaluation).

--------------------------------------------------------------------------------
Edge-direction convention (read this before interpreting any mask)
--------------------------------------------------------------------------------
The attention matrix produced by a Transformer layer has shape
``(B, n_heads, S, S)`` where ``S = n_label_tokens + n_image_tokens``
(here ``10 + 49 = 59``). For a single head the cell ``A[i, j]`` is the weight
that **query token i** places on **key/value token j**:

    * matrix cell                : ``A[i, j]``
    * query (receiver)           : token ``i``
    * key / value (source)       : token ``j``
    * information-flow edge      : ``j -> i``   (value of j flows into i's update)

Therefore *blocking* attention entry ``[query_i, key_j]`` (setting ``A[i, j]``
to ``-inf`` before softmax) means **information from label token j can no longer
be directly read by label token i**, i.e. it deletes the directed edge ``j -> i``.

Masks are pure functions of the configuration, add **no learned parameters**, and
are applied to the attention *logits* before softmax. Every blocked row keeps the
49 image keys (plus the remaining label keys) valid, so softmax never sees an
all-masked row and produces no NaNs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

# The diagonal self-mask mode has been removed; only "none" is supported. The
# ``mode`` field is retained on the config (always "none") for checkpoint-dict
# stability. Directed edge ablations use ``blocked_label_edges`` instead.
MASK_MODE_NONE = "none"
SUPPORTED_MASK_MODES = (MASK_MODE_NONE,)


@dataclass
class AttentionMaskConfig:
    """
    Configuration for the directed label-edge attention mask.

    Parameters
    ----------
    n_label_tokens : int
        Number of leading label tokens in the sequence (10 for this project).
    mode : str
        Always ``"none"`` (the diagonal self-mask was removed). Kept for
        checkpoint-dict compatibility.
    masked_layers : Optional[Sequence[int]]
        Indices of the encoder layers (0-based) at which the edge mask is applied.
        ``None`` means *all* layers. For "final three layers only" with 6 layers,
        use ``[3, 4, 5]``.
    blocked_label_edges : Optional[Sequence[Tuple[int, int]]]
        Directed label-edge ablations. Each pair ``(i, j)`` blocks attention entry
        ``A[i, j]`` (query label ``i`` reading key label ``j``), i.e. it deletes
        the information-flow edge ``j -> i``. Both indices must be in
        ``[0, n_label_tokens)``.
    """

    n_label_tokens: int = 10
    mode: str = MASK_MODE_NONE
    masked_layers: Optional[Sequence[int]] = None
    blocked_label_edges: Optional[Sequence[Tuple[int, int]]] = None

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_MASK_MODES:
            raise ValueError(
                f"Unsupported attention_mask_mode={self.mode!r}; the diagonal "
                f"self-mask was removed -- only {SUPPORTED_MASK_MODES} is valid. "
                f"Use blocked_label_edges for directed-edge ablations."
            )
        if self.blocked_label_edges is not None:
            for (i, j) in self.blocked_label_edges:
                if not (0 <= i < self.n_label_tokens and 0 <= j < self.n_label_tokens):
                    raise ValueError(
                        f"blocked_label_edge ({i}, {j}) out of range for "
                        f"n_label_tokens={self.n_label_tokens}."
                    )

    # --- helpers -------------------------------------------------------------
    def is_active(self) -> bool:
        """True if this config produces any masking (i.e. has blocked edges)."""
        return bool(self.blocked_label_edges)

    def resolve_masked_layers(self, n_layers: int) -> List[int]:
        """Return the concrete list of layer indices the mask applies to."""
        if self.masked_layers is None:
            return list(range(n_layers))
        layers = [int(l) for l in self.masked_layers]
        for l in layers:
            if not (0 <= l < n_layers):
                raise ValueError(
                    f"masked_layers contains {l}, out of range for "
                    f"n_layers={n_layers}."
                )
        return layers

    def to_dict(self) -> dict:
        """JSON/checkpoint-serialisable representation."""
        return {
            "n_label_tokens": int(self.n_label_tokens),
            "mode": str(self.mode),
            "masked_layers": (None if self.masked_layers is None
                              else [int(l) for l in self.masked_layers]),
            "blocked_label_edges": (None if self.blocked_label_edges is None
                                    else [(int(i), int(j)) for (i, j) in self.blocked_label_edges]),
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "AttentionMaskConfig":
        if d is None:
            return cls()
        edges = d.get("blocked_label_edges")
        if edges is not None:
            edges = [tuple(e) for e in edges]
        return cls(
            n_label_tokens=int(d.get("n_label_tokens", 10)),
            mode=MASK_MODE_NONE,  # diagonal mode removed; force none
            masked_layers=d.get("masked_layers"),
            blocked_label_edges=edges,
        )


def build_attention_block_mask(
    seq_len: int,
    config: AttentionMaskConfig,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """
    Build a boolean *block* mask for one Transformer layer.

    Returns a boolean ``(S, S)`` tensor where ``True`` marks an entry to block
    (set to ``-inf`` before softmax), or ``None`` if no edges are blocked. The
    mask broadcasts over the ``(B, n_heads, S, S)`` logits.

    Convention: ``mask[i, j] == True`` blocks ``A[i, j]`` (query ``i`` reading key
    ``j``), deleting information-flow edge ``j -> i``.
    """
    if not config.is_active():
        return None

    n_lab = config.n_label_tokens
    if n_lab > seq_len:
        raise ValueError(f"n_label_tokens={n_lab} exceeds seq_len={seq_len}.")

    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    for (i, j) in config.blocked_label_edges:
        mask[i, j] = True  # block query i reading key j  =>  delete j -> i

    # Safety: no attention row may be fully masked (softmax would yield NaN).
    fully_blocked = mask.all(dim=-1)
    if bool(fully_blocked.any()):
        bad = torch.nonzero(fully_blocked, as_tuple=False).flatten().tolist()
        raise ValueError(
            f"Attention mask fully blocks row(s) {bad}; every query must keep "
            f"at least one valid key. Check blocked_label_edges."
        )
    return mask


def apply_block_mask_to_logits(
    attention_logits: torch.Tensor,
    block_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Apply a boolean block mask to attention logits before softmax.

    Blocked entries are set to ``-inf`` so ``softmax`` assigns them exactly zero
    probability (no NaNs, since at least one key per row stays valid). Returns a
    new tensor; the input is not modified in place.
    """
    if block_mask is None:
        return attention_logits
    return attention_logits.masked_fill(block_mask, float("-inf"))


def directed_edge_to_matrix_cell(source_j: int, receiver_i: int) -> Tuple[int, int]:
    """
    Translate a directed information-flow edge ``j -> i`` into the attention
    matrix cell ``A[i, j]`` that carries it.

    Returns ``(i, j)`` = ``(receiver, source)``.
    """
    return (receiver_i, source_j)
