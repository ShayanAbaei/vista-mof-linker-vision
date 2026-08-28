"""
Attention extraction helpers for the label-token Transformer.

These operate on the per-layer attention tensors returned by
``MultiLabelVisionTransformer(..., return_attention_weights=True)``: a list of
``n_layers`` tensors, each shaped ``(B, n_heads, S, S)`` with
``S = n_label_tokens + n_image_tokens`` (10 + 49 = 59 here).

Block layout of each ``S x S`` attention matrix (rows = queries, cols = keys):

    label-to-label : ``A[:n_lab, :n_lab]``
    label-to-image : ``A[:n_lab, n_lab:]``
    image-to-label : ``A[n_lab:, :n_lab]``
    image-to-image : ``A[n_lab:, n_lab:]``

Direction convention (see ``masking``): ``A[i, j]`` is query ``i`` attending key
``j`` — information flows ``j -> i``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch


def split_attention_blocks(attn: torch.Tensor, n_labels: int) -> Dict[str, torch.Tensor]:
    """
    Split one layer's attention ``(B, H, S, S)`` into the four named blocks.

    Returns a dict with keys ``label_label`` ``(B,H,L,L)``, ``label_image``
    ``(B,H,L,P)``, ``image_label`` ``(B,H,P,L)``, ``image_image`` ``(B,H,P,P)``.
    """
    L = n_labels
    return {
        'label_label': attn[..., :L, :L],
        'label_image': attn[..., :L, L:],
        'image_label': attn[..., L:, :L],
        'image_image': attn[..., L:, L:],
    }


def label_to_label_matrix(
    attn: torch.Tensor,
    n_labels: int,
    head_reduction: str = 'mean',
) -> torch.Tensor:
    """
    Extract the label-to-label block and reduce over heads -> ``(B, L, L)``.
    ``head_reduction`` is ``'mean'`` (default) or ``'none'`` (keep ``(B,H,L,L)``).
    """
    block = attn[..., :n_labels, :n_labels]
    if head_reduction == 'none':
        return block
    if head_reduction == 'mean':
        return block.mean(dim=1)
    raise ValueError(f"Unknown head_reduction={head_reduction!r}")


def label_to_image_maps(
    attn: torch.Tensor,
    n_labels: int,
    grid_hw: Tuple[int, int] = (7, 7),
    head_reduction: str = 'mean',
) -> torch.Tensor:
    """
    Extract the label-to-image block and reshape the 49 image keys to the spatial
    grid -> ``(B, L, gh, gw)`` (after mean over heads). The 49 image tokens are
    assumed to be in row-major order matching the Swin 7x7 feature map (the same
    order produced by ``FeatureEmbedding.forward``'s ``flatten(2)``).
    """
    gh, gw = grid_hw
    block = attn[..., :n_labels, n_labels:]  # (B, H, L, P)
    if head_reduction == 'mean':
        block = block.mean(dim=1)             # (B, L, P)
    elif head_reduction != 'none':
        raise ValueError(f"Unknown head_reduction={head_reduction!r}")
    B, L, P = block.shape[0], block.shape[-2], block.shape[-1]
    if P != gh * gw:
        raise ValueError(f"image-token count {P} != grid {gh}x{gw}={gh*gw}")
    return block.reshape(*block.shape[:-1], gh, gw)


def label_to_image_mass(attn: torch.Tensor, n_labels: int,
                        head_reduction: str = 'mean') -> torch.Tensor:
    """
    Total label-to-image attention mass per label *before* spatial normalisation
    -> ``(B, L)``. This is the fraction of each label query's attention spent on
    image tokens (vs other label tokens).
    """
    block = attn[..., :n_labels, n_labels:]
    if head_reduction == 'mean':
        block = block.mean(dim=1)
    return block.sum(dim=-1)


def attention_rollout(
    attn_list: Sequence[torch.Tensor],
    add_residual: bool = True,
    head_reduction: str = 'mean',
) -> torch.Tensor:
    """
    Attention rollout (Abnar & Zuidema, 2020).

    Steps, per the project spec:
      1. mean attention across heads per layer,
      2. add the residual identity contribution (``0.5*(A + I)``),
      3. row-normalise,
      4. multiply the matrices through all layers in order (first layer applied
         first), i.e. ``R = A_L @ ... @ A_2 @ A_1``,
    returning the rollout matrix ``(B, S, S)``. Row ``i`` of ``R`` gives the
    fraction of token ``i``'s information attributable to each input token.
    """
    if len(attn_list) == 0:
        raise ValueError("attn_list is empty")

    device = attn_list[0].device
    B, _, S, _ = attn_list[0].shape
    eye = torch.eye(S, device=device).unsqueeze(0)  # (1, S, S)
    rollout = eye.expand(B, S, S).clone()

    for attn in attn_list:
        if head_reduction == 'mean':
            a = attn.mean(dim=1)  # (B, S, S)
        elif head_reduction == 'none':
            a = attn
        else:
            raise ValueError(f"Unknown head_reduction={head_reduction!r}")
        if add_residual:
            a = a + eye
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rollout = torch.bmm(a, rollout)
    return rollout


def rollout_label_to_image_maps(
    attn_list: Sequence[torch.Tensor],
    n_labels: int,
    grid_hw: Tuple[int, int] = (7, 7),
) -> torch.Tensor:
    """Run rollout then extract the label-to-image block reshaped to the grid
    -> ``(B, L, gh, gw)``."""
    R = attention_rollout(attn_list)            # (B, S, S)
    gh, gw = grid_hw
    block = R[:, :n_labels, n_labels:]          # (B, L, P)
    return block.reshape(block.shape[0], n_labels, gh, gw)
