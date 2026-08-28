#!/usr/bin/env python3
"""CPU model-contract smoke test; does not instantiate/download Swin-B."""
from __future__ import annotations
import torch
from _bootstrap import add_vista_legacy_path
add_vista_legacy_path()
from ml_vit import MultiLabelVisionTransformer

def main():
    torch.manual_seed(42)
    m=MultiLabelVisionTransformer(
        feature_size=(7,7),
        feature_channels=1024,
        embed_dim=384,
        n_encoder_layers=6,
        n_heads=12,
        n_labels=10,
    ).eval()
    x=torch.randn(1,1024,7,7)
    with torch.no_grad():
        logits,attn=m(x,return_attention_weights=True)
    assert tuple(logits.shape)==(1,10),logits.shape
    assert len(attn)==6,len(attn)
    assert tuple(attn[0].shape)==(1,12,59,59),attn[0].shape
    for a in attn:
        assert torch.isfinite(a).all()
        assert torch.allclose(a.sum(-1),torch.ones_like(a.sum(-1)),atol=1e-4)
    assert m.n_labels==10 and m.n_patches==49 and m.seq_len==59
    print("PASS — VISTA model contract")
    print("logits:",tuple(logits.shape))
    print("attention layers:",len(attn))
    print("attention shape:",tuple(attn[0].shape))
    print("tokens:",m.n_labels,"+",m.n_patches,"=",m.seq_len)
if __name__=="__main__":
    main()
