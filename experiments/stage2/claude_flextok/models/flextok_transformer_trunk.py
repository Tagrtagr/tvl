"""
FlexTok-style Transformer trunk (experiments version).

This mirrors the high-level structure of `ml-flextok/flextok/model/trunks/transformers.py`:
- modules read from / write to a `data_dict` using explicit keys
- trunk is reusable and can be composed into a module pipeline

We intentionally implement this trunk using standard PyTorch `nn.TransformerEncoderLayer`
to avoid depending on FlexAttention / fairscale / mup from `ml-flextok`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

__all__ = ["FlexTransformer"]


def get_from_dict(data_dict: Dict[str, Any], key: Optional[str], default=None):
    return data_dict[key] if key is not None else default


class FlexTransformer(nn.Module):
    """
    Dict-based Transformer trunk.

    Args:
        input_seq_read_key: Key to read the input sequence (B, L, D).
        output_seq_write_key: Key to write the output sequence (B, L, D).
        dim: Feature dimension D.
        depth: Number of transformer encoder layers.
        n_heads: Number of attention heads.
        mlp_ratio: FFN expansion ratio.
        drop: Dropout rate.
        attn_mask_read_key: Optional key to read an additive attention mask (L, L).
            Mask values should be 0 for allowed and -inf for disallowed (PyTorch convention).
        key_padding_mask_read_key: Optional key to read a key padding mask (B, L), True=ignore.
    """

    def __init__(
        self,
        input_seq_read_key: str,
        output_seq_write_key: str,
        *,
        dim: int,
        depth: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_mask_read_key: Optional[str] = None,
        key_padding_mask_read_key: Optional[str] = None,
    ):
        super().__init__()
        self.input_seq_read_key = input_seq_read_key
        self.output_seq_write_key = output_seq_write_key
        self.attn_mask_read_key = attn_mask_read_key
        self.key_padding_mask_read_key = key_padding_mask_read_key

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        x = data_dict[self.input_seq_read_key]
        attn_mask = get_from_dict(data_dict, self.attn_mask_read_key, default=None)
        key_padding_mask = get_from_dict(
            data_dict, self.key_padding_mask_read_key, default=None
        )

        x = self.encoder(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        data_dict[self.output_seq_write_key] = x
        return data_dict

