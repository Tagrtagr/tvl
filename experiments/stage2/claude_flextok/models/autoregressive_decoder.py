"""
Autoregressive Transformer Decoder for register token reconstruction.

Replaces the previous ConvTranspose-only decoder with a standard transformer
decoder architecture where register tokens serve as encoder memory and
learnable spatial queries decode the image via cross-attention.

This is the approach used in FlexTok (Bachmann et al., 2025) for decoding
from variable-length token representations.

Key advantages over the ConvTranspose-only decoder:
  - Cross-attention to register tokens naturally handles variable-length input
  - Causal self-attention among spatial queries enables autoregressive generation
  - Prefix decoding is clean: just pass fewer memory tokens, no zero-padding

Architecture:
    register_tokens (B, n_reg, hidden_dim) → transformer memory
    spatial_queries  (1, n_spatial, hidden_dim) → decoder input (learnable)

    Decoder Layers (× N):
        1. Causal self-attention among spatial queries
        2. Cross-attention: queries attend to register tokens
        3. Feed-forward network

    decoded queries → reshape to (B, C, h0, w0) → ConvTranspose upsample → pixels

Reference: FlexTok (Bachmann et al., 2025) - arXiv 2502.13967
"""

import math

import torch
import torch.nn as nn


class AutoregressiveDecoder(nn.Module):
    """
    Decodes register tokens to pixel space via transformer cross-attention.

    Args:
        n_registers: Number of register tokens (memory length for cross-attention).
        hidden_dim: Dimension of register tokens and decoder queries.
        n_spatial: Number of spatial query positions (must be a perfect square).
            Default: 196 (14x14, matching ViT patch grid for 224x224 images).
        base_channels: Channel width after projecting decoded queries to spatial grid.
        output_channels: Output image channels (3 for RGB).
        output_size: Target spatial resolution (224).
        n_decoder_layers: Number of transformer decoder layers.
        n_heads: Attention heads per layer.
        dropout: Dropout rate in transformer layers.
    """

    def __init__(
        self,
        n_registers: int = 32,
        hidden_dim: int = 512,
        n_spatial: int = 196,  # 14x14
        base_channels: int = 64,
        output_channels: int = 3,
        output_size: int = 224,
        n_decoder_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_registers = n_registers
        self.hidden_dim = hidden_dim
        self.n_spatial = n_spatial
        self.output_size = output_size

        # Spatial grid dimensions
        self.h0 = int(math.sqrt(n_spatial))
        self.w0 = self.h0
        assert self.h0 * self.w0 == n_spatial, \
            f"n_spatial must be a perfect square, got {n_spatial}"

        # Learnable spatial queries (one per output patch position)
        self.spatial_queries = nn.Parameter(
            torch.randn(1, n_spatial, hidden_dim) * 0.02
        )

        # Positional embeddings for spatial queries
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, n_spatial, hidden_dim) * 0.02
        )

        # Transformer decoder: cross-attention to register tokens
        # + causal self-attention among spatial queries
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_decoder_layers,
        )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # Project decoded queries to spatial feature channels
        self.to_spatial = nn.Linear(hidden_dim, base_channels)

        # Progressive upsampling to output resolution
        # 14x14 → 224x224 needs 4 blocks (each 2x), 7x7 → 224 needs 5
        n_upsample = int(round(math.log2(output_size / self.h0)))
        c = base_channels
        layers = []
        for i in range(n_upsample):
            c_out = max(c // 2, 8) if i > 0 else c
            layers.append(_UpBlock(c, c_out))
            c = c_out
        self.upsample = nn.Sequential(*layers)

        # Final projection to pixel channels (no activation — output is
        # in normalized image space which can be negative)
        self.to_pixels = nn.Conv2d(c, output_channels, kernel_size=3, padding=1)

        # Causal mask for autoregressive self-attention among spatial queries
        causal = torch.triu(
            torch.full((n_spatial, n_spatial), float("-inf")), diagonal=1,
        )
        self.register_buffer("_causal_mask", causal, persistent=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.spatial_queries, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.to_spatial.weight)
        nn.init.zeros_(self.to_spatial.bias)
        nn.init.xavier_uniform_(self.to_pixels.weight)
        nn.init.zeros_(self.to_pixels.bias)

    def _decode(self, memory: torch.Tensor) -> torch.Tensor:
        """
        Core decode: cross-attend to memory tokens, reshape, upsample.

        Args:
            memory: (B, n_mem, hidden_dim) register tokens to attend to.

        Returns:
            (B, output_channels, output_size, output_size)
        """
        B = memory.shape[0]

        # Expand learnable queries for the batch
        queries = self.spatial_queries.expand(B, -1, -1) + self.spatial_pos_embed

        # Transformer decoder: causal self-attention + cross-attention to memory
        decoded = self.decoder(
            tgt=queries,
            memory=memory,
            tgt_mask=self._causal_mask,
        )
        decoded = self.decoder_norm(decoded)  # (B, n_spatial, hidden_dim)

        # Project to channel space and reshape to spatial grid
        x = self.to_spatial(decoded)  # (B, n_spatial, base_channels)
        x = x.transpose(1, 2).reshape(B, -1, self.h0, self.w0)

        # Upsample to full resolution
        x = self.upsample(x)

        # Project to pixels
        x = self.to_pixels(x)

        return x

    def forward(self, register_tokens: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct from all register tokens.

        Args:
            register_tokens: (B, n_registers, hidden_dim)

        Returns:
            (B, output_channels, output_size, output_size)
        """
        return self._decode(register_tokens)

    def forward_prefix(self, register_tokens: torch.Tensor, k: int) -> torch.Tensor:
        """
        OAT-style prefix decoding: reconstruct using only the first K tokens.

        Cross-attention naturally handles variable-length memory, so we simply
        pass the first K tokens as memory — no zero-padding needed. This is
        cleaner than the ConvTranspose decoder's zero-padding approach.

        Args:
            register_tokens: (B, n_registers, hidden_dim) all tokens.
            k: Number of prefix tokens to use for decoding.

        Returns:
            (B, output_channels, output_size, output_size)
        """
        k = min(k, register_tokens.shape[1])
        return self._decode(register_tokens[:, :k, :])


class _UpBlock(nn.Module):
    """Upsample 2x via ConvTranspose2d + BatchNorm + GELU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)
