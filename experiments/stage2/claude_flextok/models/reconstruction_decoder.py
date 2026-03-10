"""
Reconstruction Decoder for FlexTok-style register tokens.

Takes ALL register tokens (shared + private) and decodes back to pixel space.
This is the core FlexTok contribution: register tokens should be sufficient
to reconstruct the original image/tactile input.

Supports OAT-style prefix decoding: reconstruct using only the first K tokens
(zero-padding the rest) to evaluate what information each token captures.
See: Ordered Action Tokenization (https://ordered-action-tokenization.github.io/)

Architecture:
    register_tokens (B, n_reg, hidden_dim)
    -> small transformer (self-attention among tokens)
    -> flatten + linear project -> reshape to (B, C, 7, 7)
    -> progressive ConvTranspose2d upsample -> (B, 3, 224, 224)

Reference: FlexTok (Bachmann et al., 2025) - arXiv 2502.13967
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ReconstructionDecoder(nn.Module):
    """
    Decodes register tokens back to pixel space.

    Args:
        n_registers: Number of register tokens (shared + private).
        hidden_dim: Dimension of each register token.
        base_channels: Channels after initial spatial projection.
        output_channels: Output image channels (3 for RGB).
        output_size: Target spatial resolution (224).
        n_decoder_layers: Number of transformer self-attention layers
            applied to register tokens before decoding.
        n_heads: Attention heads in decoder transformer.
    """

    def __init__(
        self,
        n_registers: int = 32,
        hidden_dim: int = 512,
        base_channels: int = 256,
        output_channels: int = 3,
        output_size: int = 224,
        n_decoder_layers: int = 2,
        n_heads: int = 8,
    ):
        super().__init__()

        self.n_registers = n_registers
        self.hidden_dim = hidden_dim
        self.base_channels = base_channels
        self.output_size = output_size

        # Initial spatial grid size (7x7 = 49 spatial positions)
        self.h0, self.w0 = 7, 7

        # Optional transformer layers to let tokens interact before decoding
        if n_decoder_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.token_processor = nn.TransformerEncoder(
                layer, num_layers=n_decoder_layers
            )
        else:
            self.token_processor = nn.Identity()

        self.token_norm = nn.LayerNorm(hidden_dim)

        # Bottleneck: project each token to smaller dim before flattening
        # This avoids a massive (n_reg*hidden_dim x base_channels*49) linear
        bottleneck_dim = hidden_dim // 4  # 512 -> 128
        self.token_proj = nn.Linear(hidden_dim, bottleneck_dim)

        # Project bottlenecked tokens to spatial feature map
        self.to_spatial = nn.Linear(
            n_registers * bottleneck_dim, base_channels * self.h0 * self.w0
        )

        # Progressive upsampling: 7 -> 14 -> 28 -> 56 -> 112 -> 224
        # Channel widths scale with base_channels
        c = base_channels
        c2, c4, c8 = max(c // 2, 8), max(c // 4, 8), max(c // 8, 8)
        self.upsample = nn.Sequential(
            _UpBlock(c, c),      # 7x7 -> 14x14
            _UpBlock(c, c2),     # 14x14 -> 28x28
            _UpBlock(c2, c4),    # 28x28 -> 56x56
            _UpBlock(c4, c8),    # 56x56 -> 112x112
            _UpBlock(c8, c8),    # 112x112 -> 224x224
        )
        self._final_channels = c8

        # Final projection to output channels (no activation — output is
        # in normalized image space which can be negative)
        self.to_pixels = nn.Conv2d(c8, output_channels, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.to_spatial.weight)
        nn.init.zeros_(self.to_spatial.bias)
        nn.init.xavier_uniform_(self.to_pixels.weight)
        nn.init.zeros_(self.to_pixels.bias)

    def forward(self, register_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            register_tokens: (B, n_registers, hidden_dim) all register tokens
                (concatenation of shared + private tokens).

        Returns:
            reconstruction: (B, output_channels, output_size, output_size)
        """
        B = register_tokens.shape[0]

        # Process tokens with self-attention
        x = self.token_processor(register_tokens)
        x = self.token_norm(x)

        # Bottleneck projection per token, then flatten
        x = self.token_proj(x)  # (B, n_reg, bottleneck_dim)
        x = x.reshape(B, -1)  # (B, n_reg * bottleneck_dim)
        x = self.to_spatial(x)  # (B, base_channels * h0 * w0)
        x = x.reshape(B, self.base_channels, self.h0, self.w0)

        # Progressive upsample
        x = self.upsample(x)  # (B, 16, 224, 224)

        # To pixels
        x = self.to_pixels(x)  # (B, 3, 224, 224)

        return x

    def forward_prefix(self, register_tokens: torch.Tensor, k: int) -> torch.Tensor:
        """
        OAT-style prefix decoding: reconstruct using only the first K tokens.

        Following OAT's "anytime reconstruction" principle, earlier tokens should
        capture global structure while later tokens refine details. By decoding
        with only a prefix of tokens, we can evaluate this coarse-to-fine hierarchy.

        Only the first K tokens are processed through self-attention; suffix
        positions are excluded via a padding mask and then zeroed before the
        spatial projection to ensure they contribute nothing to the output.

        Args:
            register_tokens: (B, n_registers, hidden_dim) all register tokens.
            k: Number of prefix tokens to use (rest are masked and zeroed).

        Returns:
            reconstruction: (B, output_channels, output_size, output_size)
        """
        B = register_tokens.shape[0]
        k = max(0, min(k, self.n_registers))

        # Build padding mask: True = ignore. Shape (B, n_registers).
        mask = torch.ones(B, self.n_registers, dtype=torch.bool,
                          device=register_tokens.device)
        if k > 0:
            mask[:, :k] = False  # keep first k tokens

        # Process with attention mask so suffix tokens don't participate
        x = self.token_processor(register_tokens, src_key_padding_mask=mask)
        x = self.token_norm(x)
        x = self.token_proj(x)

        # Zero out suffix positions after projection to remove any residual
        # bias-term activations from masked positions
        if k < self.n_registers:
            x[:, k:, :] = 0.0

        x = x.reshape(B, -1)
        x = self.to_spatial(x)
        x = x.reshape(B, self.base_channels, self.h0, self.w0)
        x = self.upsample(x)
        x = self.to_pixels(x)

        return x


class _UpBlock(nn.Module):
    """Upsample 2x with ConvTranspose2d + BatchNorm + GELU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)
