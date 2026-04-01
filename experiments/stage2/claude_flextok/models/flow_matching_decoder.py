"""
Flow-Matching Reconstruction Decoder (Rectified Flow) for Stage 2.

This implements a conditional rectified-flow model that reconstructs pixels
from ALL register tokens (shared + private), matching the idea used in FlexTok:
train a generative decoder to ensure the bottleneck tokens retain sufficient
information for reconstruction.

Training objective (rectified flow):
  - Sample x0 ~ N(0, I) (noise), x1 = target image
  - Sample t in [0, 1]
  - Interpolate x_t = (1 - t) * x0 + t * x1
  - Target velocity v = x1 - x0
  - Predict v_hat = f(x_t, t, tokens)
  - Minimize ||v_hat - v||^2

At inference, integrate ODE with Euler steps:
  x_{t+dt} = x_t + dt * f(x_t, t, tokens)

Implementation notes:
  - Operates in patch space (ViT-like): 224x224 -> 14x14 patches of size 16.
  - Uses lightweight transformer blocks with self-attn on patches and cross-attn
    from patches to register tokens.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class FlowMatchingReconstructionDecoder(nn.Module):
    def __init__(
        self,
        n_registers: int = 32,
        hidden_dim: int = 512,
        patch_size: int = 16,
        img_size: int = 224,
        n_decoder_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        base_channels: int = 64,  # kept for API parity; not used directly here
    ):
        super().__init__()
        self.n_registers = n_registers
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.img_size = img_size

        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid = img_size // patch_size
        self.n_patches = self.grid * self.grid

        # Patchify / unpatchify
        self.to_patches = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.to_velocity_patches = nn.Linear(hidden_dim, 3 * patch_size * patch_size, bias=True)

        # Learned positional embeddings for patches
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, hidden_dim) * 0.02)

        # Time embedding (sinusoidal -> MLP)
        time_dim = hidden_dim * 2
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [
                PatchTokenBlock(hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout)
                for _ in range(n_decoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.to_velocity_patches.weight)
        nn.init.zeros_(self.to_velocity_patches.bias)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, H, W) -> (B, n_patches, hidden_dim)
        h = self.to_patches(x)  # (B, D, G, G)
        return h.flatten(2).transpose(1, 2)  # (B, G*G, D)

    def _unpatchify_velocity(self, v_patch: torch.Tensor) -> torch.Tensor:
        # (B, n_patches, 3*P*P) -> (B, 3, H, W)
        B = v_patch.shape[0]
        P = self.patch_size
        G = self.grid
        v = v_patch.view(B, G, G, 3, P, P)          # (B, G, G, 3, P, P)
        v = v.permute(0, 3, 1, 4, 2, 5).contiguous()  # (B, 3, G, P, G, P)
        return v.view(B, 3, G * P, G * P)

    def predict_velocity(self, x_t: torch.Tensor, t: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        Predict pixel-space velocity v_hat at (x_t, t) conditioned on register tokens.

        Args:
            x_t: (B, 3, 224, 224) interpolated image
            t: (B,) timesteps in [0, 1]
            tokens: (B, n_registers, hidden_dim) conditioning memory

        Returns:
            v_hat: (B, 3, 224, 224)
        """
        if t.ndim != 1:
            t = t.view(-1)

        x = self._patchify(x_t)  # (B, N, D)
        x = x + self.pos_embed

        t_emb = self.time_embed(t)  # (B, D)
        for blk in self.blocks:
            x = blk(x, tokens, t_emb)

        x = self.norm(x)
        v_patch = self.to_velocity_patches(x)  # (B, N, 3*P*P)
        return self._unpatchify_velocity(v_patch)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """DDP-friendly wrapper used by flow reconstruction loss."""
        return self.predict_velocity(x_t, t, tokens)

    @torch.no_grad()
    def sample(self, tokens: torch.Tensor, n_steps: int = 20, x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate a reconstruction by integrating the rectified-flow ODE.

        Args:
            tokens: (B, n_registers, hidden_dim)
            n_steps: Euler steps
            x0: Optional initial image (B, 3, 224, 224). If None, uses N(0,1).
        """
        B = tokens.shape[0]
        device = tokens.device
        if x0 is None:
            x = torch.randn(B, 3, self.img_size, self.img_size, device=device)
        else:
            x = x0.clone()

        dt = 1.0 / float(n_steps)
        for i in range(n_steps):
            t = torch.full((B,), (i + 0.5) * dt, device=device)
            v = self.predict_velocity(x, t, tokens)
            x = x + dt * v
        return x


class PatchTokenBlock(nn.Module):
    """Self-attn over patches + cross-attn to tokens, with AdaLN(time)."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm_kv = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.ada = nn.Linear(hidden_dim, hidden_dim * 6)  # scales/shifts/gates for 3 sublayers

    def forward(self, x: torch.Tensor, tokens: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # Compute AdaLN params
        # (B, 6D) -> scale/shift/gate for self-attn, cross-attn, mlp
        ssa_s, ssa_b, ssa_g, ca_s, ca_b, mlp_g = self.ada(t_emb).chunk(6, dim=-1)

        # Self-attn (patches)
        h = self.norm_q(x) * (1 + ssa_s.unsqueeze(1)) + ssa_b.unsqueeze(1)
        attn, _ = self.self_attn(h, h, h)
        x = x + torch.tanh(ssa_g).unsqueeze(1) * attn

        # Cross-attn (patches query, tokens key/value)
        q = self.norm_q(x) * (1 + ca_s.unsqueeze(1)) + ca_b.unsqueeze(1)
        kv = self.norm_kv(tokens)
        attn2, _ = self.cross_attn(q, kv, kv)
        x = x + attn2

        # MLP
        x = x + torch.tanh(mlp_g).unsqueeze(1) * self.mlp(self.norm_q(x))
        return x


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

