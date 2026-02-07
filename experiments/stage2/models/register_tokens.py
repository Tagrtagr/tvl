"""
Register Token Module with Nested Dropout.

Adapted from FlexTok (Bachmann et al., 2025) for cross-modality alignment.
Key ideas:
  - Learnable register tokens capture information at varying granularity
  - Causal attention among registers enforces coarse-to-fine ordering
  - Nested dropout ensures each prefix is a valid representation
  - Tokens are split into "shared" (for cross-modal alignment) and
    "private" (for modality-specific information preservation)

Reference: https://arxiv.org/abs/2502.13967
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RegisterTokenModule(nn.Module):
    """
    Processes frozen encoder features through a lightweight transformer
    with learnable register tokens. The register tokens learn to capture
    multi-granularity representations of the input modality.

    After encoding, only the register tokens are kept as the bottleneck
    representation. They are split into:
      - shared_tokens[:n_shared]: aligned across modalities via contrastive loss
      - private_tokens[n_shared:]: preserve modality-specific information

    Args:
        input_dim: Dimension of frozen encoder output features.
        hidden_dim: Internal transformer dimension.
        n_registers: Total number of register tokens.
        n_shared: Number of registers allocated for cross-modal alignment.
            The remaining (n_registers - n_shared) are modality-private.
        n_layers: Number of transformer encoder layers.
        n_heads: Number of attention heads.
        dropout: Standard dropout rate.
        nested_dropout: Whether to apply nested (prefix) dropout during training.
        nested_dropout_mode: "power_of_two" or "uniform" sampling for K_keep.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        n_registers: int = 32,
        n_shared: int = 8,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        nested_dropout: bool = True,
        nested_dropout_mode: str = "power_of_two",
    ):
        super().__init__()
        assert n_shared <= n_registers, "n_shared must be <= n_registers"
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_registers = n_registers
        self.n_shared = n_shared
        self.n_private = n_registers - n_shared
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.nested_dropout = nested_dropout
        self.nested_dropout_mode = nested_dropout_mode

        # Project frozen encoder features to hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable register tokens
        self.register_tokens = nn.Parameter(
            torch.randn(1, n_registers, hidden_dim) * 0.02
        )

        # Positional embeddings for register tokens (learnable)
        self.register_pos_embed = nn.Parameter(
            torch.randn(1, n_registers, hidden_dim) * 0.02
        )

        # Transformer layers with custom attention masking
        self.layers = nn.ModuleList([
            RegisterTransformerLayer(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        # Pre-compute valid K_keep values for nested dropout
        if nested_dropout_mode == "power_of_two":
            # {1, 2, 4, 8, ..., n_registers} intersected with powers of 2
            self._k_keep_values = []
            k = 1
            while k <= n_registers:
                self._k_keep_values.append(k)
                k *= 2
            if self._k_keep_values[-1] != n_registers:
                self._k_keep_values.append(n_registers)
        else:
            self._k_keep_values = list(range(1, n_registers + 1))

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.register_tokens, std=0.02)
        nn.init.trunc_normal_(self.register_pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def _build_attention_mask(self, n_input: int, device: torch.device) -> torch.Tensor:
        """
        Build the attention mask for joint self-attention.

        Sequence layout: [input_tokens (n_input), register_tokens (n_registers)]

        Attention rules (following FlexTok):
          - Input tokens attend to all input tokens and all registers (full)
          - Register i attends to all input tokens (full)
          - Register i attends to register j only if i >= j (causal among registers)

        Returns:
            Float mask of shape (n_input + n_registers, n_input + n_registers).
            0.0 = allowed, -inf = blocked (additive mask for scaled dot-product).
        """
        total = n_input + self.n_registers
        # Start with all allowed
        mask = torch.zeros(total, total, device=device)

        # Block: register i cannot attend to register j if i < j (causal)
        reg_start = n_input
        for i in range(self.n_registers):
            for j in range(i + 1, self.n_registers):
                mask[reg_start + i, reg_start + j] = float("-inf")

        return mask

    def _sample_k_keep(self) -> int:
        """Sample number of register tokens to keep during nested dropout."""
        idx = torch.randint(0, len(self._k_keep_values), (1,)).item()
        return self._k_keep_values[idx]

    def forward(
        self,
        encoder_features: torch.Tensor,
        apply_nested_dropout: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Args:
            encoder_features: (B, N, input_dim) frozen encoder output.
                For CLS-pooled features of shape (B, input_dim), unsqueeze dim=1 first.
            apply_nested_dropout: Override self.nested_dropout if provided.

        Returns:
            shared_tokens: (B, n_shared, hidden_dim) - for cross-modal contrastive alignment
            private_tokens: (B, n_private, hidden_dim) - modality-specific preservation
            k_keep: Number of tokens kept (n_registers if no dropout applied)
        """
        B, N, _ = encoder_features.shape
        device = encoder_features.device

        # Project input features
        x_input = self.input_proj(encoder_features)  # (B, N, hidden_dim)

        # Expand register tokens for the batch
        registers = self.register_tokens.expand(B, -1, -1) + self.register_pos_embed

        # Concatenate: [input_tokens, register_tokens]
        x = torch.cat([x_input, registers], dim=1)  # (B, N + n_reg, hidden_dim)

        # Build attention mask
        attn_mask = self._build_attention_mask(N, device)

        # Process through transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        x = self.norm(x)

        # Extract only register tokens (discard input tokens)
        register_out = x[:, N:, :]  # (B, n_registers, hidden_dim)

        # Apply nested dropout
        use_dropout = apply_nested_dropout if apply_nested_dropout is not None else (self.nested_dropout and self.training)
        if use_dropout:
            k_keep = self._sample_k_keep()
            # Zero out tokens beyond k_keep
            if k_keep < self.n_registers:
                register_out = register_out.clone()
                register_out[:, k_keep:, :] = 0.0
        else:
            k_keep = self.n_registers

        # Split into shared and private
        shared_tokens = register_out[:, :self.n_shared, :]
        private_tokens = register_out[:, self.n_shared:, :]

        return shared_tokens, private_tokens, k_keep


class RegisterTransformerLayer(nn.Module):
    """
    Standard pre-norm transformer encoder layer with support for
    additive attention masks (to implement causal masking among registers).
    """

    def __init__(self, hidden_dim: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm self-attention with optional mask
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_out

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x
