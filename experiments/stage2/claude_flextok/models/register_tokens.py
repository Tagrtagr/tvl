"""
Register Token Module with Nested Dropout and OAT-style Ordering.

Adapted from FlexTok (Bachmann et al., 2025) for cross-modality alignment,
with insights from Ordered Action Tokenization (OAT) for enforcing
coarse-to-fine information hierarchy in register tokens.

Key ideas:
  - Learnable register tokens capture information at varying granularity
  - Causal attention among registers enforces coarse-to-fine ordering
  - Nested dropout ensures each prefix is a valid representation
  - Token type embeddings signal role (shared vs private) — OAT-inspired
  - Tokens are split into "shared" (for cross-modal alignment) and
    "private" (for modality-specific information preservation)

References:
  - FlexTok: https://arxiv.org/abs/2502.13967
  - OAT: https://ordered-action-tokenization.github.io/
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .flextok_transformer_trunk import FlexTransformer


class Registers1D(nn.Module):
    """
    Register-token *only* component.

    Responsibilities:
      - Own learnable register token parameters (content + positional + optional type)
      - Expand registers to batch size
      - Apply nested (prefix) dropout to the *shared* subset (optionally)

    This is intentionally transformer-free so it can be composed with different
    sequence processors (e.g., transformer, conv, etc.).
    """

    def __init__(
        self,
        hidden_dim: int,
        n_registers: int,
        n_shared: int,
        *,
        nested_dropout: bool = True,
        nested_dropout_mode: str = "power_of_two",
        use_token_type_embed: bool = True,
    ):
        super().__init__()
        assert n_shared <= n_registers, "n_shared must be <= n_registers"
        self.hidden_dim = int(hidden_dim)
        self.n_registers = int(n_registers)
        self.n_shared = int(n_shared)
        self.n_private = int(n_registers - n_shared)
        self.nested_dropout = bool(nested_dropout)
        self.nested_dropout_mode = str(nested_dropout_mode)
        self.use_token_type_embed = bool(use_token_type_embed)

        # Learnable register tokens
        self.register_tokens = nn.Parameter(torch.randn(1, n_registers, hidden_dim) * 0.02)

        # Positional embeddings for register tokens (learnable)
        self.register_pos_embed = nn.Parameter(torch.randn(1, n_registers, hidden_dim) * 0.02)

        # OAT-inspired token type embeddings to signal role (shared vs private).
        if use_token_type_embed:
            self.token_type_embed = nn.Parameter(torch.randn(1, n_registers, hidden_dim) * 0.02)

        # Pre-compute valid K_keep values for nested dropout on SHARED tokens only.
        if nested_dropout_mode == "power_of_two":
            self._k_keep_shared_values = []
            k = 1
            while k <= n_shared:
                self._k_keep_shared_values.append(k)
                k *= 2
            if self._k_keep_shared_values and self._k_keep_shared_values[-1] != n_shared:
                self._k_keep_shared_values.append(n_shared)
        else:
            self._k_keep_shared_values = list(range(1, n_shared + 1))

        self._init_weights()

        # Seed shared vs private distinction after initialization.
        if use_token_type_embed:
            with torch.no_grad():
                self.token_type_embed[:, :n_shared, :] *= 0.5
                self.token_type_embed[:, n_shared:, :] *= -0.5

    def _init_weights(self):
        nn.init.trunc_normal_(self.register_tokens, std=0.02)
        nn.init.trunc_normal_(self.register_pos_embed, std=0.02)
        if self.use_token_type_embed:
            nn.init.trunc_normal_(self.token_type_embed, std=0.02)

    def _sample_k_keep_shared(self) -> int:
        if not self._k_keep_shared_values:
            return self.n_shared
        idx = torch.randint(0, len(self._k_keep_shared_values), (1,)).item()
        return int(self._k_keep_shared_values[idx])

    def expand_registers(self, batch_size: int) -> torch.Tensor:
        """Return (B, n_registers, hidden_dim) registers (no dropout applied)."""
        regs = self.register_tokens.expand(batch_size, -1, -1) + self.register_pos_embed
        if self.use_token_type_embed:
            regs = regs + self.token_type_embed
        return regs

    def apply_shared_nested_dropout(
        self,
        registers: torch.Tensor,
        *,
        apply: bool,
    ) -> Tuple[torch.Tensor, int]:
        """
        Optionally zero-out shared registers beyond k_keep.

        Args:
            registers: (B, n_registers, D)
            apply: Whether to apply nested dropout (already accounts for train/eval).

        Returns:
            registers_dropped: (B, n_registers, D) (private tokens unchanged)
            k_keep: number of shared tokens kept (1..n_shared) or n_shared if not applied
        """
        if not apply:
            return registers, self.n_shared

        k_keep = self._sample_k_keep_shared()
        if k_keep >= self.n_shared:
            return registers, self.n_shared

        regs = registers.clone()
        regs[:, k_keep:self.n_shared, :] = 0.0
        return regs, k_keep


class RegisterTokenTransformer(nn.Module):
    """
    Transformer *only* component used to mix input tokens with register tokens.

    Responsibilities:
      - Construct attention mask implementing causal ordering among registers
      - Apply a stack of transformer layers + final norm
    """

    def __init__(self, hidden_dim: int, n_registers: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.hidden_dim = int(hidden_dim)
        self.n_registers = int(n_registers)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        # FlexTok-style trunk: dict-based read/write keys.
        # We still build the same additive attention mask to enforce causal ordering among registers.
        self.trunk = FlexTransformer(
            input_seq_read_key="seq_in",
            output_seq_write_key="seq_out",
            dim=hidden_dim,
            depth=n_layers,
            n_heads=n_heads,
            mlp_ratio=4.0,
            drop=dropout,
            attn_mask_read_key="attn_mask",
        )

    def _build_attention_mask(self, n_input: int, device: torch.device) -> torch.Tensor:
        """
        Sequence layout: [input_tokens (n_input), register_tokens (n_registers)]

        Attention rules (FlexTok-style):
          - Input tokens attend to all inputs + all registers
          - Register i attends to all input tokens
          - Register i attends to register j only if i >= j (causal among registers)
        """
        total = n_input + self.n_registers
        mask = torch.zeros(total, total, device=device)
        reg_start = n_input
        for i in range(self.n_registers):
            for j in range(i + 1, self.n_registers):
                mask[reg_start + i, reg_start + j] = float("-inf")
        return mask

    def forward(self, x: torch.Tensor, *, n_input: int) -> torch.Tensor:
        """
        Args:
            x: (B, n_input + n_registers, D) concatenated input+register tokens
            n_input: number of input tokens (to build mask / split later)
        Returns:
            x_out: (B, n_input + n_registers, D)
        """
        attn_mask = self._build_attention_mask(n_input, x.device)
        data_dict = {"seq_in": x, "attn_mask": attn_mask}
        data_dict = self.trunk(data_dict)
        return data_dict["seq_out"]


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
        n_registers: int = 32,
        n_shared: int = 8,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        nested_dropout: bool = True,
        nested_dropout_mode: str = "power_of_two",
        use_token_type_embed: bool = True,
    ):
        super().__init__()
        assert n_shared <= n_registers, "n_shared must be <= n_registers"
        hidden_dim = input_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_registers = n_registers
        self.n_shared = n_shared
        self.n_private = n_registers - n_shared
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.nested_dropout = nested_dropout
        self.nested_dropout_mode = nested_dropout_mode
        self.use_token_type_embed = use_token_type_embed

        self.registers = Registers1D(
            hidden_dim=hidden_dim,
            n_registers=n_registers,
            n_shared=n_shared,
            nested_dropout=nested_dropout,
            nested_dropout_mode=nested_dropout_mode,
            use_token_type_embed=use_token_type_embed,
        )
        self.transformer = RegisterTokenTransformer(
            hidden_dim=hidden_dim,
            n_registers=n_registers,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

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
            k_keep: Number of shared tokens kept (n_shared if no dropout applied)
        """
        B, N, _ = encoder_features.shape

        # Projection-free: encoder_features already live in hidden_dim (= input_dim)
        x_input = encoder_features  # (B, N, hidden_dim)

        # Expand register tokens for the batch
        registers = self.registers.expand_registers(B)

        # Concatenate: [input_tokens, register_tokens]
        x = torch.cat([x_input, registers], dim=1)  # (B, N + n_reg, hidden_dim)
        x = self.transformer(x, n_input=N)

        # Extract only register tokens (discard input tokens)
        register_out = x[:, N:, :]  # (B, n_registers, hidden_dim)

        # Apply nested dropout only to shared registers (private always intact).
        use_dropout = (
            apply_nested_dropout
            if apply_nested_dropout is not None
            else (self.nested_dropout and self.training)
        )
        register_out, k_keep = self.registers.apply_shared_nested_dropout(register_out, apply=use_dropout)

        shared_tokens = register_out[:, : self.n_shared, :]
        private_tokens = register_out[:, self.n_shared :, :]

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
