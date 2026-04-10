"""
Cross-Modality Alignment Model (Stage 2).

Wraps frozen Stage-1 encoders with RegisterTokenModules for each modality.
The shared register tokens from different modalities are aligned via
contrastive learning, while private tokens preserve modality-specific info.

Architecture:
    [Frozen Vision Encoder] -> [RegisterTokenModule_vision]
    [Frozen Tactile Encoder] -> [RegisterTokenModule_tactile]
    [Frozen Text Encoder] -> [RegisterTokenModule_text]

    shared_vision, shared_tactile -> Contrastive Alignment
    private_vision -> Preserves vision-only information
    private_tactile -> Preserves tactile-only information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from types import SimpleNamespace

from .register_tokens import RegisterTokenModule

# Re-use the modality type definitions from TVL
ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    TACTILE="tactile",
)


def _exists(v: Any) -> bool:
    return v is not None


class FSQEncoding(nn.Module):
    """
    Minimal FSQ encoding (Finite Scalar Quantization), adapted from FlexTok.

    This is a tensor-based implementation so Stage2 can use FSQ token IDs
    without depending on importing `ml-flextok` as a package.
    """

    def __init__(self, levels: List[int], drop_quant_p: float = 0.0):
        super().__init__()
        if not isinstance(levels, (list, tuple)) or len(levels) == 0:
            raise ValueError("FSQEncoding requires a non-empty levels list.")
        self.levels = [int(x) for x in levels]
        self.drop_quant_p = float(drop_quant_p)

        _levels = torch.tensor(self.levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)
        _basis = torch.cumprod(torch.tensor([1] + self.levels[:-1], dtype=torch.int32), dim=0)
        self.register_buffer("_basis", _basis, persistent=False)

        self.dim = len(self.levels)
        self.codebook_size = int(_levels.prod().item())

    @staticmethod
    def _round_ste(z: torch.Tensor) -> torch.Tensor:
        zhat = z.round()
        return z + (zhat - z).detach()

    def _round_ste_quant_dropout(self, z: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.drop_quant_p <= 0.0:
            return self._round_ste(z)
        zhat = z.round()
        bsz = z.shape[0]
        mask = torch.bernoulli(torch.full((bsz,), self.drop_quant_p, device=z.device, dtype=z.dtype))
        mask = mask.view(bsz, *([1] * (z.ndim - 1)))
        return z + ((1 - mask) * (zhat - z)).detach()

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = torch.atanh(offset / half_l)
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        bounded = self.bound(z)
        quantized = self._round_ste_quant_dropout(bounded)
        half_width = self._levels // 2  # renormalize to [-1, 1]
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def codes_to_indices(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        if zhat_normalized.shape[-1] != self.dim:
            raise ValueError(f"FSQ expects last dim={self.dim}, got {zhat_normalized.shape[-1]}")
        zhat = self._scale_and_shift(zhat_normalized)
        return (zhat * self._basis).sum(dim=-1).to(torch.int32).long()

    def forward_z(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, L, dim) float tensor (channel last).
        Returns:
            quant: (B, L, dim) quantized (in [-1, 1] space)
            token_ids: (B, L) discrete token ids
        """
        if z.shape[-1] != self.dim:
            raise ValueError(f"FSQEncoding expected last dim {self.dim}, got {z.shape[-1]}")
        quant = self.quantize(z.float())
        token_ids = self.codes_to_indices(quant)
        return quant.to(dtype=z.dtype), token_ids


class CrossModalAlignmentModel(nn.Module):
    """
    Stage 2 model that takes frozen encoder features and produces
    aligned cross-modal representations via register tokens.

    Args:
        modality_configs: Dict mapping modality name to encoder output config.
            Each entry: {"input_dim": int, "feature_type": "pooled" | "sequence"}
        n_registers: Number of register tokens per modality.
        n_shared: Number of shared (aligned) registers per modality.
        n_layers: Transformer depth in each register module.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        nested_dropout: Whether to use nested dropout.
        nested_dropout_mode: "power_of_two" or "uniform".
        init_logit_scale: Initial value for learnable logit scale (temperature).
    """

    def __init__(
        self,
        modality_configs: Optional[Dict] = None,
        n_registers: int = 32,
        n_shared: int = 8,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        nested_dropout: bool = True,
        nested_dropout_mode: str = "power_of_two",
        init_logit_scale: float = np.log(1 / 0.07),
        use_token_type_embed: bool = True,
        fsq_levels: Optional[List[int]] = None,
        fsq_drop_quant_p: float = 0.0,
    ):
        super().__init__()

        if modality_configs is None:
            # Default: vision/tactile token features projected to 768d
            modality_configs = {
                ModalityType.VISION: {"input_dim": 768, "feature_type": "sequence"},
                ModalityType.TACTILE: {"input_dim": 768, "feature_type": "sequence"},
            }

        self.modality_names = list(modality_configs.keys())
        input_dims = [cfg["input_dim"] for cfg in modality_configs.values()]
        if len(set(input_dims)) != 1:
            raise ValueError(
                f"All modality input_dim values must match, got: {input_dims}"
            )
        self.hidden_dim = input_dims[0]
        self.n_registers = n_registers
        self.n_shared = n_shared

        # FlexTok-like staged pipeline modules (per modality).
        # Using an explicit module dictionary makes it easier to mirror the ml-flextok structure.
        self.module_dict = nn.ModuleDict()
        for mod_name, config in modality_configs.items():
            self.module_dict[f"{mod_name}_register_module"] = RegisterTokenModule(
                input_dim=config["input_dim"],
                n_registers=n_registers,
                n_shared=n_shared,
                n_layers=n_layers,
                n_heads=n_heads,
                dropout=dropout,
                nested_dropout=nested_dropout,
                nested_dropout_mode=nested_dropout_mode,
                use_token_type_embed=use_token_type_embed,
            )

        self.feature_types = {k: v["feature_type"] for k, v in modality_configs.items()}

        # Learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

        # Projection heads: map shared tokens to a normalized embedding for contrastive loss
        # We pool the shared tokens (mean pool) then project
        self.shared_projectors = nn.ModuleDict()
        for mod_name in modality_configs:
            self.shared_projectors[mod_name] = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        # Preservation projectors: map pooled private tokens back to encoder dim
        # for the preservation loss (hidden_dim -> input_dim)
        self.preservation_projectors = nn.ModuleDict()
        for mod_name, config in modality_configs.items():
            input_dim = config["input_dim"]
            if input_dim != self.hidden_dim:
                self.preservation_projectors[mod_name] = nn.Linear(self.hidden_dim, input_dim)
            else:
                self.preservation_projectors[mod_name] = nn.Identity()

        # FSQ encoding path (FlexTok-style regularizer stage).
        # Always enabled (per user request). Default levels are the commonly used 6D FSQ setup.
        if fsq_levels is None:
            fsq_levels = [8, 8, 8, 5, 5, 5]
        self.fsq_levels = list(fsq_levels)
        self.fsq = FSQEncoding(levels=self.fsq_levels, drop_quant_p=fsq_drop_quant_p)
        fsq_dim = self.fsq.dim
        self.fsq_in_proj = nn.ModuleDict({m: nn.Linear(self.hidden_dim, fsq_dim) for m in modality_configs})
        self.fsq_out_proj = nn.ModuleDict({m: nn.Linear(fsq_dim, self.hidden_dim) for m in modality_configs})

    @staticmethod
    def _prefix_schedule(n_tokens: int) -> List[int]:
        """Power-of-two prefix schedule, excluding full length."""
        lengths: List[int] = []
        k = 1
        while k < n_tokens:
            lengths.append(k)
            k *= 2
        return lengths

    def _prepare_features(self, features: torch.Tensor, feature_type: str) -> torch.Tensor:
        """Ensure features are (B, N, D) for the register module."""
        if feature_type == "pooled" and features.ndim == 2:
            return features.unsqueeze(1)  # (B, D) -> (B, 1, D)
        return features

    def forward(
        self,
        feature_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            feature_dict: Maps modality name -> frozen encoder features.
                Vision: (B, 768) or (B, N, 768) depending on pooling.
                Tactile: (B, 768) or (B, N, 768).

        Returns:
            Dictionary with:
                - "{mod}_shared": (B, hidden_dim) L2-normalized shared embedding
                - "{mod}_private": (B, n_private, hidden_dim) private tokens
                - "{mod}_shared_tokens": (B, n_shared, hidden_dim) raw shared tokens
                - "{mod}_k_keep": int, tokens kept after nested dropout
                - "logit_scale": scalar temperature
        """
        output = {}

        for mod_name in self.modality_names:
            if mod_name not in feature_dict:
                continue

            features = feature_dict[mod_name]
            features = self._prepare_features(features, self.feature_types[mod_name])

            shared_tokens, private_tokens, k_keep = self.module_dict[f"{mod_name}_register_module"](features)

            # Pool only non-zeroed shared tokens to avoid dilution from nested dropout.
            # k_keep indicates how many shared tokens are active (rest are zeroed).
            if k_keep < shared_tokens.shape[1]:
                shared_pooled = shared_tokens[:, :k_keep, :].mean(dim=1)
            else:
                shared_pooled = shared_tokens.mean(dim=1)  # (B, hidden_dim)
            shared_proj = self.shared_projectors[mod_name](shared_pooled)
            shared_proj = F.normalize(shared_proj, dim=-1)

            output[f"{mod_name}_shared"] = shared_proj
            output[f"{mod_name}_private"] = private_tokens
            output[f"{mod_name}_shared_tokens"] = shared_tokens
            output[f"{mod_name}_k_keep"] = k_keep
            # All register tokens (for reconstruction decoder)
            all_tokens = torch.cat([shared_tokens, private_tokens], dim=1)
            output[f"{mod_name}_all_tokens"] = all_tokens

            # Default prefix schedule/tokens for reconstruction losses.
            # Power-of-two schedule excludes full length to avoid duplicate full decode.
            prefix_lengths = self._prefix_schedule(all_tokens.shape[1])
            output[f"{mod_name}_prefix_lengths"] = prefix_lengths
            if prefix_lengths:
                idx = torch.randint(0, len(prefix_lengths), (1,), device=all_tokens.device).item()
                k_prefix = int(prefix_lengths[idx])
            else:
                k_prefix = int(all_tokens.shape[1])
            output[f"{mod_name}_prefix_k"] = k_prefix
            output[f"{mod_name}_prefix_tokens"] = all_tokens[:, :k_prefix, :]

            # FSQ encoding (token ids + quantized embeddings)
            z = self.fsq_in_proj[mod_name](all_tokens)  # (B, L, fsq_dim)
            quant, token_ids = self.fsq.forward_z(z)
            output[f"{mod_name}_fsq_tokens"] = token_ids  # (B, L)
            output[f"{mod_name}_fsq_quants"] = self.fsq_out_proj[mod_name](quant)  # (B, L, hidden_dim)

        output["logit_scale"] = self.logit_scale.exp()
        return output



class Stage2Wrapper(nn.Module):
    """
    Full Stage 2 pipeline: frozen encoders + cross-modal alignment.

    This wraps the frozen TVL Stage-1 model and the trainable Stage-2
    alignment module into a single forward pass.
    """

    def __init__(
        self,
        frozen_encoder,
        alignment_model: CrossModalAlignmentModel,
    ):
        super().__init__()
        self.frozen_encoder = frozen_encoder
        self.alignment_model = alignment_model

        # Freeze the encoder
        for param in self.frozen_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run frozen Stage-1 encoders."""
        self.frozen_encoder.eval()
        out = self.frozen_encoder(
            input_dict,
            return_token_sequences=True,
            drop_cls_token=True,
        )
        # Remove non-feature keys
        out.pop("logit_scale", None)
        out.pop("logit_bias", None)
        return out

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Full forward: frozen encode -> alignment."""
        frozen_features = self.encode(input_dict)
        return self.alignment_model(frozen_features)
