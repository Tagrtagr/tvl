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
import numpy as np
from typing import Dict, Optional, Tuple, List
from types import SimpleNamespace

from .register_tokens import RegisterTokenModule

# Re-use the modality type definitions from TVL
ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    TACTILE="tactile",
)


class CrossModalAlignmentModel(nn.Module):
    """
    Stage 2 model that takes frozen encoder features and produces
    aligned cross-modal representations via register tokens.

    Args:
        modality_configs: Dict mapping modality name to encoder output config.
            Each entry: {"input_dim": int, "feature_type": "pooled" | "sequence"}
        hidden_dim: Shared hidden dimension for all register token modules.
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
        hidden_dim: int = 512,
        n_registers: int = 32,
        n_shared: int = 8,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        nested_dropout: bool = True,
        nested_dropout_mode: str = "power_of_two",
        init_logit_scale: float = np.log(1 / 0.07),
    ):
        super().__init__()

        if modality_configs is None:
            # Default: vision (OpenCLIP ViT-L-14 = 768d) and tactile (ViT = 768d)
            modality_configs = {
                ModalityType.VISION: {"input_dim": 768, "feature_type": "pooled"},
                ModalityType.TACTILE: {"input_dim": 768, "feature_type": "pooled"},
            }

        self.modality_names = list(modality_configs.keys())
        self.hidden_dim = hidden_dim
        self.n_registers = n_registers
        self.n_shared = n_shared

        # Create a RegisterTokenModule for each modality
        self.register_modules = nn.ModuleDict()
        for mod_name, config in modality_configs.items():
            self.register_modules[mod_name] = RegisterTokenModule(
                input_dim=config["input_dim"],
                hidden_dim=hidden_dim,
                n_registers=n_registers,
                n_shared=n_shared,
                n_layers=n_layers,
                n_heads=n_heads,
                dropout=dropout,
                nested_dropout=nested_dropout,
                nested_dropout_mode=nested_dropout_mode,
            )

        self.feature_types = {k: v["feature_type"] for k, v in modality_configs.items()}

        # Learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

        # Projection heads: map shared tokens to a normalized embedding for contrastive loss
        # We pool the shared tokens (mean pool) then project
        self.shared_projectors = nn.ModuleDict()
        for mod_name in modality_configs:
            self.shared_projectors[mod_name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # Preservation projectors: map pooled private tokens back to encoder dim
        # for the preservation loss (hidden_dim -> input_dim)
        self.preservation_projectors = nn.ModuleDict()
        for mod_name, config in modality_configs.items():
            input_dim = config["input_dim"]
            if input_dim != hidden_dim:
                self.preservation_projectors[mod_name] = nn.Linear(hidden_dim, input_dim)
            else:
                self.preservation_projectors[mod_name] = nn.Identity()

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

            shared_tokens, private_tokens, k_keep = self.register_modules[mod_name](features)

            # Pool shared tokens -> single vector, then project and normalize
            shared_pooled = shared_tokens.mean(dim=1)  # (B, hidden_dim)
            shared_proj = self.shared_projectors[mod_name](shared_pooled)
            shared_proj = F.normalize(shared_proj, dim=-1)

            output[f"{mod_name}_shared"] = shared_proj
            output[f"{mod_name}_private"] = private_tokens
            output[f"{mod_name}_shared_tokens"] = shared_tokens
            output[f"{mod_name}_k_keep"] = k_keep

        output["logit_scale"] = self.logit_scale.exp()
        return output


# Need F for normalize
import torch.nn.functional as F


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
        out = self.frozen_encoder(input_dict)
        # Remove non-feature keys
        out.pop("logit_scale", None)
        out.pop("logit_bias", None)
        return out

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Full forward: frozen encode -> alignment."""
        frozen_features = self.encode(input_dict)
        return self.alignment_model(frozen_features)
