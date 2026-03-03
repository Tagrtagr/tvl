"""
Reconstruction Loss for FlexTok-style register token decoding.

Measures how well the decoder reconstructs the original image/tactile
from the register token bottleneck. Supports:
  - MSE (L2) loss in pixel space
  - L1 loss in pixel space
  - Optional perceptual loss via frozen encoder features (lightweight)
  - OAT-style prefix reconstruction loss (train with random prefix lengths)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import random


class ReconstructionLoss(nn.Module):
    """
    Pixel-space reconstruction loss.

    Args:
        loss_type: "mse", "l1", or "smooth_l1".
        perceptual_weight: Weight for optional feature-space loss.
            If 0, only pixel loss is used (faster).
    """

    def __init__(
        self,
        loss_type: str = "mse",
        perceptual_weight: float = 0.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.perceptual_weight = perceptual_weight

        if loss_type == "mse":
            self.pixel_loss_fn = nn.MSELoss()
        elif loss_type == "l1":
            self.pixel_loss_fn = nn.L1Loss()
        elif loss_type == "smooth_l1":
            self.pixel_loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def forward(
        self,
        reconstructions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        frozen_features_recon: Optional[Dict[str, torch.Tensor]] = None,
        frozen_features_target: Optional[Dict[str, torch.Tensor]] = None,
        output_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            reconstructions: Maps modality name -> (B, 3, 224, 224) decoded images.
            targets: Maps modality name -> (B, 3, 224, 224) original images.
            frozen_features_recon: Optional frozen encoder features of reconstructions.
            frozen_features_target: Optional frozen encoder features of targets.
            output_dict: Return detailed dict or scalar.

        Returns:
            Dict with per-modality and total reconstruction losses.
        """
        losses = {}
        total_pixel = 0.0
        total_perceptual = 0.0
        n_modalities = 0

        for mod_name in reconstructions:
            if mod_name not in targets:
                continue

            recon = reconstructions[mod_name]
            target = targets[mod_name]

            # Pixel loss
            pixel_loss = self.pixel_loss_fn(recon, target)
            losses[f"recon_pixel_{mod_name}"] = pixel_loss
            total_pixel += pixel_loss

            # Optional perceptual loss (feature-space MSE)
            if (self.perceptual_weight > 0
                    and frozen_features_recon is not None
                    and frozen_features_target is not None
                    and mod_name in frozen_features_recon
                    and mod_name in frozen_features_target):
                feat_recon = frozen_features_recon[mod_name]
                feat_target = frozen_features_target[mod_name].detach()
                perceptual_loss = F.mse_loss(feat_recon, feat_target)
                losses[f"recon_perceptual_{mod_name}"] = perceptual_loss
                total_perceptual += perceptual_loss

            n_modalities += 1

        if n_modalities > 0:
            losses["recon_pixel_avg"] = total_pixel / n_modalities
            if self.perceptual_weight > 0:
                losses["recon_perceptual_avg"] = total_perceptual / n_modalities

        # Total reconstruction loss
        total = losses.get("recon_pixel_avg", torch.tensor(0.0))
        if self.perceptual_weight > 0:
            total = total + self.perceptual_weight * losses.get(
                "recon_perceptual_avg", torch.tensor(0.0)
            )
        losses["recon_total"] = total

        if output_dict:
            return losses
        return total


class PrefixReconstructionLoss(nn.Module):
    """
    OAT-style prefix reconstruction loss.

    Instead of only reconstructing from all tokens, randomly sample a prefix
    length K and reconstruct using only the first K tokens. This enforces
    the coarse-to-fine ordering: earlier tokens must capture enough global
    information to produce a reasonable reconstruction on their own.

    The loss is a weighted average of full reconstruction and prefix
    reconstruction, encouraging the model to distribute information
    hierarchically across the token sequence.

    Args:
        loss_type: "mse", "l1", or "smooth_l1".
        prefix_weight: Weight for prefix reconstruction loss relative to full.
        prefix_schedule: List of valid prefix lengths to sample from.
            If None, uses power-of-two schedule up to n_registers.
    """

    def __init__(
        self,
        loss_type: str = "mse",
        prefix_weight: float = 0.5,
        prefix_schedule: Optional[List[int]] = None,
    ):
        super().__init__()
        self.prefix_weight = prefix_weight
        self.prefix_schedule = prefix_schedule

        if loss_type == "mse":
            self.pixel_loss_fn = nn.MSELoss()
        elif loss_type == "l1":
            self.pixel_loss_fn = nn.L1Loss()
        elif loss_type == "smooth_l1":
            self.pixel_loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def _get_prefix_schedule(self, n_registers: int) -> List[int]:
        """Get power-of-two prefix lengths up to n_registers."""
        if self.prefix_schedule is not None:
            return self.prefix_schedule
        lengths = []
        k = 1
        while k <= n_registers:
            lengths.append(k)
            k *= 2
        if lengths[-1] != n_registers:
            lengths.append(n_registers)
        return lengths

    def forward(
        self,
        all_tokens: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        decoders: Dict[str, nn.Module],
        output_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            all_tokens: Maps modality -> (B, n_registers, hidden_dim) all register tokens.
            targets: Maps modality -> (B, 3, 224, 224) original images.
            decoders: Maps modality -> ReconstructionDecoder.

        Returns:
            Dict with full recon loss, prefix recon loss, and combined total.
        """
        losses = {}
        total_full = 0.0
        total_prefix = 0.0
        n_modalities = 0

        for mod_name in all_tokens:
            if mod_name not in targets or mod_name not in decoders:
                continue

            tokens = all_tokens[mod_name]
            target = targets[mod_name]
            decoder = decoders[mod_name]
            n_reg = tokens.shape[1]

            # Full reconstruction
            full_recon = decoder(tokens)
            full_loss = self.pixel_loss_fn(full_recon, target)
            losses[f"recon_full_{mod_name}"] = full_loss
            total_full += full_loss

            # Prefix reconstruction (sample a random prefix length)
            schedule = self._get_prefix_schedule(n_reg)
            # Exclude the full length — we already computed that
            prefix_candidates = [k for k in schedule if k < n_reg]
            if prefix_candidates:
                k = random.choice(prefix_candidates)
                prefix_recon = decoder.forward_prefix(tokens, k=k)
                prefix_loss = self.pixel_loss_fn(prefix_recon, target)
                losses[f"recon_prefix_k{k}_{mod_name}"] = prefix_loss
                total_prefix += prefix_loss

            n_modalities += 1

        if n_modalities > 0:
            losses["recon_full_avg"] = total_full / n_modalities
            losses["recon_prefix_avg"] = total_prefix / n_modalities

        # Combined: full + prefix_weight * prefix
        full_avg = losses.get("recon_full_avg", torch.tensor(0.0))
        prefix_avg = losses.get("recon_prefix_avg", torch.tensor(0.0))
        losses["recon_total"] = full_avg + self.prefix_weight * prefix_avg

        if output_dict:
            return losses
        return losses["recon_total"]
