"""
Reconstruction Loss for FlexTok-style register token decoding.

Measures how well the decoder reconstructs the original image/tactile
from the register token bottleneck. Supports:
  - MSE (L2) loss in pixel space
  - L1 loss in pixel space
  - Optional perceptual loss via frozen encoder features (lightweight)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


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
