"""
Flow Matching Reconstruction Loss (Rectified Flow) for Stage 2.

Trains a conditional rectified-flow decoder to reconstruct pixel space from
register tokens, without requiring explicit full decoding each step.

Given:
  tokens: (B, n_registers, hidden_dim)
  x1: target image (B, 3, 224, 224)

We sample x0 ~ N(0, I), t ~ U(0,1) (or logit-normal), and compute:
  x_t = (1-t) x0 + t x1
  v_target = x1 - x0
  v_pred = decoder.predict_velocity(x_t, t, tokens)
  loss = ||v_pred - v_target||^2

Optionally, prefix reconstruction is included by running the same flow objective
with prefix tokens produced by the alignment model.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowReconstructionLoss(nn.Module):
    def __init__(
        self,
        loss_type: str = "mse",
        timestep_sampling: str = "logit_normal",
        pixel_mse_weight: float = 1.0,
        use_prefix_recon: bool = False,
        prefix_weight: float = 0.5,
    ):
        super().__init__()
        if loss_type != "mse":
            # Rectified flow is typically trained with MSE on velocity.
            # We keep the arg for CLI compatibility but enforce MSE.
            raise ValueError("FlowReconstructionLoss currently supports loss_type='mse' only")
        self.timestep_sampling = timestep_sampling
        self.pixel_mse_weight = float(pixel_mse_weight)
        self.use_prefix_recon = bool(use_prefix_recon)
        self.prefix_weight = float(prefix_weight)

    def sample_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.timestep_sampling == "logit_normal":
            u = torch.randn(batch_size, device=device) * 0.25
            t = torch.sigmoid(u)
        else:
            t = torch.rand(batch_size, device=device)
        return t

    def _flow_objective(
        self,
        x1: torch.Tensor,
        decoder: nn.Module,
        tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Single flow objective term for a given conditioning token tensor."""
        B = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = self.sample_timestep(B, x1.device)  # (B,)
        t_img = t.view(B, 1, 1, 1)
        x_t = (1.0 - t_img) * x0 + t_img * x1

        v_target = x1 - x0
        v_pred = decoder(x_t, t, tokens)

        flow_loss = F.mse_loss(v_pred, v_target)

        # x_t = (1-t)x0 + t x1, v = x1 - x0  =>  x1 = x_t + (1-t) v
        one_minus_t = (1.0 - t).view(B, 1, 1, 1)
        x1_pred = x_t + one_minus_t * v_pred
        pixel_mse = F.mse_loss(x1_pred, x1)

        total = flow_loss + self.pixel_mse_weight * pixel_mse
        return {
            "flow": flow_loss,
            "pixel_mse": pixel_mse,
            "total": total,
        }

    def forward(
        self,
        all_tokens: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        decoders: Dict[str, nn.Module],
        output_dict: bool = True,
        prefix_tokens: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=next(iter(targets.values())).device)
        n = 0

        for mod_name, tokens in all_tokens.items():
            if mod_name not in targets or mod_name not in decoders:
                continue
            x1 = targets[mod_name]
            decoder = decoders[mod_name]

            full_terms = self._flow_objective(x1, decoder, tokens)
            losses[f"flow_recon_{mod_name}"] = full_terms["flow"]
            losses[f"flow_recon_pixel_mse_{mod_name}"] = full_terms["pixel_mse"]

            mod_total = full_terms["total"]
            losses[f"flow_recon_total_{mod_name}"] = mod_total

            if self.use_prefix_recon and prefix_tokens is not None and mod_name in prefix_tokens:
                p_tokens = prefix_tokens[mod_name]
                if p_tokens is not None and p_tokens.shape[1] > 0:
                    prefix_terms = self._flow_objective(x1, decoder, p_tokens)
                    losses[f"flow_recon_prefix_{mod_name}"] = prefix_terms["flow"]
                    losses[f"flow_recon_prefix_pixel_mse_{mod_name}"] = prefix_terms["pixel_mse"]
                    losses[f"flow_recon_prefix_total_{mod_name}"] = prefix_terms["total"]
                    mod_total = mod_total + self.prefix_weight * prefix_terms["total"]

            losses[f"flow_recon_combined_total_{mod_name}"] = mod_total
            total = total + mod_total
            n += 1

        if n > 0:
            losses["recon_total"] = total / n
        else:
            losses["recon_total"] = total

        return losses if output_dict else losses["recon_total"]
