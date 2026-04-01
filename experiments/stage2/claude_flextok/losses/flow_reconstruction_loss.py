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
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowReconstructionLoss(nn.Module):
    def __init__(self, loss_type: str = "mse", timestep_sampling: str = "logit_normal"):
        super().__init__()
        if loss_type != "mse":
            # Rectified flow is typically trained with MSE on velocity.
            # We keep the arg for CLI compatibility but enforce MSE.
            raise ValueError("FlowReconstructionLoss currently supports loss_type='mse' only")
        self.timestep_sampling = timestep_sampling

    def sample_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.timestep_sampling == "logit_normal":
            u = torch.randn(batch_size, device=device) * 0.25
            t = torch.sigmoid(u)
        else:
            t = torch.rand(batch_size, device=device)
        return t

    def forward(
        self,
        all_tokens: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        decoders: Dict[str, nn.Module],
        output_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=next(iter(targets.values())).device)
        n = 0

        for mod_name, tokens in all_tokens.items():
            if mod_name not in targets or mod_name not in decoders:
                continue
            x1 = targets[mod_name]
            decoder = decoders[mod_name]

            B = x1.shape[0]
            x0 = torch.randn_like(x1)
            t = self.sample_timestep(B, x1.device)  # (B,)
            t_img = t.view(B, 1, 1, 1)
            x_t = (1.0 - t_img) * x0 + t_img * x1

            v_target = x1 - x0
            v_pred = decoder(x_t, t, tokens)

            loss = F.mse_loss(v_pred, v_target)
            losses[f"flow_recon_{mod_name}"] = loss
            total = total + loss
            n += 1

        if n > 0:
            losses["recon_total"] = total / n
        else:
            losses["recon_total"] = total

        return losses if output_dict else losses["recon_total"]

