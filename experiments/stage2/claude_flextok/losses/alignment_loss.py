"""
Cross-Modal Alignment Loss for Stage 2.

Combines:
  1. Contrastive loss on shared register tokens (CLIP-style) to align
     overlapping information across modalities.
  2. Private-token preservation via nested-drop tail masking +
     transformer-based rectified-flow latent modeling + latent-to-image decode.

Preservation pathway (per modality):
  - Randomly keep a prefix of private tokens; zero masked tail tokens.
  - Use a transformer flow head conditioned on masked private tokens to predict
    latent velocity in a rectified-flow objective.
  - Decode predicted latent to image space and supervise with pixel MSE.
"""

import math
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from typing import Dict, Optional, List


def topk_accuracy(output, target, topk=(1,)):
    """Top-k accuracy (matches timm.utils.accuracy interface)."""
    maxk = min(max(topk), output.size(1))
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal embedding for continuous timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class TransformerFlowLatentHead(nn.Module):
    """Transformer-conditioned rectified-flow head for token-sequence latent prediction."""

    def __init__(self, token_dim: int, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(token_dim * 2),
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.query_proj = nn.Linear(token_dim, token_dim)
        self.out = nn.Linear(token_dim, token_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_t: torch.Tensor, private_tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_t: (B, N, D), private_tokens: (B, Np, D), t: (B,)
        t_emb = self.time_embed(t).unsqueeze(1)
        q = self.query_proj(x_t) + t_emb
        seq = torch.cat([q, private_tokens], dim=1)
        h = self.encoder(seq)
        n_query = x_t.shape[1]
        return self.out(h[:, :n_query, :])


class LatentToImageDecoder(nn.Module):
    """Decode latent vector to image via spatial projection + transposed conv."""

    def __init__(self, latent_dim: int, base_channels: int = 64, out_channels: int = 3):
        super().__init__()
        self.h0 = 7
        self.w0 = 7
        self.base_channels = base_channels
        self.to_spatial = nn.Linear(latent_dim, base_channels * self.h0 * self.w0)

        c = base_channels
        c2, c4, c8 = max(c // 2, 8), max(c // 4, 8), max(c // 8, 8)
        self.up = nn.Sequential(
            _UpBlock(c, c),
            _UpBlock(c, c2),
            _UpBlock(c2, c4),
            _UpBlock(c4, c8),
            _UpBlock(c8, c8),
        )
        self.to_pixels = nn.Conv2d(c8, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        x = self.to_spatial(z).view(b, self.base_channels, self.h0, self.w0)
        x = self.up(x)
        return self.to_pixels(x)


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CrossModalAlignmentLoss(nn.Module):
    """Loss for Stage 2 cross-modal alignment."""

    def __init__(
        self,
        modality_names: List[str] = None,
        contrastive_weight: float = 1.0,
        preservation_weight: float = 0.5,
        temperature_clamp: float = 100.0,
        latent_dim: int = 768,
        private_nested_dropout_mode: str = "power_of_two",
        preservation_flow_weight: float = 1.0,
        preservation_image_weight: float = 1.0,
        flow_timestep_sampling: str = "logit_normal",
    ):
        super().__init__()
        if modality_names is None:
            modality_names = ["vision", "tactile"]
        self.modality_names = modality_names
        self.contrastive_weight = contrastive_weight
        self.preservation_weight = preservation_weight
        self.temperature_clamp = temperature_clamp
        self.private_nested_dropout_mode = private_nested_dropout_mode
        self.preservation_flow_weight = preservation_flow_weight
        self.preservation_image_weight = preservation_image_weight
        self.flow_timestep_sampling = flow_timestep_sampling
        self._debug_log_path = "/n/holylabs/ydu_lab/Lab/pwu/Projects/tactile_encoder/tvl/.cursor/debug-1a6460.log"

        self.private_flow_heads = nn.ModuleDict({
            m: TransformerFlowLatentHead(token_dim=latent_dim) for m in self.modality_names
        })
        # Match FlexTok-style nested token dropout semantics by masking with a learnable token.
        self.private_dropout_mask_tokens = nn.ParameterDict({
            m: nn.Parameter(torch.randn(latent_dim)) for m in self.modality_names
        })
        for mask in self.private_dropout_mask_tokens.values():
            nn.init.trunc_normal_(mask, std=0.02)

        self.latent_image_decoders = nn.ModuleDict({
            m: LatentToImageDecoder(latent_dim=latent_dim) for m in self.modality_names
        })

    # #region agent log
    def _debug_log(self, run_id: str, hypothesis_id: str, location: str, message: str, data: Dict) -> None:
        payload = {
            "sessionId": "1a6460",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        try:
            with open(self._debug_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            pass
    # #endregion

    def _build_k_values(self, n_tokens: int) -> List[int]:
        if n_tokens <= 0:
            return [0]
        if self.private_nested_dropout_mode == "uniform":
            return list(range(1, n_tokens + 1))
        vals = []
        k = 1
        while k <= n_tokens:
            vals.append(k)
            k *= 2
        if vals[-1] != n_tokens:
            vals.append(n_tokens)
        return vals

    def _sample_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.flow_timestep_sampling == "logit_normal":
            u = torch.randn(batch_size, device=device) * 0.25
            t = torch.sigmoid(u)
        else:
            t = torch.rand(batch_size, device=device)
        return t

    def contrastive_loss(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> tuple:
        # Clamp logit scale for stability
        logit_scale = torch.clamp(logit_scale, max=self.temperature_clamp)

        labels = torch.arange(feat_a.shape[0], device=feat_a.device, dtype=torch.long)
        affinity = logit_scale * feat_a @ feat_b.T
        loss_ab = F.cross_entropy(affinity, labels)
        loss_ba = F.cross_entropy(affinity.T, labels)
        return (loss_ab + loss_ba) / 2, affinity

    def preservation_loss(
        self,
        mod_name: str,
        private_tokens: torch.Tensor,
        frozen_features: torch.Tensor,
        target_image: torch.Tensor,
        projector: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Nested-drop private tokens, predict latent via transformer flow head,
        then decode latent to image.
        """
        device = private_tokens.device
        # #region agent log
        self._debug_log(
            run_id="initial",
            hypothesis_id="H1",
            location="alignment_loss.py:preservation_loss:entry",
            message="preservation_loss entry tensor shapes",
            data={
                "mod_name": mod_name,
                "private_shape": list(private_tokens.shape),
                "frozen_shape": list(frozen_features.shape),
                "private_last_dim": int(private_tokens.shape[-1]),
            },
        )
        # #endregion
        if private_tokens.shape[1] == 0:
            z = torch.zeros(private_tokens.shape[0], private_tokens.shape[-1], device=device)
            return {
                "preservation_flow": z.mean(),
                "preservation_image": z.mean(),
                "preservation_total": z.mean(),
            }

        # Target latent tokens from frozen features, projected token-wise to latent dim.
        if frozen_features.ndim == 2:
            z_target = frozen_features.unsqueeze(1)
        else:
            z_target = frozen_features
        if projector is not None:
            # #region agent log
            self._debug_log(
                run_id="initial",
                hypothesis_id="H3",
                location="alignment_loss.py:preservation_loss:projector",
                message="projector exists for preservation path",
                data={"mod_name": mod_name, "projector_type": projector.__class__.__name__},
            )
            # #endregion
            z_target = projector(z_target)
        z_target = z_target.detach()

        bsz, n_private, _ = private_tokens.shape
        n_target = z_target.shape[1]

        # Nested-drop on private tokens: keep random prefix, mask tail with learnable token.
        k_values = self._build_k_values(n_private)
        k_keep = k_values[torch.randint(0, len(k_values), (1,), device=device).item()]
        masked_private = private_tokens.clone()
        # #region agent log
        self._debug_log(
            run_id="initial",
            hypothesis_id="H2",
            location="alignment_loss.py:preservation_loss:mask_setup",
            message="mask setup dimensions before assignment",
            data={
                "mod_name": mod_name,
                "n_private": int(n_private),
                "k_keep": int(k_keep),
                "masked_private_shape": list(masked_private.shape),
                "mask_token_shape": list(self.private_dropout_mask_tokens[mod_name].shape),
            },
        )
        # #endregion
        if k_keep < n_private:
            mask_token = self.private_dropout_mask_tokens[mod_name].to(device=device, dtype=masked_private.dtype)
            # Be defensive: align mask token dim with private token dim.
            # This can happen if the loss module was constructed with a different latent_dim.
            target_dim = masked_private.shape[2]
            if mask_token.numel() != target_dim:
                if mask_token.numel() < target_dim:
                    mask_token = F.pad(mask_token, (0, target_dim - mask_token.numel()))
                else:
                    mask_token = mask_token[:target_dim]
            # #region agent log
            self._debug_log(
                run_id="initial",
                hypothesis_id="H2",
                location="alignment_loss.py:preservation_loss:mask_assign",
                message="attempting mask token broadcast assignment",
                data={
                    "mod_name": mod_name,
                    "target_slice_shape": [int(masked_private.shape[0]), int(n_private - k_keep), int(masked_private.shape[2])],
                    "mask_token_numel": int(mask_token.numel()),
                },
            )
            # #endregion
            masked_private[:, k_keep:, :] = mask_token.view(1, 1, -1)

        # Rectified-flow latent objective on full token sequence.
        z0 = torch.randn_like(z_target)
        t = self._sample_timestep(bsz, device)
        t_expand = t.view(-1, 1, 1)
        z_t = (1.0 - t_expand) * z0 + t_expand * z_target
        v_target = z_target - z0

        v_pred = self.private_flow_heads[mod_name](z_t, masked_private, t)
        # #region agent log
        self._debug_log(
            run_id="initial",
            hypothesis_id="H4",
            location="alignment_loss.py:preservation_loss:flow_shapes",
            message="flow head input/output dimensions",
            data={
                "mod_name": mod_name,
                "z_t_shape": list(z_t.shape),
                "masked_private_shape": list(masked_private.shape),
                "v_pred_shape": list(v_pred.shape),
                "v_target_shape": list(v_target.shape),
            },
        )
        # #endregion
        flow_loss = F.mse_loss(v_pred, v_target)

        # One-step denoised latent-token estimate + image reconstruction supervision.
        z_pred = z0 + v_pred
        z_pred_pooled = z_pred.mean(dim=1) if n_target > 1 else z_pred[:, 0, :]
        img_pred = self.latent_image_decoders[mod_name](z_pred_pooled)
        img_loss = F.mse_loss(img_pred, target_image)

        total = self.preservation_flow_weight * flow_loss + self.preservation_image_weight * img_loss
        return {
            "preservation_flow": flow_loss,
            "preservation_image": img_loss,
            "preservation_total": total,
        }

    def get_acc_from_affinity(self, affinity_matrix: torch.Tensor) -> tuple:
        """Compute top-1 and top-5 retrieval accuracy from affinity matrix."""
        labels = torch.arange(affinity_matrix.shape[0], device=affinity_matrix.device, dtype=torch.long)
        acc1, acc5 = topk_accuracy(affinity_matrix, labels, topk=(1, min(5, affinity_matrix.shape[0])))
        acc1_t, acc5_t = topk_accuracy(affinity_matrix.T, labels, topk=(1, min(5, affinity_matrix.shape[0])))
        return (acc1 + acc1_t) / 2, (acc5 + acc5_t) / 2

    def forward(
        self,
        alignment_output: Dict[str, torch.Tensor],
        frozen_features: Optional[Dict[str, torch.Tensor]] = None,
        preservation_projectors: Optional[Dict[str, nn.Module]] = None,
        input_images: Optional[Dict[str, torch.Tensor]] = None,
        output_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined alignment loss."""
        logit_scale = alignment_output["logit_scale"]
        losses = {}
        total_contrastive = 0.0
        total_preservation = 0.0
        n_pairs = 0

        # Contrastive loss for all modality pairs
        active_modalities = [m for m in self.modality_names if f"{m}_shared" in alignment_output]
        pairs = list(combinations(active_modalities, 2))

        for mod_a, mod_b in pairs:
            feat_a = alignment_output[f"{mod_a}_shared"]
            feat_b = alignment_output[f"{mod_b}_shared"]
            loss, affinity = self.contrastive_loss(feat_a, feat_b, logit_scale)
            acc1, acc5 = self.get_acc_from_affinity(affinity)

            pair_name = f"{mod_a}_{mod_b}"
            losses[f"contrastive_{pair_name}"] = loss
            losses[f"acc1_{pair_name}"] = acc1
            losses[f"acc5_{pair_name}"] = acc5
            total_contrastive += loss
            n_pairs += 1

        if n_pairs > 0:
            losses["contrastive_avg"] = total_contrastive / n_pairs

        # Preservation loss per modality
        if (
            frozen_features is not None
            and input_images is not None
            and self.preservation_weight > 0
        ):
            n_pres = 0
            for mod_name in active_modalities:
                if mod_name in frozen_features and mod_name in input_images:
                    private = alignment_output.get(f"{mod_name}_private")
                    if private is not None:
                        proj = preservation_projectors[mod_name] if (preservation_projectors and mod_name in preservation_projectors) else None
                        p = self.preservation_loss(
                            mod_name=mod_name,
                            private_tokens=private,
                            frozen_features=frozen_features[mod_name],
                            target_image=input_images[mod_name],
                            projector=proj,
                        )
                        losses[f"preservation_flow_{mod_name}"] = p["preservation_flow"]
                        losses[f"preservation_image_{mod_name}"] = p["preservation_image"]
                        losses[f"preservation_{mod_name}"] = p["preservation_total"]
                        total_preservation += p["preservation_total"]
                        n_pres += 1

            if n_pres > 0:
                losses["preservation_avg"] = total_preservation / n_pres

        # Combined loss
        total = self.contrastive_weight * losses.get("contrastive_avg", torch.tensor(0.0, device=logit_scale.device))
        total += self.preservation_weight * losses.get("preservation_avg", torch.tensor(0.0, device=logit_scale.device))
        losses["total_loss"] = total

        # Average accuracy across pairs
        acc1_vals = [v for k, v in losses.items() if k.startswith("acc1_")]
        acc5_vals = [v for k, v in losses.items() if k.startswith("acc5_")]
        if acc1_vals:
            losses["acc1_avg"] = torch.stack(acc1_vals).mean()
            losses["acc5_avg"] = torch.stack(acc5_vals).mean()

        losses["logit_scale"] = logit_scale

        if output_dict:
            return losses
        return losses["total_loss"]
