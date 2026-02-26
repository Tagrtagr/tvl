"""
Flow Matching Alignment Loss (Alternative to Contrastive).

Instead of using contrastive loss to align cross-modal embeddings,
this module uses rectified flow matching to learn a direct mapping
between the shared embedding distributions of different modalities.

Inspired by:
  - CrossFlow (arXiv 2412.15213): Direct cross-modal flow matching
  - FlexTok decoder: Rectified flow for reconstruction
  - OmniFlow (arXiv 2412.01169): Multi-modal rectified flows

The flow model learns to transform one modality's shared representation
into another's, providing a generative alignment objective that may
capture richer cross-modal relationships than contrastive loss alone.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from itertools import combinations


class FlowMatchingHead(nn.Module):
    """
    A small MLP-based flow matching model that learns to map between
    two embedding distributions using rectified flow.

    Given source embedding X_0 (modality A) and target embedding X_1 (modality B):
      - Interpolate: X_t = (1 - t) * X_0 + t * X_1
      - Predict velocity: v_hat = model(X_t, t)
      - Target velocity: v = X_1 - X_0
      - Loss: ||v_hat - v||^2

    At inference, solve ODE from X_0 to predict X_1.

    Args:
        embed_dim: Dimension of the input embeddings.
        hidden_dim: Hidden dimension of the flow network.
        n_layers: Number of hidden layers.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        n_layers: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Flow prediction network with AdaLN-style conditioning
        layers = []
        layers.append(nn.Linear(embed_dim, hidden_dim))
        for _ in range(n_layers - 1):
            layers.append(AdaLNBlock(hidden_dim))
        self.layers = nn.ModuleList(layers)

        self.output_proj = nn.Linear(hidden_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        # Zero-init output projection (like DiT)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity field at (x_t, t).

        Args:
            x_t: (B, embed_dim) noised/interpolated embedding.
            t: (B,) or (B, 1) timestep in [0, 1].

        Returns:
            v_hat: (B, embed_dim) predicted velocity.
        """
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_embed(t.squeeze(-1))  # (B, hidden_dim)

        h = self.layers[0](x_t)  # Initial projection
        for layer in self.layers[1:]:
            h = layer(h, t_emb)

        return self.output_proj(h)


class AdaLNBlock(nn.Module):
    """MLP block with Adaptive Layer Normalization conditioned on timestep."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.adaLN = nn.Linear(hidden_dim, hidden_dim * 2)  # scale and shift
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale_shift = self.adaLN(t_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        return x + self.mlp(h)


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal embedding for continuous timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class FlowMatchingAlignmentLoss(nn.Module):
    """
    Rectified flow matching loss for cross-modal alignment.

    For each modality pair (A, B), trains a flow model to transform
    A's shared embedding distribution to B's (and optionally vice versa).

    This provides a richer alignment signal than contrastive loss:
      - Contrastive loss only pulls matched pairs together / pushes apart
      - Flow matching learns the full conditional transport map

    Args:
        modality_names: List of modality names.
        embed_dim: Embedding dimension (output of shared projectors).
        hidden_dim: Hidden dim of flow network.
        n_layers: Depth of flow network.
        bidirectional: Whether to train A->B and B->A flows.
        timestep_sampling: "uniform" or "logit_normal" (SD3 style).
    """

    def __init__(
        self,
        modality_names: List[str] = None,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        n_layers: int = 3,
        bidirectional: bool = True,
        timestep_sampling: str = "logit_normal",
    ):
        super().__init__()
        if modality_names is None:
            modality_names = ["vision", "tactile"]
        self.modality_names = modality_names
        self.bidirectional = bidirectional
        self.timestep_sampling = timestep_sampling

        # Create flow heads for each modality pair
        self.flow_heads = nn.ModuleDict()
        for mod_a, mod_b in combinations(modality_names, 2):
            self.flow_heads[f"{mod_a}_to_{mod_b}"] = FlowMatchingHead(
                embed_dim=embed_dim, hidden_dim=hidden_dim, n_layers=n_layers,
            )
            if bidirectional:
                self.flow_heads[f"{mod_b}_to_{mod_a}"] = FlowMatchingHead(
                    embed_dim=embed_dim, hidden_dim=hidden_dim, n_layers=n_layers,
                )

    def sample_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps for flow matching."""
        if self.timestep_sampling == "logit_normal":
            # Logit-normal with scale 0.25 (biases toward middle of [0,1])
            u = torch.randn(batch_size, device=device) * 0.25
            t = torch.sigmoid(u)
        else:
            t = torch.rand(batch_size, device=device)
        return t

    def rectified_flow_loss(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        flow_head: FlowMatchingHead,
    ) -> torch.Tensor:
        """
        Compute rectified flow matching loss.

        Args:
            x_0: (B, D) source modality shared embedding.
            x_1: (B, D) target modality shared embedding.
            flow_head: Flow prediction network.

        Returns:
            Scalar MSE loss on predicted velocity.
        """
        B = x_0.shape[0]
        t = self.sample_timestep(B, x_0.device)  # (B,)

        # Interpolate
        t_expand = t.unsqueeze(-1)  # (B, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1  # (B, D)

        # Target velocity
        v_target = x_1 - x_0  # (B, D)

        # Predict velocity
        v_pred = flow_head(x_t, t)

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)
        return loss

    def forward(
        self,
        alignment_output: Dict[str, torch.Tensor],
        output_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute flow matching alignment loss for all modality pairs.

        Args:
            alignment_output: Output from CrossModalAlignmentModel.forward().
            output_dict: Return detailed dict or just total loss.

        Returns:
            Dict with per-pair flow matching losses.
        """
        losses = {}
        total_loss = 0.0
        n_losses = 0

        active_modalities = [m for m in self.modality_names if f"{m}_shared" in alignment_output]
        pairs = list(combinations(active_modalities, 2))

        for mod_a, mod_b in pairs:
            feat_a = alignment_output[f"{mod_a}_shared"]
            feat_b = alignment_output[f"{mod_b}_shared"]

            # A -> B
            key_ab = f"{mod_a}_to_{mod_b}"
            if key_ab in self.flow_heads:
                loss_ab = self.rectified_flow_loss(feat_a, feat_b, self.flow_heads[key_ab])
                losses[f"flow_{key_ab}"] = loss_ab
                total_loss += loss_ab
                n_losses += 1

            # B -> A (if bidirectional)
            key_ba = f"{mod_b}_to_{mod_a}"
            if key_ba in self.flow_heads:
                loss_ba = self.rectified_flow_loss(feat_b, feat_a, self.flow_heads[key_ba])
                losses[f"flow_{key_ba}"] = loss_ba
                total_loss += loss_ba
                n_losses += 1

        if n_losses > 0:
            losses["flow_avg"] = total_loss / n_losses

        losses["flow_total"] = total_loss

        if output_dict:
            return losses
        return losses.get("flow_total", torch.tensor(0.0))

    @torch.no_grad()
    def transport(
        self,
        x_source: torch.Tensor,
        source_mod: str,
        target_mod: str,
        n_steps: int = 20,
    ) -> torch.Tensor:
        """
        Transport source embedding to target modality space via ODE solving.

        Args:
            x_source: (B, D) source modality embedding.
            source_mod: Name of source modality.
            target_mod: Name of target modality.
            n_steps: Number of Euler steps.

        Returns:
            (B, D) transported embedding in target modality space.
        """
        key = f"{source_mod}_to_{target_mod}"
        assert key in self.flow_heads, f"No flow head for {key}"
        flow_head = self.flow_heads[key]

        dt = 1.0 / n_steps
        x = x_source.clone()
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            v = flow_head(x, t)
            x = x + dt * v
        return x
