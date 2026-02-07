"""
Cross-Modal Alignment Loss for Stage 2.

Combines:
  1. Contrastive loss on shared register tokens (CLIP-style) to align
     overlapping information across modalities.
  2. Reconstruction preservation loss on private register tokens to ensure
     modality-specific information is not lost.

The contrastive loss operates on the pooled+projected shared embeddings.
The preservation loss ensures private tokens still carry useful information
by requiring them to reconstruct the original frozen encoder features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from typing import Dict, Optional, List
from timm.utils import accuracy


class CrossModalAlignmentLoss(nn.Module):
    """
    Loss for Stage 2 cross-modal alignment.

    Components:
        1. contrastive_loss: Symmetric CLIP-style contrastive loss between
           shared embeddings of different modalities.
        2. preservation_loss: MSE between pooled private tokens and frozen
           encoder features (ensures private tokens preserve modality info).

    Args:
        modality_names: List of modality names to align.
        contrastive_weight: Weight for contrastive alignment loss.
        preservation_weight: Weight for information preservation loss.
        temperature_clamp: Max/min for learned temperature to prevent instability.
    """

    def __init__(
        self,
        modality_names: List[str] = None,
        contrastive_weight: float = 1.0,
        preservation_weight: float = 0.5,
        temperature_clamp: float = 100.0,
    ):
        super().__init__()
        if modality_names is None:
            modality_names = ["vision", "tactile"]
        self.modality_names = modality_names
        self.contrastive_weight = contrastive_weight
        self.preservation_weight = preservation_weight
        self.temperature_clamp = temperature_clamp

    def contrastive_loss(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> tuple:
        """
        Symmetric CLIP-style contrastive loss.

        Args:
            feat_a: (B, D) L2-normalized shared embedding for modality A.
            feat_b: (B, D) L2-normalized shared embedding for modality B.
            logit_scale: Scalar temperature (exp of learned parameter).

        Returns:
            loss: Scalar contrastive loss.
            affinity_matrix: (B, B) similarity matrix for metrics.
        """
        # Clamp logit scale for stability
        logit_scale = torch.clamp(logit_scale, max=self.temperature_clamp)

        labels = torch.arange(feat_a.shape[0], device=feat_a.device, dtype=torch.long)
        affinity = logit_scale * feat_a @ feat_b.T
        loss_ab = F.cross_entropy(affinity, labels)
        loss_ba = F.cross_entropy(affinity.T, labels)
        return (loss_ab + loss_ba) / 2, affinity

    def preservation_loss(
        self,
        private_tokens: torch.Tensor,
        frozen_features: torch.Tensor,
        projector: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Ensures private register tokens preserve modality-specific information.

        Uses cosine similarity loss between pooled private tokens and the
        original frozen encoder features.

        Args:
            private_tokens: (B, n_private, hidden_dim) private register tokens.
            frozen_features: (B, D) original frozen encoder features.
            projector: Optional linear projection to match dimensions.

        Returns:
            Scalar preservation loss (1 - cosine_similarity).
        """
        if private_tokens.shape[1] == 0:
            return torch.tensor(0.0, device=private_tokens.device)

        pooled = private_tokens.mean(dim=1)  # (B, hidden_dim)
        if projector is not None:
            pooled = projector(pooled)

        # Cosine similarity loss
        loss = 1.0 - F.cosine_similarity(pooled, frozen_features.detach(), dim=-1).mean()
        return loss

    def get_acc_from_affinity(self, affinity_matrix: torch.Tensor) -> tuple:
        """Compute top-1 and top-5 retrieval accuracy from affinity matrix."""
        labels = torch.arange(affinity_matrix.shape[0], device=affinity_matrix.device, dtype=torch.long)
        acc1, acc5 = accuracy(affinity_matrix, labels, topk=(1, min(5, affinity_matrix.shape[0])))
        acc1_t, acc5_t = accuracy(affinity_matrix.T, labels, topk=(1, min(5, affinity_matrix.shape[0])))
        return (acc1 + acc1_t) / 2, (acc5 + acc5_t) / 2

    def forward(
        self,
        alignment_output: Dict[str, torch.Tensor],
        frozen_features: Optional[Dict[str, torch.Tensor]] = None,
        preservation_projectors: Optional[Dict[str, nn.Module]] = None,
        output_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined alignment loss.

        Args:
            alignment_output: Output from CrossModalAlignmentModel.forward().
            frozen_features: Original frozen encoder features for preservation loss.
            preservation_projectors: Optional projectors for matching dims.
            output_dict: If True, return dict of all losses and metrics.

        Returns:
            Dict with losses and accuracy metrics, or scalar average loss.
        """
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
        if frozen_features is not None and self.preservation_weight > 0:
            for mod_name in active_modalities:
                if mod_name in frozen_features:
                    private = alignment_output.get(f"{mod_name}_private")
                    if private is not None:
                        proj = preservation_projectors[mod_name] if (preservation_projectors and mod_name in preservation_projectors) else None
                        p_loss = self.preservation_loss(
                            private, frozen_features[mod_name], proj
                        )
                        losses[f"preservation_{mod_name}"] = p_loss
                        total_preservation += p_loss

            if len(active_modalities) > 0:
                losses["preservation_avg"] = total_preservation / len(active_modalities)

        # Combined loss
        total = self.contrastive_weight * losses.get("contrastive_avg", 0.0)
        total += self.preservation_weight * losses.get("preservation_avg", 0.0)
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
