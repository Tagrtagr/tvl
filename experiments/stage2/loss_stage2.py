from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path to import tvl_enc modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from tvl_enc.tvl import ModalityType
from tvl_enc.loss import TVLLoss, construct_top_k_mask
from timm.utils import accuracy


class Stage2Loss(nn.Module):
    """
    Loss function for Stage 2 cross-modality alignment.
    
    Components:
    1. Contrastive loss on overlapping register tokens across modalities
    2. Optional regularization on complementary tokens to maintain diversity
    """
    
    def __init__(
        self,
        active_modalities=[ModalityType.VISION, ModalityType.TACTILE, ModalityType.TEXT],
        num_overlapping_tokens: int = 3,
        complementary_reg_weight: float = 0.01,
        similarity_thres: float = 0.9,
        disable_vision_text_loss: bool = False,
        disable_tactile_text_loss: bool = False,
    ):
        """
        Args:
            active_modalities: List of active modalities
            num_overlapping_tokens: Number of overlapping tokens to align
            complementary_reg_weight: Weight for complementary token regularization
            similarity_thres: Similarity threshold for text-based accuracy computation
            disable_vision_text_loss: Whether to disable vision-text loss
            disable_tactile_text_loss: Whether to disable tactile-text loss
        """
        super(Stage2Loss, self).__init__()
        self.active_modalities = active_modalities
        self.num_overlapping_tokens = num_overlapping_tokens
        self.complementary_reg_weight = complementary_reg_weight
        self.similarity_thres = similarity_thres
        self.disable_vision_text_loss = disable_vision_text_loss
        self.disable_tactile_text_loss = disable_tactile_text_loss
        
        # Base loss for contrastive learning
        self.base_loss = TVLLoss(
            active_modalities=active_modalities,
            similarity_thres=similarity_thres,
            disable_vision_text_loss=disable_vision_text_loss,
            disable_tactile_text_loss=disable_tactile_text_loss,
        )
    
    def compute_overlapping_token_loss(
        self,
        overlapping_tokens_dict: Dict[str, torch.Tensor],
        logit_scale: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive loss on overlapping tokens across modalities.
        
        Args:
            overlapping_tokens_dict: Dictionary mapping modality to overlapping tokens
                Shape: (batch_size, num_overlapping_tokens, token_dim)
            logit_scale: Logit scale parameter
            
        Returns:
            Dictionary with loss components and metrics
        """
        losses = {}
        total_loss = 0.0
        
        # Get all modality pairs
        modality_pairs = list(combinations(self.active_modalities, 2))
        
        for class_a, class_b in modality_pairs:
            class_a, class_b = sorted([class_a, class_b])
            
            # Skip disabled pairs
            if {class_a, class_b} == {ModalityType.VISION, ModalityType.TEXT} and self.disable_vision_text_loss:
                continue
            if {class_a, class_b} == {ModalityType.TEXT, ModalityType.TACTILE} and self.disable_tactile_text_loss:
                continue
            
            if class_a not in overlapping_tokens_dict or class_b not in overlapping_tokens_dict:
                continue
            
            tokens_a = overlapping_tokens_dict[class_a]  # (batch_size, num_overlapping_tokens, token_dim)
            tokens_b = overlapping_tokens_dict[class_b]
            
            # Average pool overlapping tokens to get single representation per modality
            # Shape: (batch_size, token_dim)
            feat_a = tokens_a.mean(dim=1)
            feat_b = tokens_b.mean(dim=1)
            
            # Normalize features
            feat_a = feat_a / (torch.norm(feat_a, dim=1, keepdim=True) + 1e-8)
            feat_b = feat_b / (torch.norm(feat_b, dim=1, keepdim=True) + 1e-8)
            
            # Compute contrastive loss
            loss, affinity_mat = self.base_loss.clip_loss(feat_a, feat_b, logit_scale)
            
            # Compute accuracy
            if ModalityType.TEXT in {class_a, class_b}:
                # For text pairs, use text similarity as ground truth
                text_modality = class_a if class_a == ModalityType.TEXT else class_b
                # We need access to original text features for GT distribution
                # For now, use standard accuracy
                acc1, acc5 = self.base_loss.get_acc_from_affinity(affinity_mat)
            else:
                acc1, acc5 = self.base_loss.get_acc_from_affinity(affinity_mat)
            
            pair_name = f"{class_a}_{class_b}"
            losses[f"{pair_name}_loss"] = loss
            losses[f"{pair_name}_acc1"] = acc1
            losses[f"{pair_name}_acc5"] = acc5
            total_loss += loss
        
        # Count actual pairs that were processed
        num_processed_pairs = len([k for k in losses.keys() if k.endswith('_loss')])
        losses["overlapping_loss"] = total_loss / num_processed_pairs if num_processed_pairs > 0 else torch.tensor(0.0, device=logit_scale.device)
        return losses
    
    def compute_complementary_regularization(
        self,
        complementary_tokens_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute regularization loss on complementary tokens to maintain diversity.
        
        Args:
            complementary_tokens_dict: Dictionary mapping modality to complementary tokens
                Shape: (batch_size, num_complementary_tokens, token_dim)
                
        Returns:
            Regularization loss (scalar)
        """
        if self.complementary_reg_weight == 0.0 or len(complementary_tokens_dict) == 0:
            if len(complementary_tokens_dict) > 0:
                return torch.tensor(0.0, device=next(iter(complementary_tokens_dict.values())).device)
            else:
                return torch.tensor(0.0)
        
        reg_loss = 0.0
        
        for modality, tokens in complementary_tokens_dict.items():
            # tokens: (batch_size, num_complementary_tokens, token_dim)
            batch_size = tokens.shape[0]
            
            # L2 regularization to prevent tokens from collapsing
            reg_loss += tokens.norm(dim=-1).mean()
            
            # Encourage diversity: penalize high similarity between complementary tokens
            # Compute pairwise similarity within tokens
            tokens_normalized = tokens / (torch.norm(tokens, dim=-1, keepdim=True) + 1e-8)
            similarity_matrix = torch.bmm(
                tokens_normalized, 
                tokens_normalized.transpose(1, 2)
            )  # (batch_size, num_complementary_tokens, num_complementary_tokens)
            
            # Mask diagonal (self-similarity)
            # Create mask for batch: (num_complementary_tokens, num_complementary_tokens)
            eye_mask = torch.eye(similarity_matrix.shape[1], device=tokens.device).bool()
            # Expand to batch dimension: (1, num_tokens, num_tokens)
            eye_mask = eye_mask.unsqueeze(0)
            # Get off-diagonal elements
            off_diagonal = similarity_matrix[:, ~eye_mask.squeeze(0)].abs().mean()
            
            # Penalize high off-diagonal similarity (encourage diversity)
            reg_loss += off_diagonal
        
        return reg_loss * self.complementary_reg_weight / len(complementary_tokens_dict)
    
    def forward(
        self,
        output_dict: Dict[str, torch.Tensor],
        logit_scale: torch.Tensor,
        output_dict_format: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Stage 2 loss.
        
        Args:
            output_dict: Output dictionary from TVLStage2.forward()
                Should contain:
                - 'overlapping_tokens': Dict mapping modality to overlapping tokens
                - 'complementary_tokens': Dict mapping modality to complementary tokens
                - Or 'overlapping_tokens' and 'complementary_tokens' as single tensors
            logit_scale: Logit scale parameter
            output_dict_format: Whether to return full dictionary of losses
            
        Returns:
            Loss dictionary or scalar loss
        """
        # Extract overlapping and complementary tokens per modality
        overlapping_tokens_dict = output_dict.get('overlapping_tokens', {})
        complementary_tokens_dict = output_dict.get('complementary_tokens', {})
        
        # Ensure we have dictionaries
        if not isinstance(overlapping_tokens_dict, dict):
            # If not a dict, create empty dict (shouldn't happen with updated model)
            overlapping_tokens_dict = {}
        if not isinstance(complementary_tokens_dict, dict):
            complementary_tokens_dict = {}
        
        # Compute overlapping token alignment loss
        overlapping_losses = self.compute_overlapping_token_loss(
            overlapping_tokens_dict, logit_scale
        )
        
        # Compute complementary token regularization
        complementary_reg = self.compute_complementary_regularization(
            complementary_tokens_dict
        )
        
        # Total loss
        total_loss = overlapping_losses["overlapping_loss"] + complementary_reg
        
        losses = {
            **overlapping_losses,
            "complementary_reg": complementary_reg,
            "total_loss": total_loss,
        }
        
        # Compute average accuracy
        acc_keys = [k for k in losses.keys() if "acc1" in k]
        if len(acc_keys) > 0:
            losses["average_acc1"] = torch.mean(torch.stack([losses[k] for k in acc_keys]))
        acc5_keys = [k for k in losses.keys() if "acc5" in k]
        if len(acc5_keys) > 0:
            losses["average_acc5"] = torch.mean(torch.stack([losses[k] for k in acc5_keys]))
        
        return losses if output_dict_format else total_loss
