<<<<<<< Current (Your changes)
=======
"""
Stage 2 Model: FluxTalk Register Tokens Implementation

Created by: Cursor (Composer)
Model: Composer (Cursor's AI coding assistant)
Date: February 7, 2026
Experiment ID: flux_talk_register_tokens
Version: 1.0

This implementation uses FluxTalk-style register tokens with nested dropout
for cross-modality alignment in Stage 2 of the TVL pipeline.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import sys
import os

# Add parent directory to path to import tvl_enc modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../'))
from tvl_enc.tvl import TVL, ModalityType
from tvl_enc.transformer_utils import CrossAttentionBlock


class TVLStage2(nn.Module):
    """
    Stage 2 model for cross-modality alignment using FluxTalk-style register tokens.
    
    Architecture:
    - Frozen Stage 1 TVL encoders
    - Learnable register tokens (4 tokens: 3 overlapping + 1 complementary)
    - Cross-attention transformer layers between register tokens and modality features
    - Nested dropout during training
    """
    
    def __init__(
        self,
        stage1_model: TVL,
        num_register_tokens: int = 4,
        num_overlapping_tokens: int = 3,
        register_token_dim: int = 768,
        num_cross_attn_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        nested_dropout_rate: float = 0.5,
    ):
        """
        Args:
            stage1_model: Pre-trained Stage 1 TVL model (will be frozen)
            num_register_tokens: Total number of register tokens (default: 4)
            num_overlapping_tokens: Number of tokens for overlapping info (default: 3)
            register_token_dim: Dimension of register tokens (default: 768)
            num_cross_attn_layers: Number of cross-attention transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Drop path rate
            nested_dropout_rate: Base dropout rate for nested dropout
        """
        super(TVLStage2, self).__init__()
        
        assert num_overlapping_tokens < num_register_tokens, \
            "num_overlapping_tokens must be less than num_register_tokens"
        
        self.stage1_model = stage1_model
        self.num_register_tokens = num_register_tokens
        self.num_overlapping_tokens = num_overlapping_tokens
        self.num_complementary_tokens = num_register_tokens - num_overlapping_tokens
        self.register_token_dim = register_token_dim
        self.nested_dropout_rate = nested_dropout_rate
        
        # Freeze Stage 1 model
        for param in self.stage1_model.parameters():
            param.requires_grad = False
        
        # Get feature dimension from Stage 1 model
        # For tactile encoder, we need to check the output dimension
        # For CLIP, it's self.clip.transformer.width
        if hasattr(self.stage1_model, 'common_latent_dim') and self.stage1_model.common_latent_dim is not None:
            self.feature_dim = self.stage1_model.common_latent_dim
        else:
            # Default to CLIP width
            self.feature_dim = self.stage1_model.clip.transformer.width
        
        # Ensure register_token_dim matches feature_dim
        if register_token_dim != self.feature_dim:
            print(f"Warning: register_token_dim ({register_token_dim}) != feature_dim ({self.feature_dim}). "
                  f"Using feature_dim={self.feature_dim}")
            self.register_token_dim = self.feature_dim
        
        # Initialize register tokens as learnable parameters
        # Shape: (num_register_tokens, register_token_dim)
        self.register_tokens = nn.Parameter(
            torch.randn(num_register_tokens, self.register_token_dim) * 0.02
        )
        
        # Cross-attention transformer layers
        # Query: register tokens, Key/Value: modality features
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(
                encoder_dim=self.feature_dim,
                decoder_dim=self.register_token_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=False,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                self_attn=(i > 0),  # Add self-attention after first layer
            )
            for i in range(num_cross_attn_layers)
        ])
        
        # Layer norm for register tokens
        self.register_norm = nn.LayerNorm(self.register_token_dim)
        
    def get_register_tokens(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns overlapping and complementary register tokens.
        
        Returns:
            overlapping_tokens: (num_overlapping_tokens, register_token_dim)
            complementary_tokens: (num_complementary_tokens, register_token_dim)
        """
        overlapping_tokens = self.register_tokens[:self.num_overlapping_tokens]
        complementary_tokens = self.register_tokens[self.num_overlapping_tokens:]
        return overlapping_tokens, complementary_tokens
    
    def apply_nested_dropout(
        self, 
        register_tokens: torch.Tensor, 
        training: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply nested dropout to register tokens.
        
        Dropout probability increases for later tokens:
        - Token 0: 0%
        - Token 1: nested_dropout_rate * 0.33
        - Token 2: nested_dropout_rate * 0.67
        - Token 3: nested_dropout_rate * 1.0
        
        Args:
            register_tokens: (batch_size, num_tokens, token_dim)
            training: Whether in training mode
            
        Returns:
            masked_tokens: Register tokens with nested dropout applied
            mask: Boolean mask indicating which tokens were kept (for loss computation)
        """
        if not training:
            return register_tokens, None
        
        batch_size, num_tokens, token_dim = register_tokens.shape
        device = register_tokens.device
        
        # Create nested dropout schedule
        # Probability increases linearly for each token
        dropout_probs = torch.linspace(
            0.0, 
            self.nested_dropout_rate, 
            num_tokens,
            device=device
        )
        
        # Sample dropout mask for each token
        # Keep token if random value > dropout_prob
        random_values = torch.rand(batch_size, num_tokens, device=device)
        mask = random_values > dropout_probs.unsqueeze(0)  # (batch_size, num_tokens)
        
        # Apply mask: set dropped tokens to zero
        mask_expanded = mask.unsqueeze(-1).float()  # (batch_size, num_tokens, 1)
        masked_tokens = register_tokens * mask_expanded
        
        return masked_tokens, mask
    
    def forward(
        self, 
        input_dict: Dict[str, torch.Tensor],
        apply_nested_dropout: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Stage 2 model.
        
        Args:
            input_dict: Dictionary with keys 'vision', 'tactile', 'text'
            apply_nested_dropout: Whether to apply nested dropout (default: True in training)
            
        Returns:
            Dictionary containing:
            - 'overlapping_tokens': (batch_size, num_overlapping_tokens, token_dim)
            - 'complementary_tokens': (batch_size, num_complementary_tokens, token_dim)
            - 'register_tokens': (batch_size, num_register_tokens, token_dim) - all tokens
            - 'stage1_features': Original Stage 1 features
            - 'nested_dropout_mask': Mask used for nested dropout (if applied)
        """
        # Forward through Stage 1 encoders (frozen)
        with torch.no_grad():
            stage1_features = self.stage1_model(input_dict)
        
        # Extract modality features
        modality_features = []
        for modality in self.stage1_model.active_modalities:
            if modality in stage1_features:
                feat = stage1_features[modality]  # (batch_size, feature_dim)
                # Expand to (batch_size, 1, feature_dim) for concatenation
                modality_features.append(feat.unsqueeze(1))
        
        # Concatenate modality features
        # Shape: (batch_size, num_modalities, feature_dim)
        concatenated_features = torch.cat(modality_features, dim=1)
        
        # Expand register tokens to batch size
        batch_size = concatenated_features.shape[0]
        register_tokens = self.register_tokens.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (batch_size, num_register_tokens, register_token_dim)
        
        # Apply nested dropout
        if apply_nested_dropout and self.training:
            register_tokens, nested_dropout_mask = self.apply_nested_dropout(
                register_tokens, training=True
            )
        else:
            nested_dropout_mask = None
        
        # Process each modality separately to get modality-specific token representations
        # This allows us to align overlapping tokens across modalities
        modality_token_dict = {}
        
        for modality in self.stage1_model.active_modalities:
            if modality not in stage1_features:
                continue
            
            # Get modality-specific features
            modality_feat = stage1_features[modality]  # (batch_size, feature_dim)
            modality_feat_expanded = modality_feat.unsqueeze(1)  # (batch_size, 1, feature_dim)
            
            # Apply cross-attention layers with this modality's features
            # Start fresh with original register tokens for each modality
            x = register_tokens.clone()
            for layer in self.cross_attn_layers:
                x = layer(x, modality_feat_expanded)
            
            # Apply layer norm
            x = self.register_norm(x)
            
            # Split into overlapping and complementary tokens
            overlapping_tokens = x[:, :self.num_overlapping_tokens, :]
            complementary_tokens = x[:, self.num_overlapping_tokens:, :]
            
            # Normalize tokens for contrastive learning
            overlapping_tokens = overlapping_tokens / (
                torch.norm(overlapping_tokens, dim=-1, keepdim=True) + 1e-8
            )
            complementary_tokens = complementary_tokens / (
                torch.norm(complementary_tokens, dim=-1, keepdim=True) + 1e-8
            )
            
            modality_token_dict[modality] = {
                'overlapping': overlapping_tokens,
                'complementary': complementary_tokens,
            }
        
        # Also compute shared register tokens (for reference)
        x_shared = register_tokens
        for layer in self.cross_attn_layers:
            x_shared = layer(x_shared, concatenated_features)
        x_shared = self.register_norm(x_shared)
        
        return {
            'overlapping_tokens': {k: v['overlapping'] for k, v in modality_token_dict.items()},
            'complementary_tokens': {k: v['complementary'] for k, v in modality_token_dict.items()},
            'register_tokens': x_shared,
            'stage1_features': stage1_features,
            'nested_dropout_mask': nested_dropout_mask,
        }
>>>>>>> Incoming (Background Agent changes)
