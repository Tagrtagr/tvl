"""
CURSOR IMPLEMENTATION - FluxTalk Register Tokens

Created by: Cursor (Composer)
Model: Composer (Cursor's AI coding assistant)
Date: February 7, 2026
Experiment ID: cursor_flux_talk_register_tokens
Version: 1.0

This package contains Cursor's implementation of Stage 2 cross-modality alignment
using FluxTalk-style register tokens with nested dropout.
"""

from experiments.stage2.cursor.register_tokens import RegisterTokenModule
from experiments.stage2.cursor.loss import RegisterTokenLoss

__all__ = ['RegisterTokenModule', 'RegisterTokenLoss']
