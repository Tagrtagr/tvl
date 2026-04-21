from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from tvl_enc.tvl import ModalityType, TVL

from models.cross_modal_alignment import CrossModalAlignmentModel
from models.simple_vae import VAEFeatureBackbone


class IdentityTokenizer(nn.Module):
    """Default tokenizer placeholder when FSQ is internal to cross-modal module."""

    def forward(self, alignment_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return alignment_output


class DecoderCollection(nn.Module):
    """ModuleDict wrapper for modality-specific decoders."""

    def __init__(self, decoders: Dict[str, nn.Module]):
        super().__init__()
        self.decoders = nn.ModuleDict(decoders)

    def forward(self, modality_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            mod_name: self.decoders[mod_name](tokens)
            for mod_name, tokens in modality_tokens.items()
            if mod_name in self.decoders
        }

    def get(self, mod_name: str) -> Optional[nn.Module]:
        return self.decoders.get(mod_name)

    def values(self):
        return self.decoders.values()

    def items(self):
        return self.decoders.items()


@dataclass
class PipelineOutput:
    frozen_features: Dict[str, torch.Tensor]
    alignment_output: Dict[str, torch.Tensor]


class Stage2Pipeline(nn.Module):
    """
    High-level Stage-2 pipeline:
      1) Modality-specific encoder
      2) Register tokens + cross-modality encoder
      3) FSQ tokenizer (optional external stage, identity by default)
      4) Decoder stack (optional, trained outside the main forward)
    """

    def __init__(
        self,
        modality_encoder: nn.Module,
        cross_modality_encoder: CrossModalAlignmentModel,
        tokenizer: Optional[nn.Module] = None,
        decoder_collection: Optional[DecoderCollection] = None,
    ):
        super().__init__()
        self.modality_encoder = modality_encoder
        self.cross_modality_encoder = cross_modality_encoder
        self.tokenizer = tokenizer if tokenizer is not None else IdentityTokenizer()
        self.decoder_collection = decoder_collection

        for param in self.modality_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.modality_encoder.eval()
        out = self.modality_encoder(
            input_dict,
            return_token_sequences=True,
            drop_cls_token=True,
        )
        out.pop("logit_scale", None)
        out.pop("logit_bias", None)
        return out

    def forward_alignment(self, frozen_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        alignment_output = self.cross_modality_encoder(frozen_features)
        return self.tokenizer(alignment_output)

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        frozen_features = self.encode(input_dict)
        return self.forward_alignment(frozen_features)


def build_modality_encoder(
    feature_backbone: str,
    tactile_model: str,
    stage1_checkpoint: Optional[str],
    vae_checkpoint: Optional[str],
    vae_latent_channels: int,
    vae_scaling_factor: float,
) -> nn.Module:
    if feature_backbone == "vae":
        encoder = VAEFeatureBackbone(
            hidden_dim=768,
            latent_channels=vae_latent_channels,
            scaling_factor=vae_scaling_factor,
        )
        if not vae_checkpoint:
            raise FileNotFoundError("VAE backbone requires a valid vae_checkpoint")
        missing, unexpected = encoder.load_pretrained_checkpoint(vae_checkpoint, strict=False)
        if missing:
            print(f"[warn] VAE missing keys: {len(missing)}")
        if unexpected:
            print(f"[warn] VAE unexpected keys: {len(unexpected)}")
        return encoder

    encoder = TVL(
        tactile_model=tactile_model,
        active_modalities=[ModalityType.VISION, ModalityType.TACTILE],
    )
    if stage1_checkpoint:
        ckpt = torch.load(stage1_checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        encoder.load_state_dict(state_dict, strict=False)
    return encoder
