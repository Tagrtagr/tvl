from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tvl_enc.tvl import ModalityType

try:
    from diffusers.models import AutoencoderKL
except ImportError as e:
    AutoencoderKL = None
    _DIFFUSERS_IMPORT_ERROR = e
else:
    _DIFFUSERS_IMPORT_ERROR = None


VAE_BASE_CFG = {
    "_class_name": "AutoencoderKL",
    "_diffusers_version": "0.18.0.dev0",
    "_name_or_path": ".",
    "act_fn": "silu",
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ],
    "in_channels": 3,
    "latent_channels": None,
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "out_channels": 3,
    "sample_size": 224,
    "scaling_factor": 0.18215,
    "up_block_types": [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ],
}


class ConvVAE(nn.Module):
    """
    FlexTok-style VAE wrapper around diffusers AutoencoderKL.

    This keeps the old ConvVAE name for backward compatibility with existing imports.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        scaling_factor: Optional[float] = None,
        hf_hub_path: Optional[str] = None,
        sample_posterior: bool = True,
    ):
        super().__init__()
        if AutoencoderKL is None:
            raise ImportError(
                "diffusers is required for FlexTok-style VAE. "
                f"Original error: {_DIFFUSERS_IMPORT_ERROR}"
            )

        if hf_hub_path is not None:
            self.vae = AutoencoderKL.from_pretrained(hf_hub_path, low_cpu_mem_usage=False)
        else:
            vae_cfg = VAE_BASE_CFG.copy()
            vae_cfg["in_channels"] = in_channels
            vae_cfg["out_channels"] = in_channels
            vae_cfg["latent_channels"] = latent_channels
            self.vae = AutoencoderKL.from_config(vae_cfg)

        self.sample_posterior = sample_posterior
        self.scaling_factor = (
            scaling_factor
            if scaling_factor is not None
            else float(getattr(self.vae.config, "scaling_factor", 0.18215))
        )

    @property
    def latent_channels(self) -> int:
        return int(getattr(self.vae.config, "latent_channels"))

    def encode(self, x: torch.Tensor):
        dist = self.vae.encode(x).latent_dist
        mu = dist.mean * self.scaling_factor
        logvar = dist.logvar
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        z = z / self.scaling_factor
        return self.vae.decode(z).sample

    def forward(self, x: torch.Tensor):
        dist = self.vae.encode(x).latent_dist
        if self.sample_posterior:
            z = dist.sample() * self.scaling_factor
        else:
            z = dist.mode() * self.scaling_factor
        x_rec = self.decode(z)
        mu = dist.mean * self.scaling_factor
        logvar = dist.logvar
        return x_rec, mu, logvar

    def load_pretrained_checkpoint(self, checkpoint_path: str, strict: bool = False):
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint_path}")

        if ckpt_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file as load_safetensors
            except ImportError as e:
                raise ImportError(
                    "safetensors is required to load .safetensors checkpoints"
                ) from e
            state = load_safetensors(str(ckpt_path))
        else:
            raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            state = raw.get("model", raw.get("state_dict", raw))

        # Try direct VAE load first.
        missing, unexpected = self.vae.load_state_dict(state, strict=strict)
        if len(missing) == 0 and len(unexpected) == 0:
            return missing, unexpected

        # FlexTok checkpoints may prefix parameters (e.g., 'vae.' or 'vae.vae.').
        prefix_candidates = ["vae.", "vae.vae.", "module.vae.", "module.vae.vae."]
        for prefix in prefix_candidates:
            sub_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if not sub_state:
                continue
            m2, u2 = self.vae.load_state_dict(sub_state, strict=False)
            if len(m2) < len(missing):
                return m2, u2

        return missing, unexpected


class VAEFeatureBackbone(nn.Module):
    """Frozen VAE encoder that outputs token sequences for stage-2 alignment."""

    def __init__(self, hidden_dim: int = 768, latent_channels: int = 4, scaling_factor: float = 0.18215):
        super().__init__()
        self.vae = ConvVAE(in_channels=3, latent_channels=latent_channels, scaling_factor=scaling_factor)
        self.hidden_dim = hidden_dim
        self.latent_channels = latent_channels

    def load_pretrained_checkpoint(self, checkpoint_path: str, strict: bool = False):
        return self.vae.load_pretrained_checkpoint(checkpoint_path=checkpoint_path, strict=strict)

    def _expand_latent_dim(self, tokens: torch.Tensor) -> torch.Tensor:
        """Deterministically map latent channels -> hidden_dim without trainable params."""
        _, _, c = tokens.shape
        if c == self.hidden_dim:
            return tokens
        repeat = self.hidden_dim // c
        rem = self.hidden_dim % c
        expanded = tokens.repeat(1, 1, max(repeat, 1))
        if rem > 0:
            expanded = torch.cat([expanded, tokens[:, :, :rem]], dim=-1)
        return expanded[:, :, :self.hidden_dim]

    @torch.no_grad()
    def encode_tokens(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out_dict = {}
        for mod in [ModalityType.VISION, ModalityType.TACTILE]:
            if mod not in input_dict:
                continue
            x = input_dict[mod]
            mu, _ = self.vae.encode(x)
            # (B, C, H, W) -> (B, N, C) -> (B, N, hidden_dim)
            tokens = mu.flatten(2).transpose(1, 2)
            tokens = self._expand_latent_dim(tokens)
            tokens = F.normalize(tokens, dim=-1)
            out_dict[mod] = tokens
        out_dict["logit_scale"] = torch.tensor(1.0, device=next(self.parameters()).device)
        return out_dict

    @torch.no_grad()
    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        return_token_sequences: bool = True,
        drop_cls_token: bool = True,
    ):
        _ = return_token_sequences, drop_cls_token
        return self.encode_tokens(input_dict)
