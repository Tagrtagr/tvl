import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import timm 
import open_clip
from typing import Any, Dict, Optional
from types import SimpleNamespace
from collections import OrderedDict

ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    TACTILE="tactile"
)

CLIP_VISION_MODEL = "ViT-L-14"
CLIP_PRETRAIN_DATA = "datacomp_xl_s13b_b90k"

tokenizer = open_clip.get_tokenizer(CLIP_VISION_MODEL)

class TVL(nn.Module):
    def __init__(
        self, active_modalities = [ModalityType.VISION, ModalityType.TACTILE, ModalityType.TEXT], 
        clip_vision_model=CLIP_VISION_MODEL, 
        clip_pretrain_data=CLIP_PRETRAIN_DATA, 
        tactile_model="vit_tiny_patch16_224", 
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        common_latent_dim: int = None, # for imagebind this is set to 1024, and last layer has width 1280 (ViT-H-14)
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super(TVL, self).__init__()
        assert len(active_modalities) > 1, "At least two modalities must be active"
        self.active_modalities = active_modalities
        self.clip, _, self.vision_preprocess = open_clip.create_model_and_transforms(clip_vision_model, pretrained=clip_pretrain_data)
        self.tokenizer = open_clip.get_tokenizer(clip_vision_model)
        
        if common_latent_dim is not None: 
            # then we will put all the modality head self.modality_head
            assert common_latent_dim > 0, "common_latent_dim must be positive"
            num_classes = 0 
        else:
            # we merge the modality head into the model
            num_classes = self.clip.transformer.width
        
        self.tactile_encoder = timm.create_model(tactile_model, pretrained=False, num_classes=num_classes, global_pool="avg", drop_rate=drop_rate, drop_path_rate=drop_path_rate)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

        modality_heads = {}
        self.common_latent_dim = common_latent_dim
        if common_latent_dim is not None:
            for modality in self.active_modalities:
                if modality == ModalityType.TACTILE:
                    modality_heads[modality] = nn.Linear(self.tactile_encoder.num_features, common_latent_dim, bias=False)
                else:
                    modality_heads[modality] = nn.Linear(self.clip.transformer.width, common_latent_dim, bias=False)
        self.modality_heads = nn.ModuleDict(modality_heads)

        # by default, we freeze openclip 
        self.freeze_vision()
        self.freeze_text()

        if ModalityType.VISION not in self.active_modalities:
            # we remove the clip.visual module
            del self.clip.visual
        if ModalityType.TEXT not in self.active_modalities:
            # we remove the clip.transformer module
            del self.clip.transformer
        
        # we clear torch cache 
        torch.cuda.empty_cache()

    def freeze_openclip(self):
        for param in self.clip.parameters():
            param.requires_grad = False

    def freeze_vision(self):
        for param in self.clip.visual.parameters():
            param.requires_grad = False
    
    def freeze_tactile(self):
        for param in self.tactile_encoder.parameters():
            param.requires_grad = False
    
    def freeze_text(self):
        for param in self.clip.transformer.parameters():
            param.requires_grad = False

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super(TVL, self).state_dict(destination, prefix, keep_vars)
        # we remove all clip related weights and only save the tactile encoder
        new_state_dict = OrderedDict()
        for k in state_dict:
            if "clip" not in k:
                new_state_dict[k] = state_dict[k]
        del state_dict
        return new_state_dict

    def _encode_image_tokens(self, image: torch.Tensor, drop_cls_token: bool = True, normalize: bool = True) -> torch.Tensor:
        """
        Return CLIP vision token sequence after transformer blocks (no pooling).
        We still apply the pretrained projection matrix token-wise to keep the
        token embedding dim aligned with the CLIP output space.
        """
        visual = self.clip.visual
        x = visual.conv1(image)  # (B, C, Gh, Gw)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, N, C)

        x = torch.cat([
            visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1),
            x,
        ], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.patch_dropout(x)
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = visual.ln_post(x)

        # CLS token is optimized for the original global objective.
        # For register-token aggregation, patch-level tokens are preferred.
        if drop_cls_token:
            x = x[:, 1:, :]

        if visual.proj is not None:
            x = x @ visual.proj

        if normalize:
            x = F.normalize(x, dim=-1)
        return x

    def _encode_tactile_tokens(self, tactile: torch.Tensor, drop_cls_token: bool = True, normalize: bool = True) -> torch.Tensor:
        """
        Return tactile ViT token sequence after transformer blocks (no pooling).
        Keep the learned head projection, but apply it token-wise.
        """
        x = self.tactile_encoder.forward_features(tactile)  # (B, N(+1), D_backbone)

        if drop_cls_token:
            num_prefix = getattr(self.tactile_encoder, "num_prefix_tokens", 1)
            x = x[:, num_prefix:, :]

        # Preserve pretrained post-transformer mappings while avoiding pooling.
        if hasattr(self.tactile_encoder, "fc_norm"):
            x = self.tactile_encoder.fc_norm(x)
        if hasattr(self.tactile_encoder, "head_drop"):
            x = self.tactile_encoder.head_drop(x)
        if hasattr(self.tactile_encoder, "head") and isinstance(self.tactile_encoder.head, nn.Linear):
            x = self.tactile_encoder.head(x)

        if normalize:
            x = F.normalize(x, dim=-1)
        return x

    def forward(self, input_dict : dict, return_token_sequences: bool = False, drop_cls_token: bool = True):
        # dictionary should have keys: vision, tactile, text
        # vision: (batch, 3, 224, 224)
        # tactile: (batch, 3, 224, 224)
        # text: (batch, 77)
        out_dict = {}
        if ModalityType.VISION in input_dict.keys():
            with torch.no_grad():
                if return_token_sequences:
                    vision_features = self._encode_image_tokens(
                        input_dict[ModalityType.VISION],
                        drop_cls_token=drop_cls_token,
                        normalize=True,
                    )
                else:
                    vision_features = self.clip.encode_image(input_dict[ModalityType.VISION], normalize=True)
            out_dict[ModalityType.VISION] = vision_features
        if ModalityType.TACTILE in input_dict.keys():
            if return_token_sequences:
                tactile_features = self._encode_tactile_tokens(
                    input_dict[ModalityType.TACTILE],
                    drop_cls_token=drop_cls_token,
                    normalize=True,
                )
            else:
                tactile_features = self.tactile_encoder(input_dict[ModalityType.TACTILE])
                # normalize tactile_features
                tactile_features = tactile_features / torch.norm(tactile_features, dim=1, keepdim=True)
            out_dict[ModalityType.TACTILE] = tactile_features
        if ModalityType.TEXT in input_dict.keys():
            with torch.no_grad():
                text_features = self.clip.encode_text(input_dict[ModalityType.TEXT], normalize=True)
            out_dict[ModalityType.TEXT] = text_features
        out_dict["logit_scale"] = self.logit_scale.exp()
        if self.logit_bias is not None:
            out_dict["logit_bias"] = self.logit_bias
        return out_dict