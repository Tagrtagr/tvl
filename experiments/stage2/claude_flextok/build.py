"""
Builder functions for Stage 2 training: datasets, pipeline, losses, decoders.
All builders accept OmegaConf DictConfig directly and use hydra.utils.instantiate
on config nodes — no intermediate dict conversion needed.
"""

import os
import sys
from typing import Dict, Optional, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, Dataset

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "../../.."))
sys.path.insert(0, os.path.join(_here, "../../../tvl_enc"))

from tvl_enc.tvl import ModalityType
from tvl_enc.tacvis import TacVisDataset, TacVisDatasetV2, RGB_AUGMENTS, TAC_AUGMENTS, TAC_AUGMENTS_BG
from pipeline.stage2_pipeline import Stage2Pipeline, DecoderCollection, build_modality_encoder


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _sample_group_key_from_path(path: str) -> str:
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)
    if len(parts) >= 2 and parts[-2] in {"images_rgb", "images_tac", "images_tactile", "rgb", "tac"}:
        return parts[-3] if len(parts) >= 3 else os.path.dirname(norm)
    return os.path.basename(os.path.dirname(norm)) or os.path.dirname(norm)


def _filter_dataset_single_sample(dataset, selector=None, max_samples=1):
    """In-place filter dataset paths to a tiny debug subset."""
    if not hasattr(dataset, "paths"):
        raise TypeError("Dataset does not expose .paths; cannot apply debug filter")

    keep_n = max(0, int(max_samples or 1))
    paths = dataset.paths

    if isinstance(paths, dict):
        vision_paths = paths.get("vision", [])
        if not vision_paths:
            raise ValueError("Dataset has no vision paths to infer sample")
        start = 0 if selector is None else next(
            (i for i, p in enumerate(vision_paths)
             if selector in _sample_group_key_from_path(p) or selector in p), None,
        )
        if start is None:
            raise ValueError(f"No samples matched debug_sample_selector='{selector}'")
        keep = list(range(start, min(start + keep_n, len(vision_paths))))
        n_before = len(vision_paths)
        dataset.paths = {k: [v[i] for i in keep] for k, v in paths.items()}
        return vision_paths[start], n_before, len(dataset.paths["vision"])

    if isinstance(paths, list):
        if not paths:
            raise ValueError("Dataset has no paths to infer sample")
        start = 0 if selector is None else next(
            (i for i, p in enumerate(paths)
             if selector in _sample_group_key_from_path(p) or selector in p), None,
        )
        if start is None:
            raise ValueError(f"No samples matched debug_sample_selector='{selector}'")
        keep = list(range(start, min(start + keep_n, len(paths))))
        n_before = len(paths)
        dataset.paths = [paths[i] for i in keep]
        return paths[start], n_before, len(dataset.paths)

    raise TypeError(f"Unsupported dataset.paths type: {type(paths)}")


class DebugRepeatCachedDataset(Dataset):
    """Repeat a tiny cached dataset to avoid disk IO in debug mode."""

    def __init__(self, dataset: Dataset, repeat_size: int = 4096):
        self.repeat_size = max(1, int(repeat_size))
        self.cache = [dataset[i] for i in range(len(dataset))]
        if not self.cache:
            raise ValueError("DebugRepeatCachedDataset received an empty dataset")

    def __len__(self):
        return self.repeat_size

    def __getitem__(self, index):
        return self.cache[index % len(self.cache)]


def build_datasets(cfg: DictConfig) -> Tuple[Dataset, Dataset]:
    """Build training and validation datasets."""
    modality_types = [ModalityType.VISION, ModalityType.TACTILE]
    prompt = "This image gives tactile feelings of "
    dataset_train, dataset_val = [], []

    if "ssvtp" in cfg.datasets:
        tac_aug = TAC_AUGMENTS_BG if cfg.subtract_background == "background" else TAC_AUGMENTS
        root = os.path.join(cfg.datasets_dir, "ssvtp")
        dataset_train.append(TacVisDataset(
            root_dir=root, split="train", transform_rgb=RGB_AUGMENTS, transform_tac=tac_aug,
            modality_types=modality_types, text_prompt=prompt,
        ))
        dataset_val.append(TacVisDataset(
            root_dir=root, split="val", transform_rgb=RGB_AUGMENTS, transform_tac=tac_aug,
            modality_types=modality_types, text_prompt=prompt,
        ))

    if "hct" in cfg.datasets:
        hct_dir = os.path.join(cfg.datasets_dir, "hct")
        dirs = [
            os.path.join(hct_dir, d) for d in os.listdir(hct_dir)
            if os.path.isdir(os.path.join(hct_dir, d))
            and os.path.exists(os.path.join(hct_dir, d, "contact.json"))
        ] if os.path.exists(hct_dir) else []
        if dirs:
            tac_aug = TAC_AUGMENTS if cfg.subtract_background is None else None
            dataset_train.append(TacVisDatasetV2(
                root_dir=dirs, split="train", transform_rgb=RGB_AUGMENTS, transform_tac=tac_aug,
                modality_types=modality_types, text_prompt=prompt,
            ))
            dataset_val.append(TacVisDatasetV2(
                root_dir=dirs, split="val", transform_rgb=RGB_AUGMENTS, transform_tac=tac_aug,
                modality_types=modality_types, text_prompt=prompt,
            ))

    if not dataset_train:
        raise ValueError(f"No datasets found in {cfg.datasets_dir}")
    train_ds = dataset_train[0] if len(dataset_train) == 1 else ConcatDataset(dataset_train)
    val_ds = dataset_val[0] if len(dataset_val) == 1 else ConcatDataset(dataset_val)

    if cfg.debug_single_sample:
        def _parts(ds):
            return ds.datasets if isinstance(ds, ConcatDataset) else [ds]
        for tag, ds in [("train", train_ds), ("val", val_ds)]:
            for i, comp in enumerate(_parts(ds)):
                sel, n0, n1 = _filter_dataset_single_sample(
                    comp, cfg.debug_sample_selector, cfg.debug_max_train_samples,
                )
                print(f"[debug][{tag}][{i}] {sel!r}: {n0} -> {n1}")
        if isinstance(train_ds, ConcatDataset):
            train_ds = ConcatDataset(list(_parts(train_ds)))
        if isinstance(val_ds, ConcatDataset):
            val_ds = ConcatDataset(list(_parts(val_ds)))
        if cfg.debug_dataset_mode == "repeat_cached":
            t0, v0 = len(train_ds), len(val_ds)
            train_ds = DebugRepeatCachedDataset(train_ds, cfg.debug_repeat_size)
            val_ds = DebugRepeatCachedDataset(val_ds, cfg.debug_repeat_size)
            print(f"[debug] repeat_cached: train {t0}->{len(train_ds)}, val {v0}->{len(val_ds)}")

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_modality_configs(modality_encoder) -> dict:
    if hasattr(modality_encoder, "hidden_dim"):
        dim = modality_encoder.hidden_dim
        return {mod: {"input_dim": dim, "feature_type": "sequence"}
                for mod in [ModalityType.VISION, ModalityType.TACTILE]}
    vision_dim = getattr(modality_encoder.clip.visual, "output_dim", 768)
    tactile_dim = (
        modality_encoder.tactile_encoder.num_classes
        if modality_encoder.tactile_encoder.num_classes > 0
        else modality_encoder.tactile_encoder.num_features
    )
    return {
        ModalityType.VISION:  {"input_dim": vision_dim,  "feature_type": "sequence"},
        ModalityType.TACTILE: {"input_dim": tactile_dim, "feature_type": "sequence"},
    }


def build_pipeline(cfg: DictConfig, device) -> Stage2Pipeline:
    """Construct the Stage2Pipeline (frozen encoder + cross-modal alignment model)."""
    stage1_ckpt = cfg.stage1_checkpoint if cfg.stage1_checkpoint and os.path.exists(cfg.stage1_checkpoint) else None
    vae_ckpt = cfg.vae_checkpoint if cfg.vae_checkpoint and os.path.exists(cfg.vae_checkpoint) else None

    modality_encoder = build_modality_encoder(
        feature_backbone="vae" if cfg.feature_backbone == "vae" else "tvl",
        tactile_model=cfg.tactile_model,
        stage1_checkpoint=stage1_ckpt,
        vae_checkpoint=vae_ckpt,
        vae_latent_channels=cfg.vae_latent_channels,
        vae_scaling_factor=cfg.vae_scaling_factor,
    )

    cross_modality_encoder = hydra.utils.instantiate(
        cfg.model.register_cross_modal,
        modality_configs=_build_modality_configs(modality_encoder),
        _convert_="all",
    )
    tokenizer = hydra.utils.instantiate(cfg.pipeline.tokenizer, _convert_="all")

    return Stage2Pipeline(
        modality_encoder=modality_encoder,
        cross_modality_encoder=cross_modality_encoder,
        tokenizer=tokenizer,
        decoder_collection=None,
    ).to(device)


def build_decoders(cfg: DictConfig, device, alignment_hidden_dim: int) -> Tuple[Dict[str, nn.Module], Optional[nn.Module]]:
    """Build stage-appropriate decoders and their loss.

    alignment stage  → autoregressive probe decoder (cfg.model.probe_decoder)
    reconstruction   → autoregressive recon decoder (cfg.model.recon_decoder)
    """
    stage = cfg.stage
    modalities = [ModalityType.VISION, ModalityType.TACTILE]

    if stage == "alignment":
        if OmegaConf.select(cfg, "model.probe_decoder") is None:
            return {}, None
        decoders = {
            mod: hydra.utils.instantiate(
                cfg.model.probe_decoder[mod],
                hidden_dim=alignment_hidden_dim, _convert_="all",
            ).to(device)
            for mod in modalities
        }
        loss_fn = hydra.utils.instantiate(cfg.losses.probe.module, _convert_="all")
        label = "ar_probe"

    elif stage == "reconstruction":
        if OmegaConf.select(cfg, "model.recon_decoder") is None:
            return {}, None
        decoders = {
            mod: hydra.utils.instantiate(
                cfg.model.recon_decoder[mod], hidden_dim=alignment_hidden_dim, _convert_="all",
            ).to(device)
            for mod in modalities
        }
        loss_fn = hydra.utils.instantiate(cfg.losses.recon_decoder.pixel.module, _convert_="all")
        label = "recon_decoder"

    else:
        return {}, None

    n = sum(sum(p.numel() for p in dec.parameters()) for dec in decoders.values())
    print(f"Decoders ({label}): {n:,} parameters")
    return decoders, loss_fn


def build_losses(cfg: DictConfig, device, embed_dim: int) -> Optional[nn.Module]:
    """Build the contrastive alignment loss (alignment stage only)."""
    if OmegaConf.select(cfg, "losses.contrastive") is None:
        return None
    modality_names = [ModalityType.VISION, ModalityType.TACTILE]
    module_cfg = OmegaConf.to_container(cfg.losses.contrastive.module, resolve=True)
    module_cfg["latent_dim"] = int(embed_dim)
    return hydra.utils.instantiate(
        module_cfg, modality_names=modality_names, _convert_="all",
    ).to(device)
