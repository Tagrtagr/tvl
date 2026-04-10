"""
Stage 2 Training Script: Cross-Modality Alignment with Register Tokens.
Approach: FlexTok-inspired register tokens (Claude Code)

Supports two-stage training following FlexTok (Bachmann et al., 2025):
  Stage 2a ("alignment"): Train register tokens with contrastive + preservation loss.
  Stage 2b ("reconstruction"): Freeze register tokens, train autoregressive decoder only.
  "joint" mode trains both together (legacy behavior).

Usage:
    # Stage 2a: Train alignment (register tokens only, no decoder)
    python experiments/stage2/claude_flextok/train.py \
        --stage alignment \
        --stage1_checkpoint /path/to/stage1.pth \
        --datasets_dir /path/to/datasets \
        --output_dir ./output/stage2a

    # Stage 2b: Train reconstruction decoder (frozen alignment model)
    python experiments/stage2/claude_flextok/train.py \
        --stage reconstruction \
        --alignment_checkpoint ./output/stage2a/checkpoint_best.pth \
        --stage1_checkpoint /path/to/stage1.pth \
        --datasets_dir /path/to/datasets \
        --decoder_type autoregressive \
        --output_dir ./output/stage2b

    # Joint training (legacy, both together)
    python experiments/stage2/claude_flextok/train.py \
        --stage1_checkpoint /path/to/stage1.pth \
        --datasets_dir /path/to/datasets \
        --use_reconstruction --decoder_type autoregressive \
        --output_dir ./output/stage2_joint

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 experiments/stage2/claude_flextok/train.py \
        --stage alignment \
        --stage1_checkpoint /path/to/stage1.pth \
        --datasets_dir /path/to/datasets
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from types import SimpleNamespace

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset
from torchvision.utils import save_image
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


# Add parent directories to path for imports
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)  # for local models/ and losses/ imports
sys.path.insert(0, os.path.join(_here, "../../.."))  # project root
sys.path.insert(0, os.path.join(_here, "../../../tvl_enc"))  # tvl_enc utilities

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched

from tvl_enc.tvl import TVL, ModalityType
from tvl_enc.tacvis import (
    TacVisDataset, TacVisDatasetV2,
    RGB_AUGMENTS, TAC_AUGMENTS, TAC_AUGMENTS_BG, TAC_AUGMENTS_BG_CJ,
    RGB_MEAN, RGB_STD, TAC_MEAN, TAC_STD, TAC_MEAN_BG, TAC_STD_BG,
)

from models.cross_modal_alignment import CrossModalAlignmentModel, Stage2Wrapper
from models.reconstruction_decoder import ReconstructionDecoder
from models.autoregressive_decoder import AutoregressiveDecoder
from models.flow_matching_decoder import FlowMatchingReconstructionDecoder
from models.simple_vae import VAEFeatureBackbone
from losses.alignment_loss import CrossModalAlignmentLoss
from losses.flow_matching import FlowMatchingAlignmentLoss
from losses.reconstruction_loss import ReconstructionLoss
from losses.flow_reconstruction_loss import FlowReconstructionLoss

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def get_args_parser():
    parser = argparse.ArgumentParser("Stage 2: Cross-Modality Alignment", add_help=False)

    # Model
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_registers", type=int, default=32)
    parser.add_argument("--n_shared", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--model_dropout", type=float, default=0.1)
    parser.add_argument("--nested_dropout", action="store_true", default=True)
    parser.add_argument("--no_nested_dropout", action="store_false", dest="nested_dropout")
    parser.add_argument("--nested_dropout_mode", type=str, default="power_of_two",
                        choices=["power_of_two", "uniform"])

    # Loss
    parser.add_argument("--loss_type", type=str, default="contrastive",
                        choices=["contrastive", "flow_matching", "combined"])
    parser.add_argument("--use_flow_matching_alignment_loss", action="store_true", default=False, help="Enable flow-matching alignment loss (optional)")
    parser.add_argument("--contrastive_weight", type=float, default=1.0)
    parser.add_argument("--preservation_weight", type=float, default=0.5)
    parser.add_argument("--flow_weight", type=float, default=0.5)

    # Reconstruction
    parser.add_argument("--reconstruction_weight", type=float, default=1.0)
    parser.add_argument("--recon_base_channels", type=int, default=64)
    parser.add_argument("--recon_decoder_layers", type=int, default=2)
    parser.add_argument("--recon_loss_type", type=str, default="mse",
                        choices=["mse", "l1", "smooth_l1"])
    parser.add_argument("--save_recon_images", action="store_true", default=True,
                        help="Save reconstructed images during evaluation")
    parser.add_argument("--recon_save_every", type=int, default=1,
                        help="Save reconstructions every N eval epochs")
    parser.add_argument("--recon_save_max_batches", type=int, default=1,
                        help="Max number of validation batches to save per eval")
    parser.add_argument("--flow_recon_steps", type=int, default=20,
                        help="Euler steps for flow-matching reconstruction sampling")
    parser.add_argument("--flow_recon_pixel_mse_weight", type=float, default=1.0,
                        help="Image-space MSE weight for flow reconstruction loss")

    # OAT-style prefix reconstruction (coarse-to-fine ordering)
    parser.add_argument("--use_prefix_recon", action="store_true", default=False,
                        help="Enable OAT-style prefix reconstruction loss")
    parser.add_argument("--prefix_recon_weight", type=float, default=0.5,
                        help="Weight for prefix reconstruction relative to full recon")

    # Token type embeddings (OAT-inspired)
    parser.add_argument("--use_token_type_embed", action="store_true", default=True,
                        help="Enable OAT-style token type embeddings (shared vs private)")
    parser.add_argument("--no_token_type_embed", action="store_false", dest="use_token_type_embed")

    # Training stage (FlexTok two-stage split)
    parser.add_argument("--stage", type=str, default="joint",
                        choices=["alignment", "reconstruction", "joint"],
                        help="Training stage: 'alignment' trains register tokens only, "
                             "'reconstruction' freezes alignment and trains decoder only, "
                             "'joint' trains both together (legacy)")
    parser.add_argument("--alignment_checkpoint", type=str, default=None,
                        help="Path to alignment stage checkpoint (for --stage reconstruction)")
    parser.add_argument("--decoder_type", type=str, default="autoregressive",
                        choices=["conv", "autoregressive", "flow_matching"],
                        help="Reconstruction decoder: 'autoregressive' (transformer) or 'conv' (ConvTranspose) "
                             "or 'flow_matching' (rectified-flow conditioned on register tokens)")

    # Training
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--seed", default=42, type=int)

    # Stage 1
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Path to Stage 1 TVL encoder checkpoint")
    parser.add_argument("--tactile_model", type=str, default="vit_tiny_patch16_224",
                        choices=["vit_base_patch16_224", "vit_small_patch16_224", "vit_tiny_patch16_224"])

    # Frozen feature extractor choice for Stage 2
    parser.add_argument("--feature_backbone", type=str, default="encoders",
                        choices=["encoders", "vae"],
                        help="Use pretrained stage-1 encoders (default) or pretrained VAE encoder tokens")
    parser.add_argument("--vae_checkpoint", type=str, default=None,
                        help="Path to pretrained VAE checkpoint (supports regular .pth and .safetensors when available)")
    parser.add_argument("--vae_latent_channels", type=int, default=4,
                        help="Latent channel count used by the pretrained VAE")
    parser.add_argument("--vae_scaling_factor", type=float, default=0.18215,
                        help="Latent scaling factor used by FlexTok/diffusers-style VAE")

    # Data
    parser.add_argument("--datasets_dir", type=str, default="./.datasets")
    parser.add_argument("--datasets", type=str, nargs="+", default=["ssvtp", "hct"],
                        choices=["ssvtp", "hct"])
    parser.add_argument("--subtract_background", type=str, default=None,
                        choices=[None, "mean", "median", "background"])
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true", default=True)

    # Output
    parser.add_argument("--output_dir", default="./output/stage2", type=str)
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--log_name", default=None, type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--start_epoch", default=0, type=int)

    # Config file (overrides command line)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")

    # Distributed
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--dist_eval", action="store_true", default=False)

    return parser


def _sample_group_key_from_path(path: str) -> str:
    """Best-effort sample-group key extracted from sample path.

    For TacVis-style layouts, this is usually the parent sequence/folder.
    """
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)

    # Common pattern: .../<traj_dir>/images_rgb/<image_file>
    if len(parts) >= 2 and parts[-2] in {"images_rgb", "images_tac", "images_tactile", "rgb", "tac"}:
        return parts[-3] if len(parts) >= 3 else os.path.dirname(norm)

    # Generic: use immediate parent directory as sample-group key
    return os.path.basename(os.path.dirname(norm)) or os.path.dirname(norm)


def _filter_dataset_single_sample(dataset, selector: Optional[str] = None, max_samples: Optional[int] = 1):
    """In-place filter for TacVisDataset/TacVisDatasetV2 to a tiny sample subset.

    Behavior is sample-level (paired image+tactile), not sequence-level:
    - selector=None: keep from first sample onward
    - selector=str : keep from first sample whose key/path contains selector
    - max_samples: number of paired samples to keep (default 1)

    Returns: (selected_sample, n_before, n_after)
    """
    if not hasattr(dataset, "paths"):
        raise TypeError("Dataset does not expose .paths; cannot apply debug single-sample filter")

    keep_n = 1 if max_samples is None else max(0, int(max_samples))
    paths = dataset.paths

    # TacVisDatasetV2 stores dict of modality lists
    if isinstance(paths, dict):
        vision_paths = paths.get("vision", [])
        if not vision_paths:
            raise ValueError("Dataset has no vision paths to infer sample")

        if selector is None:
            start_idx = 0
        else:
            start_idx = None
            for i, p in enumerate(vision_paths):
                key = _sample_group_key_from_path(p)
                if (selector in key) or (selector in p):
                    start_idx = i
                    break
            if start_idx is None:
                raise ValueError(f"No samples matched debug_sample_selector='{selector}'")

        keep_idx = list(range(start_idx, min(start_idx + keep_n, len(vision_paths))))
        selected_sample = vision_paths[start_idx]
        n_before = len(vision_paths)
        dataset.paths = {k: [v[i] for i in keep_idx] for k, v in paths.items()}
        n_after = len(dataset.paths["vision"])
        return selected_sample, n_before, n_after

    # TacVisDataset stores list of rgb paths
    if isinstance(paths, list):
        if not paths:
            raise ValueError("Dataset has no paths to infer sample")

        if selector is None:
            start_idx = 0
        else:
            start_idx = None
            for i, p in enumerate(paths):
                key = _sample_group_key_from_path(p)
                if (selector in key) or (selector in p):
                    start_idx = i
                    break
            if start_idx is None:
                raise ValueError(f"No samples matched debug_sample_selector='{selector}'")

        keep_idx = list(range(start_idx, min(start_idx + keep_n, len(paths))))
        selected_sample = paths[start_idx]
        n_before = len(paths)
        dataset.paths = [paths[i] for i in keep_idx]
        n_after = len(dataset.paths)
        return selected_sample, n_before, n_after

    raise TypeError(f"Unsupported dataset.paths type: {type(paths)}")


class DebugRepeatCachedDataset(Dataset):
    """Repeat a tiny dataset from RAM cache to avoid per-step disk IO in debug."""

    def __init__(self, dataset: Dataset, repeat_size: int = 4096):
        self.dataset = dataset
        self.repeat_size = max(1, int(repeat_size))
        self.cache = [dataset[i] for i in range(len(dataset))]
        if len(self.cache) == 0:
            raise ValueError("DebugRepeatCachedDataset received an empty dataset")

    def __len__(self):
        return self.repeat_size

    def __getitem__(self, index):
        return self.cache[index % len(self.cache)]


def build_datasets(args):
    """Build training and validation datasets (reuses Stage 1 data pipeline)."""
    modality_types = [ModalityType.VISION, ModalityType.TACTILE]
    prompt = "This image gives tactile feelings of "

    dataset_train = []
    dataset_val = []

    if "ssvtp" in args.datasets:
        tac_augments = TAC_AUGMENTS
        if args.subtract_background == "background":
            tac_augments = TAC_AUGMENTS_BG

        root_dir = os.path.join(args.datasets_dir, "ssvtp")
        dataset_train.append(TacVisDataset(
            root_dir=root_dir, split="train",
            transform_rgb=RGB_AUGMENTS, transform_tac=tac_augments,
            modality_types=modality_types, text_prompt=prompt,
        ))
        dataset_val.append(TacVisDataset(
            root_dir=root_dir, split="val",
            transform_rgb=RGB_AUGMENTS, transform_tac=tac_augments,
            modality_types=modality_types, text_prompt=prompt,
        ))

    if "hct" in args.datasets:
        dataset_dirs = []
        hct_dir = os.path.join(args.datasets_dir, "hct")
        if os.path.exists(hct_dir):
            for i in os.listdir(hct_dir):
                sub_dir = os.path.join(hct_dir, i)
                if os.path.isdir(sub_dir) and os.path.exists(os.path.join(sub_dir, "contact.json")):
                    dataset_dirs.append(sub_dir)

        if dataset_dirs:
            tac_augments = TAC_AUGMENTS if args.subtract_background is None else None
            dataset_train.append(TacVisDatasetV2(
                root_dir=dataset_dirs, split="train",
                transform_rgb=RGB_AUGMENTS, transform_tac=tac_augments,
                modality_types=modality_types, text_prompt=prompt,
            ))
            dataset_val.append(TacVisDatasetV2(
                root_dir=dataset_dirs, split="val",
                transform_rgb=RGB_AUGMENTS, transform_tac=tac_augments,
                modality_types=modality_types, text_prompt=prompt,
            ))

    if len(dataset_train) == 1:
        train_ds, val_ds = dataset_train[0], dataset_val[0]
    elif len(dataset_train) > 1:
        train_ds, val_ds = ConcatDataset(dataset_train), ConcatDataset(dataset_val)
    else:
        raise ValueError(f"No datasets found in {args.datasets_dir}")

    # Debug mode: constrain dataset to a tiny number of paired samples.
    # Applied before DataLoader creation for deterministic one-sample overfit tests.
    if getattr(args, "debug_single_sample", False):
        selector = getattr(args, "debug_sample_selector", None)
        max_train_samples = getattr(args, "debug_max_train_samples", 1)
        debug_dataset_mode = getattr(args, "debug_dataset_mode", "filter")
        debug_repeat_size = getattr(args, "debug_repeat_size", 4096)

        def _iter_components(ds):
            if isinstance(ds, ConcatDataset):
                return ds.datasets
            return [ds]

        for i, ds_comp in enumerate(_iter_components(train_ds)):
            selected_sample, n0, n1 = _filter_dataset_single_sample(
                ds_comp,
                selector=selector,
                max_samples=max_train_samples,
            )
            print(f"[debug_single_sample][train][{i}] sample='{selected_sample}' {n0} -> {n1} samples")

        for i, ds_comp in enumerate(_iter_components(val_ds)):
            selected_sample, n0, n1 = _filter_dataset_single_sample(
                ds_comp,
                selector=selector,
                max_samples=max_train_samples,
            )
            print(f"[debug_single_sample][val][{i}] sample='{selected_sample}' {n0} -> {n1} samples")

        if debug_dataset_mode == "repeat_cached":
            train_before = len(train_ds)
            val_before = len(val_ds)
            train_ds = DebugRepeatCachedDataset(train_ds, repeat_size=debug_repeat_size)
            val_ds = DebugRepeatCachedDataset(val_ds, repeat_size=debug_repeat_size)
            print(
                f"[debug_single_sample] dataset_mode=repeat_cached "
                f"train {train_before} -> {len(train_ds)}, val {val_before} -> {len(val_ds)}"
            )
        else:
            print(f"[debug_single_sample] dataset_mode={debug_dataset_mode}")

    return train_ds, val_ds


def build_model(args, device):
    """Build frozen backbone + Stage 2 alignment model."""
    if getattr(args, "feature_backbone", "encoders") == "vae":
        # Build frozen VAE feature backbone
        frozen_encoder = VAEFeatureBackbone(
            hidden_dim=768,
            latent_channels=args.vae_latent_channels,
            scaling_factor=args.vae_scaling_factor,
        )
        if not args.vae_checkpoint or not os.path.exists(args.vae_checkpoint):
            raise FileNotFoundError(
                "--feature_backbone vae requires a valid --vae_checkpoint"
            )
        print(f"Loading pretrained VAE checkpoint from {args.vae_checkpoint}")
        missing, unexpected = frozen_encoder.load_pretrained_checkpoint(
            args.vae_checkpoint,
            strict=False,
        )
        if missing:
            print(f"[warn] VAE missing keys: {len(missing)}")
        if unexpected:
            print(f"[warn] VAE unexpected keys: {len(unexpected)}")

        modality_configs = {
            ModalityType.VISION: {"input_dim": frozen_encoder.hidden_dim, "feature_type": "sequence"},
            ModalityType.TACTILE: {"input_dim": frozen_encoder.hidden_dim, "feature_type": "sequence"},
        }
    else:
        # Build frozen Stage 1 encoders
        frozen_encoder = TVL(
            tactile_model=args.tactile_model,
            active_modalities=[ModalityType.VISION, ModalityType.TACTILE],
        )

        # Load Stage 1 checkpoint if provided
        if args.stage1_checkpoint and os.path.exists(args.stage1_checkpoint):
            print(f"Loading Stage 1 checkpoint from {args.stage1_checkpoint}")
            ckpt = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
            if "model" in ckpt:
                frozen_encoder.load_state_dict(ckpt["model"], strict=False)
            else:
                frozen_encoder.load_state_dict(ckpt, strict=False)

        # Determine input dims from the frozen encoder
        # Vision: OpenCLIP ViT-L-14 outputs 768-dim
        # Tactile: depends on tactile_model + whether num_classes matches CLIP width
        vision_dim = frozen_encoder.clip.visual.output_dim if hasattr(frozen_encoder.clip.visual, "output_dim") else 768
        tactile_dim = frozen_encoder.tactile_encoder.num_classes if frozen_encoder.tactile_encoder.num_classes > 0 else frozen_encoder.tactile_encoder.num_features

        modality_configs = {
            ModalityType.VISION: {"input_dim": vision_dim, "feature_type": "sequence"},
            ModalityType.TACTILE: {"input_dim": tactile_dim, "feature_type": "sequence"},
        }

    # Build Stage 2 alignment model
    use_token_type = getattr(args, "use_token_type_embed", True)
    alignment_model = CrossModalAlignmentModel(
        modality_configs=modality_configs,
        n_registers=args.n_registers,
        n_shared=args.n_shared,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.model_dropout,
        nested_dropout=args.nested_dropout,
        nested_dropout_mode=args.nested_dropout_mode,
        use_token_type_embed=use_token_type,
    )

    # Wrap into Stage 2 pipeline
    model = Stage2Wrapper(frozen_encoder, alignment_model)
    model.to(device)

    return model


def build_loss(args, device, embed_dim):
    """Build loss function(s) based on configuration."""
    modality_names = [ModalityType.VISION, ModalityType.TACTILE]

    contrastive_loss = CrossModalAlignmentLoss(
        modality_names=modality_names,
        contrastive_weight=args.contrastive_weight,
        preservation_weight=args.preservation_weight,
    ).to(device)

    flow_loss = None
    if getattr(args, "use_flow_matching_alignment_loss", False) and args.loss_type in ["flow_matching", "combined"]:
        flow_loss = FlowMatchingAlignmentLoss(
            modality_names=modality_names,
            embed_dim=embed_dim,
        ).to(device)

    return contrastive_loss, flow_loss


def _unwrap(model):
    """Unwrap DDP model to access inner attributes."""
    return model.module if hasattr(model, "module") else model


def _unnormalize_for_display(x: torch.Tensor, mod_name: str, args) -> torch.Tensor:
    """Map normalized tensors back to [0,1]-like pixel space for visualization/flow targets."""
    if mod_name == ModalityType.VISION:
        mean = torch.tensor(RGB_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(RGB_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    elif mod_name == ModalityType.TACTILE:
        if getattr(args, "subtract_background", None) == "background":
            mean_np = TAC_MEAN_BG
            std_np = TAC_STD_BG
        else:
            mean_np = TAC_MEAN
            std_np = TAC_STD
        mean = torch.tensor(mean_np, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(std_np, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    else:
        return x
    return x * std + mean


def _save_recon_images(
    batch_idx,
    alignment_output,
    batch,
    recon_decoders,
    output_dir,
    epoch,
    stage,
    args,
):
    if recon_decoders is None or output_dir is None:
        return
    if not getattr(args, "save_recon_images", False):
        return
    if epoch is None:
        return
    if args.recon_save_every <= 0:
        return
    if epoch % args.recon_save_every != 0:
        return
    if batch_idx >= args.recon_save_max_batches:
        return

    recon_root = Path(output_dir) / "reconstructions" / stage / f"epoch_{epoch:04d}" / f"batch_{batch_idx:03d}"
    recon_root.mkdir(parents=True, exist_ok=True)

    for mod_name in [ModalityType.VISION, ModalityType.TACTILE]:
        all_tokens_key = f"{mod_name}_all_tokens"
        if all_tokens_key not in alignment_output or mod_name not in batch:
            continue
        tokens = alignment_output[all_tokens_key]
        targets = batch[mod_name]
        decoder = recon_decoders.get(mod_name)
        if decoder is None:
            continue

        if isinstance(_unwrap(decoder), FlowMatchingReconstructionDecoder):
            recons = _unwrap(decoder).sample(tokens, n_steps=args.flow_recon_steps)
        else:
            recons = decoder(tokens)

        targets = torch.clamp(_unnormalize_for_display(targets, mod_name, args), 0.0, 1.0)
        recons = torch.clamp(_unnormalize_for_display(recons, mod_name, args), 0.0, 1.0)

        save_image(targets, recon_root / f"{mod_name}_target.png", nrow=8)
        save_image(recons, recon_root / f"{mod_name}_recon.png", nrow=8)


def train_one_epoch(
    model, contrastive_loss_fn, flow_loss_fn, data_loader,
    optimizer, device, epoch, loss_scaler, args, log_writer=None,
    recon_decoders=None, recon_loss_fn=None,
):
    model.train()
    # Keep frozen encoder in eval mode
    model_inner = _unwrap(model)
    model_inner.frozen_encoder.eval()
    if recon_decoders is not None:
        for dec in recon_decoders.values():
            dec.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 20
    accum_iter = args.accum_iter

    optimizer.zero_grad()
    
    # import ipdb; ipdb.set_trace()
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Per-iteration LR schedule
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Move data to device
        for k, v in samples.items():
            if isinstance(v, list):
                v = v[0]
            samples[k] = v.to(device, non_blocking=True)

        stage = getattr(args, "stage", "joint")

        with torch.cuda.amp.autocast():
            # Get frozen encoder features
            with torch.no_grad():
                frozen_features = model_inner.encode(samples)

            # Forward through alignment model
            # In reconstruction stage, alignment is frozen — skip graph building
            if stage == "reconstruction":
                with torch.no_grad():
                    alignment_output = model_inner.alignment_model(frozen_features)
            else:
                alignment_output = model_inner.alignment_model(frozen_features)

            loss = torch.tensor(0.0, device=device)
            loss_dict = {}

            # Contrastive + preservation loss (alignment and joint stages)
            if stage != "reconstruction":
                preservation_projectors = model_inner.alignment_model.preservation_projectors
                align_dict = contrastive_loss_fn(
                    alignment_output,
                    frozen_features=frozen_features,
                    preservation_projectors=preservation_projectors,
                    input_images=samples,
                    output_dict=True,
                )
                loss = align_dict["total_loss"]
                loss_dict.update(align_dict)

                # Flow matching loss (if enabled)
                if flow_loss_fn is not None:
                    flow_dict = flow_loss_fn(alignment_output, output_dict=True)
                    flow_loss = flow_dict.get("flow_total", torch.tensor(0.0, device=device))
                    loss = loss + args.flow_weight * flow_loss
                    loss_dict["flow_loss"] = flow_loss

            # Reconstruction loss (alignment / reconstruction / joint stages)
            if recon_decoders is not None and recon_loss_fn is not None:
                targets = {}
                all_tokens = {}
                prefix_tokens = {}
                for mod_name in [ModalityType.VISION, ModalityType.TACTILE]:
                    all_tokens_key = f"{mod_name}_all_tokens"
                    prefix_tokens_key = f"{mod_name}_prefix_tokens"
                    if all_tokens_key in alignment_output and mod_name in samples:
                        all_tokens[mod_name] = alignment_output[all_tokens_key]
                        if prefix_tokens_key in alignment_output:
                            prefix_tokens[mod_name] = alignment_output[prefix_tokens_key]
                        if isinstance(recon_loss_fn, FlowReconstructionLoss):
                            targets[mod_name] = _unnormalize_for_display(samples[mod_name], mod_name, args)
                        else:
                            targets[mod_name] = samples[mod_name]
                if all_tokens:
                    if isinstance(recon_loss_fn, FlowReconstructionLoss):
                        recon_dict = recon_loss_fn(
                            all_tokens,
                            targets,
                            recon_decoders,
                            output_dict=True,
                            prefix_tokens=prefix_tokens,
                        )
                    else:
                        recons = {m: recon_decoders[m](all_tokens[m]) for m in all_tokens}
                        recon_dict = recon_loss_fn(recons, targets)

                    recon_loss = recon_dict["recon_total"]
                    loss = loss + args.reconstruction_weight * recon_loss
                    for k, v in recon_dict.items():
                        if isinstance(v, torch.Tensor):
                            loss_dict[k] = v

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter

        # Collect trainable parameters (filtered by requires_grad for stage support)
        trainable_params = list(model_inner.alignment_model.parameters())
        if flow_loss_fn is not None:
            trainable_params += list(flow_loss_fn.parameters())
        if recon_decoders is not None:
            for dec in recon_decoders.values():
                trainable_params += list(dec.parameters())
        trainable_params = [p for p in trainable_params if p.requires_grad]

        loss_scaler(loss, optimizer, parameters=trainable_params,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # Log metrics
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                metric_logger.update(**{k: v.item()})

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value, epoch_1000x)
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    log_writer.add_scalar(f"train_{k}", v.item(), epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    data_loader, contrastive_loss_fn, model, device, epoch=None, log_writer=None,
    recon_decoders=None, recon_loss_fn=None, args=None,
    stage="joint", output_dir=None,
):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"
    model.eval()
    model_inner = _unwrap(model)
    if recon_decoders is not None:
        for dec in recon_decoders.values():
            dec.eval()

    for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        for k, v in batch.items():
            if isinstance(v, list):
                v = v[0]
            batch[k] = v.to(device, non_blocking=True)
            batch_size = v.shape[0]

        with torch.cuda.amp.autocast():
            frozen_features = model_inner.encode(batch)
            alignment_output = model_inner.alignment_model(frozen_features)
            preservation_projectors = model_inner.alignment_model.preservation_projectors
            loss_dict = contrastive_loss_fn(
                alignment_output,
                frozen_features=frozen_features,
                preservation_projectors=preservation_projectors,
                input_images=batch,
                output_dict=True,
            )

            # Reconstruction loss (if enabled)
            if recon_decoders is not None and recon_loss_fn is not None:
                targets = {}
                all_tokens = {}
                prefix_tokens = {}
                for mod_name in [ModalityType.VISION, ModalityType.TACTILE]:
                    all_tokens_key = f"{mod_name}_all_tokens"
                    prefix_tokens_key = f"{mod_name}_prefix_tokens"
                    if all_tokens_key in alignment_output and mod_name in batch:
                        all_tokens[mod_name] = alignment_output[all_tokens_key]
                        if prefix_tokens_key in alignment_output:
                            prefix_tokens[mod_name] = alignment_output[prefix_tokens_key]
                        if isinstance(recon_loss_fn, FlowReconstructionLoss):
                            targets[mod_name] = _unnormalize_for_display(batch[mod_name], mod_name, args)
                        else:
                            targets[mod_name] = batch[mod_name]
                if all_tokens:
                    if isinstance(recon_loss_fn, FlowReconstructionLoss):
                        recon_dict = recon_loss_fn(
                            all_tokens,
                            targets,
                            recon_decoders,
                            output_dict=True,
                            prefix_tokens=prefix_tokens,
                        )
                    else:
                        recons = {m: recon_decoders[m](all_tokens[m]) for m in all_tokens}
                        recon_dict = recon_loss_fn(recons, targets)
                    for k_r, v_r in recon_dict.items():
                        if isinstance(v_r, torch.Tensor):
                            loss_dict[k_r] = v_r

            _save_recon_images(
                batch_idx,
                alignment_output,
                batch,
                recon_decoders,
                output_dir,
                epoch,
                stage,
                args,
            )


        loss = loss_dict["total_loss"]
        recon_total = loss_dict.get("recon_total")
        if recon_total is not None:
            recon_weight = args.reconstruction_weight if args is not None else 1.0
            loss = loss + recon_weight * recon_total
        metric_logger.update(loss=loss.item())

        acc1 = loss_dict.get("acc1_avg")
        acc5 = loss_dict.get("acc5_avg")
        if acc1 is not None:
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        if acc5 is not None:
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

        for k, v in loss_dict.items():
            if k not in ("total_loss", "acc1_avg", "acc5_avg") and isinstance(v, torch.Tensor):
                metric_logger.update(**{k: v.item()})

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if log_writer is not None and epoch is not None:
        for k, meter in metric_logger.meters.items():
            log_writer.add_scalar(f"val_{k}", meter.global_avg, epoch)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def run(args):
    misc.init_distributed_mode(args)
    torch.backends.cudnn.deterministic = True

    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"{args}".replace(", ", ",\n"))

    device = torch.device(args.device)

    # Seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Build datasets
    dataset_train, dataset_val = build_datasets(args)

    # Samplers
    if True:  # distributed
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True,
        )
        if args.dist_eval:
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True,
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    debug_mode = getattr(args, "debug_single_sample", False)
    effective_num_workers = args.num_workers
    if debug_mode:
        effective_num_workers = int(getattr(args, "debug_num_workers", 0))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=effective_num_workers,
        pin_memory=args.pin_mem, drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size, num_workers=effective_num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

    # Single-sample overfit mode: evaluate on exactly the same stream as training.
    # This is intentionally strict (same dataloader object) for debugging parity.
    if len(dataset_train) == 1 or debug_mode:
        print("[single-sample] Reusing training dataloader for evaluation")
        data_loader_val = data_loader_train

    # Build model
    model = build_model(args, device)
    model_without_ddp = model

    # --- Training stage handling ---
    stage = getattr(args, "stage", "joint")
    if stage == "reconstruction":
        # Load and freeze alignment model for decoder-only training
        if not args.alignment_checkpoint or not os.path.exists(args.alignment_checkpoint):
            raise FileNotFoundError(
                "--stage reconstruction requires a valid --alignment_checkpoint"
            )
        print(f"Loading alignment checkpoint: {args.alignment_checkpoint}")
        align_ckpt = torch.load(args.alignment_checkpoint, map_location="cpu")
        if "alignment_model" not in align_ckpt:
            raise KeyError("alignment checkpoint is missing 'alignment_model' key")
        model_without_ddp.alignment_model.load_state_dict(
            align_ckpt["alignment_model"], strict=False,
        )
        for p in model_without_ddp.alignment_model.parameters():
            p.requires_grad = False
        n_frozen = sum(p.numel() for p in model_without_ddp.alignment_model.parameters())
        print(f"Froze alignment model ({n_frozen:,} params) for reconstruction stage")
    elif stage == "alignment":
        args.decoder_type = "flow_matching"
        print("Alignment stage: register-token alignment + flow reconstruction")

    # Build reconstruction decoders (if enabled)
    recon_decoders = None
    recon_loss_fn = None
    recon_decoders = {}
    alignment_hidden_dim = model_without_ddp.alignment_model.hidden_dim
    decoder_type = getattr(args, "decoder_type", "autoregressive")
    if decoder_type == "flow_matching" and args.recon_loss_type != "mse":
        raise ValueError("--decoder_type flow_matching requires --recon_loss_type mse")
    if decoder_type == "autoregressive":
        for mod_name in [ModalityType.VISION, ModalityType.TACTILE]:
            recon_decoders[mod_name] = AutoregressiveDecoder(
                n_registers=args.n_registers,
                hidden_dim=alignment_hidden_dim,
                base_channels=args.recon_base_channels,
                n_decoder_layers=args.recon_decoder_layers,
                n_heads=args.n_heads,
            ).to(device)
    elif decoder_type == "flow_matching":
        for mod_name in [ModalityType.VISION, ModalityType.TACTILE]:
            recon_decoders[mod_name] = FlowMatchingReconstructionDecoder(
                n_registers=args.n_registers,
                hidden_dim=alignment_hidden_dim,
                base_channels=args.recon_base_channels,
                n_decoder_layers=args.recon_decoder_layers,
                n_heads=args.n_heads,
            ).to(device)
    else:
        for mod_name in [ModalityType.VISION, ModalityType.TACTILE]:
            recon_decoders[mod_name] = ReconstructionDecoder(
                n_registers=args.n_registers,
                hidden_dim=alignment_hidden_dim,
                base_channels=args.recon_base_channels,
                n_decoder_layers=args.recon_decoder_layers,
                n_heads=args.n_heads,
            ).to(device)
    # Choose reconstruction objective based on decoder type.
    # Pixel losses (mse/l1/smooth_l1) are used for conv/autoregressive decoders.
    # Flow-matching decoder is trained with a rectified-flow loss in pixel space.
    if decoder_type == "flow_matching":
        recon_loss_fn = FlowReconstructionLoss(
            loss_type=args.recon_loss_type,
            pixel_mse_weight=args.flow_recon_pixel_mse_weight,
            use_prefix_recon=getattr(args, "use_prefix_recon", False),
            prefix_weight=getattr(args, "prefix_recon_weight", 0.5),
        )
    else:
        recon_loss_fn = ReconstructionLoss(loss_type=args.recon_loss_type)
    n_decoder_params = sum(sum(p.numel() for p in dec.parameters()) for dec in recon_decoders.values())
    print(f"Reconstruction decoders: {n_decoder_params:,} parameters ({len(recon_decoders)} modalities)")

    # Prefix reconstruction is computed inside FlowReconstructionLoss when enabled.
    use_prefix = getattr(args, "use_prefix_recon", False)
    if use_prefix and decoder_type == "flow_matching":
        prefix_weight = getattr(args, "prefix_recon_weight", 0.5)
        print(f"Flow prefix reconstruction enabled (weight={prefix_weight})")

    # Logging
    if global_rank == 0 and args.log_dir is not None and HAS_TENSORBOARD:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Print trainable parameters
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")

    # DDP
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True,
        )
        model_without_ddp = model.module
        # Wrap reconstruction decoders in DDP to synchronize gradients across ranks
        if recon_decoders is not None:
            for mod_name in list(recon_decoders.keys()):
                recon_decoders[mod_name] = torch.nn.parallel.DistributedDataParallel(
                    recon_decoders[mod_name], device_ids=[args.gpu],
                )

    # Build loss
    contrastive_loss_fn, flow_loss_fn = build_loss(
        args, device, embed_dim=model_without_ddp.alignment_model.hidden_dim
    )

    # Optimizer: collect trainable parameters based on stage
    trainable_params = []
    if stage != "reconstruction":
        trainable_params += list(model_without_ddp.alignment_model.parameters())
        if flow_loss_fn is not None:
            trainable_params += list(flow_loss_fn.parameters())
    if recon_decoders is not None:
        for dec in recon_decoders.values():
            trainable_params += list(dec.parameters())
    trainable_params = [p for p in trainable_params if p.requires_grad]

    # Effective batch size for LR scaling
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"actual lr: {args.lr:.2e}")
    print(f"effective batch size: {eff_batch_size}")

    n_opt_params = sum(p.numel() for p in trainable_params)
    print(f"Optimizing {n_opt_params:,} parameters (stage={stage})")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95),
                                   weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()

    # Resume
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        # In reconstruction stage, alignment model is already loaded and frozen
        if stage != "reconstruction":
            model_without_ddp.alignment_model.load_state_dict(ckpt.get("alignment_model", {}), strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            loss_scaler.load_state_dict(ckpt["scaler"])
        if recon_decoders is not None:
            for mod_name, dec in recon_decoders.items():
                key = f"recon_decoder_{mod_name}"
                if key in ckpt:
                    dec.load_state_dict(ckpt[key])
                    print(f"  Loaded reconstruction decoder for {mod_name}")
        args.start_epoch = ckpt.get("epoch", 0) + 1

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_acc1 = 0.0
    best_recon_loss = float("inf")

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, contrastive_loss_fn, flow_loss_fn, data_loader_train,
            optimizer, device, epoch, loss_scaler, args, log_writer=log_writer,
            recon_decoders=recon_decoders, recon_loss_fn=recon_loss_fn,
        )

        val_stats = evaluate(data_loader_val, contrastive_loss_fn, model, device,
                             epoch=epoch, log_writer=log_writer,
                             recon_decoders=recon_decoders, recon_loss_fn=recon_loss_fn, args=args,
                             stage=stage, output_dir=args.output_dir)

        acc1 = val_stats.get("acc1", 0.0)
        recon_loss_val = val_stats.get("recon_total", val_stats.get("recon_pixel_avg", float("inf")))
        if stage == "reconstruction":
            print(f"Epoch {epoch}: val_recon_loss={recon_loss_val:.6f}")
        else:
            print(f"Epoch {epoch}: val_loss={val_stats.get('loss', 0):.4f}, val_acc1={acc1:.1f}%")

        # Save checkpoint
        if args.output_dir:
            save_dict = {
                "alignment_model": model_without_ddp.alignment_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": loss_scaler.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }
            if flow_loss_fn is not None:
                save_dict["flow_loss"] = flow_loss_fn.state_dict()
            if recon_decoders is not None:
                for mod_name, dec in recon_decoders.items():
                    save_dict[f"recon_decoder_{mod_name}"] = dec.state_dict()

            # In reconstruction stage, track best by recon loss (lower is better)
            # Otherwise track best by accuracy (higher is better)
            is_best = False
            if stage == "reconstruction":
                if recon_loss_val < best_recon_loss:
                    best_recon_loss = recon_loss_val
                    is_best = True
            else:
                if acc1 > best_acc1:
                    best_acc1 = acc1
                    is_best = True

            if is_best and misc.is_main_process():
                torch.save(save_dict, os.path.join(args.output_dir, "checkpoint_best.pth"))

            if misc.is_main_process():
                torch.save(save_dict, os.path.join(args.output_dir, "checkpoint_latest.pth"))

        # Log
        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                     **{f"val_{k}": v for k, v in val_stats.items()},
                     "epoch": epoch}
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print(f"Training time: {str(datetime.timedelta(seconds=int(total_time)))}")
    print(f"Best val acc@1: {best_acc1:.1f}%")


def _cfg_to_args(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    args = SimpleNamespace(**cfg_dict)
    if getattr(args, "log_name", None) is not None:
        args.output_dir = os.path.join(args.output_dir, args.log_name)
    if getattr(args, "log_dir", None) is None:
        args.log_dir = args.output_dir
    return args


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def hydra_main(cfg: DictConfig):
    args = _cfg_to_args(cfg)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.log_name is not None and HAS_WANDB and misc.is_main_process():
        wandb.init(entity="project_vit", project="tvl-stage2", config=OmegaConf.to_container(cfg, resolve=True),
                   name=args.log_name, sync_tensorboard=True)

    run(args)


if __name__ == "__main__":
    hydra_main()
