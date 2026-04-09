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

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

import yaml

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
    RGB_PREPROCESS, TAC_PREPROCESS,
    RGB_MEAN, RGB_STD, TAC_MEAN, TAC_STD,
)

from models.cross_modal_alignment import CrossModalAlignmentModel, Stage2Wrapper
from models.reconstruction_decoder import ReconstructionDecoder
from models.autoregressive_decoder import AutoregressiveDecoder
from losses.alignment_loss import CrossModalAlignmentLoss
from losses.flow_matching import FlowMatchingAlignmentLoss
from losses.reconstruction_loss import ReconstructionLoss, PrefixReconstructionLoss

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
    parser.add_argument("--contrastive_weight", type=float, default=1.0)
    parser.add_argument("--preservation_weight", type=float, default=0.5)
    parser.add_argument("--flow_weight", type=float, default=0.5)

    # Reconstruction
    parser.add_argument("--use_reconstruction", action="store_true", default=False,
                        help="Enable reconstruction decoder (FlexTok-style)")
    parser.add_argument("--reconstruction_weight", type=float, default=1.0)
    parser.add_argument("--recon_base_channels", type=int, default=64)
    parser.add_argument("--recon_decoder_layers", type=int, default=2)
    parser.add_argument("--recon_loss_type", type=str, default="mse",
                        choices=["mse", "l1", "smooth_l1"])

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
                        choices=["conv", "autoregressive"],
                        help="Reconstruction decoder: 'autoregressive' (transformer) or 'conv' (ConvTranspose)")

    # Training
    parser.add_argument("--batch_size", default=256, type=int)
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

    # Debugging: overfit to a single data point
    parser.add_argument("--overfit_one_sample", action="store_true", default=False,
                        help="Train on a single data point to verify the network can overfit. "
                             "Use this to debug reconstruction quality issues.")

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
        return dataset_train[0], dataset_val[0]
    elif len(dataset_train) > 1:
        return ConcatDataset(dataset_train), ConcatDataset(dataset_val)
    else:
        raise ValueError(f"No datasets found in {args.datasets_dir}")


def _sanity_check_preprocessing(dataset, device="cpu"):
    """End-to-end sanity check: preprocess → unnormalize → verify roundtrip.

    Runs at training startup to catch normalization/scaling bugs early.
    Checks:
      1. Normalized tensor value ranges are reasonable
      2. Unnormalization roundtrip produces valid [0,1] pixel values
      3. Vision and tactile tensor shapes are correct (3, 224, 224)
      4. No all-black or all-white images after unnormalization
    """
    print("\n" + "=" * 60)
    print("  PREPROCESSING SANITY CHECK")
    print("=" * 60)

    sample = dataset[0]
    ok = True

    for mod_name, mean, std, label in [
        (ModalityType.VISION, RGB_MEAN, RGB_STD, "Vision"),
        (ModalityType.TACTILE, TAC_MEAN, TAC_STD, "Tactile"),
    ]:
        if mod_name not in sample:
            continue
        tensor = sample[mod_name]
        if isinstance(tensor, list):
            tensor = tensor[0]
        tensor = tensor.squeeze()

        # Shape check
        if tensor.shape != (3, 224, 224):
            print(f"  [{label}] *** SHAPE ERROR: expected (3,224,224), got {tuple(tensor.shape)} ***")
            ok = False
        else:
            print(f"  [{label}] Shape: {tuple(tensor.shape)} ✓")

        # Normalized value range
        vmin, vmax = tensor.min().item(), tensor.max().item()
        vmean, vstd = tensor.mean().item(), tensor.std().item()
        print(f"  [{label}] Normalized range: [{vmin:.3f}, {vmax:.3f}], mean={vmean:.3f}, std={vstd:.3f}")
        if vmax > 200:
            print(f"  [{label}] *** ERROR: values in [0,255] range — missing /255 normalization ***")
            ok = False
        if abs(vmax - vmin) < 0.01:
            print(f"  [{label}] *** ERROR: near-constant tensor — data not loaded properly ***")
            ok = False

        # Unnormalization roundtrip
        t_mean = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
        t_std = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
        unnormed = (tensor * t_std + t_mean).clamp(0, 1)
        u_min, u_max = unnormed.min().item(), unnormed.max().item()
        print(f"  [{label}] Unnormalized range: [{u_min:.3f}, {u_max:.3f}]")

        # Check for all-black or all-white
        if u_max < 0.05:
            print(f"  [{label}] *** ERROR: unnormalized image is all black ***")
            ok = False
        if u_min > 0.95:
            print(f"  [{label}] *** ERROR: unnormalized image is all white ***")
            ok = False

        # Check clipping percentage
        pre_clamp = tensor * t_std + t_mean
        n_clipped = ((pre_clamp < 0).sum() + (pre_clamp > 1).sum()).item()
        n_total = pre_clamp.numel()
        clip_pct = 100 * n_clipped / n_total
        if clip_pct > 30:
            print(f"  [{label}] *** WARNING: {clip_pct:.1f}% pixels clipped during unnormalization ***")
        else:
            print(f"  [{label}] Clipping: {clip_pct:.1f}% ✓")

    print(f"  Result: {'PASS' if ok else 'FAIL — check errors above'}")
    print("=" * 60 + "\n")
    return ok


def _sanity_check_reconstruction(recon, target, mod_name, step):
    """Log reconstruction quality metrics during training (called periodically).

    Helps detect scaling mismatches, shape errors, and convergence issues.
    """
    if step % 50 != 0:
        return
    with torch.no_grad():
        mse = F.mse_loss(recon, target).item()
        recon_min, recon_max = recon.min().item(), recon.max().item()
        recon_mean, recon_std = recon.mean().item(), recon.std().item()
        target_min, target_max = target.min().item(), target.max().item()
        target_mean, target_std = target.mean().item(), target.std().item()
        print(f"  [recon_sanity/{mod_name}] step={step} MSE={mse:.6f}")
        print(f"    recon  range=[{recon_min:.3f}, {recon_max:.3f}] mean={recon_mean:.3f} std={recon_std:.3f}")
        print(f"    target range=[{target_min:.3f}, {target_max:.3f}] mean={target_mean:.3f} std={target_std:.3f}")
        if recon.shape != target.shape:
            print(f"    *** SHAPE MISMATCH: recon={recon.shape} target={target.shape} ***")


def _validate_stage1_checkpoint(checkpoint_path, tactile_model="vit_tiny_patch16_224"):
    """Validate that a Stage 1 checkpoint exists and loads correctly.

    Raises clear errors if the checkpoint is missing, corrupted, or doesn't
    contain the expected encoder weights. This catches a common issue where
    the checkpoint is saved in a different location than expected, causing
    the model to train with random (untrained) frozen encoder weights.
    """
    if not checkpoint_path:
        print("\n" + "!" * 60)
        print("  WARNING: No --stage1_checkpoint provided!")
        print("  The frozen encoder will use RANDOM weights.")
        print("  This will produce meaningless alignment results.")
        print("  Provide the path to your Stage 1 TVL encoder checkpoint.")
        print("!" * 60 + "\n")
        return

    if not os.path.exists(checkpoint_path):
        print("\n" + "!" * 60)
        print(f"  ERROR: Stage 1 checkpoint NOT FOUND:")
        print(f"    {checkpoint_path}")
        print("  The frozen encoder will use RANDOM weights.")
        print("  Common locations to check:")
        print("    - /viscam/u/taarush/tvl_enc_vittiny.pth")
        print("    - ./checkpoints/stage1/checkpoint_best.pth")
        print("  Check that the file path is correct.")
        print("!" * 60 + "\n")
        return

    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"Stage 1 checkpoint: {checkpoint_path} ({file_size_mb:.1f} MB)")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    if isinstance(state, dict):
        has_tactile = any("tactile" in k for k in state.keys())
        has_clip = any("clip" in k for k in state.keys())
        n_keys = len(state)
        print(f"  Contains {n_keys} keys (tactile: {has_tactile}, clip: {has_clip})")
        if not has_tactile:
            print("  WARNING: No tactile encoder weights found in checkpoint!")
    else:
        print(f"  WARNING: Unexpected checkpoint format: {type(state)}")


def _validate_loaded_weights(module, name):
    """Spot-check a few parameter tensors to confirm they are not random.

    Random init typically has small std and near-zero mean. Trained weights
    have larger values and non-trivial structure. This catches silent
    fallbacks to random initialization (e.g., wrong checkpoint path).
    """
    print(f"\n  Weight validation for '{name}':")
    checked = 0
    for param_name, param in module.named_parameters():
        if checked >= 3:
            break
        if param.numel() < 10:
            continue
        pmean = param.data.mean().item()
        pstd = param.data.std().item()
        pmin = param.data.min().item()
        pmax = param.data.max().item()
        print(f"    {param_name}: mean={pmean:.6f} std={pstd:.6f} range=[{pmin:.4f}, {pmax:.4f}]")
        checked += 1
    if checked == 0:
        print(f"    (no parameters found)")


def build_model(args, device):
    """Build frozen encoder + Stage 2 alignment model."""
    # Validate Stage 1 checkpoint before loading
    _validate_stage1_checkpoint(args.stage1_checkpoint, args.tactile_model)

    # Build frozen Stage 1 encoder
    frozen_encoder = TVL(
        tactile_model=args.tactile_model,
        active_modalities=[ModalityType.VISION, ModalityType.TACTILE],
    )

    # Load Stage 1 checkpoint if provided
    if args.stage1_checkpoint and os.path.exists(args.stage1_checkpoint):
        print(f"Loading Stage 1 checkpoint from {args.stage1_checkpoint}")
        ckpt = torch.load(args.stage1_checkpoint, map_location="cpu")
        state = ckpt["model"] if "model" in ckpt else ckpt
        msg = frozen_encoder.load_state_dict(state, strict=False)
        print(f"  Loaded (missing: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)})")
        if msg.missing_keys:
            print(f"  Missing keys (first 5): {msg.missing_keys[:5]}")

        # Validate: sample weight values to confirm non-random initialization
        _validate_loaded_weights(frozen_encoder, "frozen_encoder")

    # Determine input dims from the frozen encoder
    # Vision: OpenCLIP ViT-L-14 outputs 768-dim
    # Tactile: depends on tactile_model + whether num_classes matches CLIP width
    vision_dim = frozen_encoder.clip.visual.output_dim if hasattr(frozen_encoder.clip.visual, "output_dim") else 768
    tactile_dim = frozen_encoder.tactile_encoder.num_classes if frozen_encoder.tactile_encoder.num_classes > 0 else frozen_encoder.tactile_encoder.num_features

    modality_configs = {
        ModalityType.VISION: {"input_dim": vision_dim, "feature_type": "pooled"},
        ModalityType.TACTILE: {"input_dim": tactile_dim, "feature_type": "pooled"},
    }

    # Build Stage 2 alignment model
    use_token_type = getattr(args, "use_token_type_embed", True)
    alignment_model = CrossModalAlignmentModel(
        modality_configs=modality_configs,
        hidden_dim=args.hidden_dim,
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


def build_loss(args, device):
    """Build loss function(s) based on configuration."""
    modality_names = [ModalityType.VISION, ModalityType.TACTILE]

    contrastive_loss = CrossModalAlignmentLoss(
        modality_names=modality_names,
        contrastive_weight=args.contrastive_weight,
        preservation_weight=args.preservation_weight,
    )

    flow_loss = None
    if args.loss_type in ["flow_matching", "combined"]:
        flow_loss = FlowMatchingAlignmentLoss(
            modality_names=modality_names,
            embed_dim=args.hidden_dim,
        ).to(device)

    return contrastive_loss, flow_loss


def _unwrap(model):
    """Unwrap DDP model to access inner attributes."""
    return model.module if hasattr(model, "module") else model


def train_one_epoch(
    model, contrastive_loss_fn, flow_loss_fn, data_loader,
    optimizer, device, epoch, loss_scaler, args, log_writer=None,
    recon_decoders=None, recon_loss_fn=None, prefix_recon_loss_fn=None,
):
    model.train()
    # Keep frozen encoder in eval mode
    model_inner = _unwrap(model)
    model_inner.frozen_encoder.eval()
    # In reconstruction stage, the alignment model must also be in eval mode
    # so that nested dropout and regular dropout are disabled. Otherwise the
    # decoder receives randomly-varying register tokens each step and can
    # never learn a consistent mapping (outputs pure noise).
    if getattr(args, "stage", "joint") == "reconstruction":
        model_inner.alignment_model.eval()
    if recon_decoders is not None:
        for dec in recon_decoders.values():
            dec.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 20
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Per-iteration LR schedule
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Move data to device
        for k, v in samples.items():
            if isinstance(v, list):
                v = v[0]
            samples[k] = v.to(device, non_blocking=True).squeeze()

        # First-iteration shape validation
        if data_iter_step == 0 and epoch == 0:
            for k, v in samples.items():
                if isinstance(v, torch.Tensor) and v.ndim >= 3:
                    print(f"  [shape_check] {k}: {tuple(v.shape)}")
                    # Expect (B, 3, 224, 224) or (3, 224, 224) for batch_size=1
                    if v.shape[-3:] != (3, 224, 224) and v.shape[-2:] != (224, 224):
                        print(f"  *** WARNING: unexpected shape for {k}: {tuple(v.shape)} ***")

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

            # Reconstruction loss (reconstruction and joint stages)
            if stage != "alignment":
                # Skip standalone recon when prefix recon is enabled (it already
                # computes full reconstruction internally, avoiding double decode).
                if recon_decoders is not None and recon_loss_fn is not None and prefix_recon_loss_fn is None:
                    recons = {}
                    targets = {}
                    for mod_name in [ModalityType.VISION, ModalityType.TACTILE]:
                        all_tokens_key = f"{mod_name}_all_tokens"
                        if all_tokens_key in alignment_output and mod_name in samples:
                            recons[mod_name] = recon_decoders[mod_name](alignment_output[all_tokens_key])
                            targets[mod_name] = samples[mod_name]
                    if recons:
                        # Periodic sanity check on reconstruction outputs
                        for mod_name in recons:
                            _sanity_check_reconstruction(
                                recons[mod_name], targets[mod_name],
                                mod_name, data_iter_step,
                            )
                        recon_dict = recon_loss_fn(recons, targets)
                        recon_loss = recon_dict["recon_total"]
                        loss = loss + args.reconstruction_weight * recon_loss
                        for k, v in recon_dict.items():
                            if isinstance(v, torch.Tensor):
                                loss_dict[k] = v

                # OAT-style prefix reconstruction loss (if enabled)
                if prefix_recon_loss_fn is not None and recon_decoders is not None:
                    all_tokens = {}
                    targets_pf = {}
                    for mod_name in [ModalityType.VISION, ModalityType.TACTILE]:
                        all_tokens_key = f"{mod_name}_all_tokens"
                        if all_tokens_key in alignment_output and mod_name in samples:
                            all_tokens[mod_name] = alignment_output[all_tokens_key]
                            targets_pf[mod_name] = samples[mod_name]
                    if all_tokens:
                        # Unwrap DDP so PrefixReconstructionLoss can call decoder.forward_prefix
                        raw_decoders = {m: _unwrap(d) for m, d in recon_decoders.items()}
                        prefix_dict = prefix_recon_loss_fn(all_tokens, targets_pf, raw_decoders)
                        prefix_loss = prefix_dict["recon_total"]
                        loss = loss + prefix_loss
                        for k, v in prefix_dict.items():
                            if isinstance(v, torch.Tensor):
                                loss_dict[f"prefix_{k}"] = v

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
    recon_decoders=None, recon_loss_fn=None, prefix_recon_loss_fn=None, args=None,
):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"
    model.eval()
    model_inner = _unwrap(model)
    if recon_decoders is not None:
        for dec in recon_decoders.values():
            dec.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        for k, v in batch.items():
            if isinstance(v, list):
                v = v[0]
            batch[k] = v.to(device, non_blocking=True).squeeze()
            batch_size = v.shape[0]

        with torch.cuda.amp.autocast():
            frozen_features = model_inner.encode(batch)
            alignment_output = model_inner.alignment_model(frozen_features)
            preservation_projectors = model_inner.alignment_model.preservation_projectors
            loss_dict = contrastive_loss_fn(
                alignment_output,
                frozen_features=frozen_features,
                preservation_projectors=preservation_projectors,
                output_dict=True,
            )

            # Reconstruction loss (if enabled)
            # Skip standalone recon when prefix recon is active (mirrors training)
            if recon_decoders is not None and recon_loss_fn is not None and prefix_recon_loss_fn is None:
                recons = {}
                targets = {}
                for mod_name in [ModalityType.VISION, ModalityType.TACTILE]:
                    all_tokens_key = f"{mod_name}_all_tokens"
                    if all_tokens_key in alignment_output and mod_name in batch:
                        recons[mod_name] = recon_decoders[mod_name](alignment_output[all_tokens_key])
                        targets[mod_name] = batch[mod_name]
                if recons:
                    recon_dict = recon_loss_fn(recons, targets)
                    for k_r, v_r in recon_dict.items():
                        if isinstance(v_r, torch.Tensor):
                            loss_dict[k_r] = v_r

            # OAT-style prefix reconstruction loss (if enabled)
            if prefix_recon_loss_fn is not None and recon_decoders is not None:
                all_tokens = {}
                targets_pf = {}
                for mod_name in [ModalityType.VISION, ModalityType.TACTILE]:
                    all_tokens_key = f"{mod_name}_all_tokens"
                    if all_tokens_key in alignment_output and mod_name in batch:
                        all_tokens[mod_name] = alignment_output[all_tokens_key]
                        targets_pf[mod_name] = batch[mod_name]
                if all_tokens:
                    # Unwrap DDP so PrefixReconstructionLoss can call decoder.forward_prefix
                    raw_decoders = {m: _unwrap(d) for m, d in recon_decoders.items()}
                    prefix_dict = prefix_recon_loss_fn(all_tokens, targets_pf, raw_decoders)
                    for k_r, v_r in prefix_dict.items():
                        if isinstance(v_r, torch.Tensor):
                            loss_dict[f"prefix_{k_r}"] = v_r
                    # Use prefix recon_total for model selection
                    loss_dict["recon_total"] = prefix_dict["recon_total"]

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


def main(args):
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

    # Overfit mode: reduce dataset to a single sample for debugging
    if getattr(args, "overfit_one_sample", False):
        print("\n" + "=" * 60)
        print("  OVERFIT MODE: Training on a single data point")
        print("  Augmentations DISABLED for deterministic input")
        print("  Expected: loss should drop to ~0 within a few epochs")
        print("  If it doesn't, there's a bug in architecture/loss/data pipeline")
        print("=" * 60 + "\n")
        # Rebuild dataset without augmentations so the network sees the exact
        # same image every iteration (augmentations would change the target
        # each step, preventing true overfitting).
        root_dir = os.path.join(args.datasets_dir, "ssvtp")
        dataset_overfit = TacVisDataset(
            root_dir=root_dir, split="train",
            transform_rgb=RGB_PREPROCESS, transform_tac=TAC_PREPROCESS,
            modality_types=[ModalityType.VISION, ModalityType.TACTILE],
            text_prompt="This image gives tactile feelings of ",
        )
        dataset_train = torch.utils.data.Subset(dataset_overfit, [0])
        dataset_val = torch.utils.data.Subset(dataset_overfit, [0])
        args.batch_size = 1
        args.nested_dropout = False
        args.warmup_epochs = 0
        args.num_workers = 0  # avoid multiprocessing issues with tiny dataset

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

    overfit = getattr(args, "overfit_one_sample", False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=not overfit,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

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
        msg = model_without_ddp.alignment_model.load_state_dict(
            align_ckpt["alignment_model"], strict=False,
        )
        print(f"  Loaded alignment (missing: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)})")
        if msg.missing_keys:
            print(f"  Missing keys: {msg.missing_keys[:5]}")
        _validate_loaded_weights(model_without_ddp.alignment_model, "alignment_model")

        for p in model_without_ddp.alignment_model.parameters():
            p.requires_grad = False
        n_frozen = sum(p.numel() for p in model_without_ddp.alignment_model.parameters())
        model_without_ddp.alignment_model.eval()
        print(f"Froze alignment model ({n_frozen:,} params) for reconstruction stage")
        args.use_reconstruction = True
    elif stage == "alignment":
        args.use_reconstruction = False
        print("Alignment stage: reconstruction disabled")

    # Build reconstruction decoders (if enabled)
    recon_decoders = None
    recon_loss_fn = None
    prefix_recon_loss_fn = None
    if args.use_reconstruction:
        recon_decoders = {}
        decoder_type = getattr(args, "decoder_type", "autoregressive")
        if decoder_type == "autoregressive":
            for mod_name in [ModalityType.VISION, ModalityType.TACTILE]:
                recon_decoders[mod_name] = AutoregressiveDecoder(
                    n_registers=args.n_registers,
                    hidden_dim=args.hidden_dim,
                    base_channels=args.recon_base_channels,
                    n_decoder_layers=args.recon_decoder_layers,
                    n_heads=args.n_heads,
                ).to(device)
        else:
            for mod_name in [ModalityType.VISION, ModalityType.TACTILE]:
                recon_decoders[mod_name] = ReconstructionDecoder(
                    n_registers=args.n_registers,
                    hidden_dim=args.hidden_dim,
                    base_channels=args.recon_base_channels,
                    n_decoder_layers=args.recon_decoder_layers,
                    n_heads=args.n_heads,
                ).to(device)
        recon_loss_fn = ReconstructionLoss(loss_type=args.recon_loss_type)
        n_decoder_params = sum(
            sum(p.numel() for p in dec.parameters())
            for dec in recon_decoders.values()
        )
        print(f"Reconstruction decoders: {n_decoder_params:,} parameters ({len(recon_decoders)} modalities)")

        # OAT-style prefix reconstruction loss
        use_prefix = getattr(args, "use_prefix_recon", False)
        if use_prefix:
            prefix_weight = getattr(args, "prefix_recon_weight", 0.5)
            prefix_recon_loss_fn = PrefixReconstructionLoss(
                loss_type=args.recon_loss_type,
                prefix_weight=prefix_weight,
            )
            print(f"OAT-style prefix reconstruction enabled (weight={prefix_weight})")

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
    contrastive_loss_fn, flow_loss_fn = build_loss(args, device)

    # Optimizer: collect trainable parameters based on stage
    trainable_params = []
    if stage != "reconstruction":
        trainable_params += list(model_without_ddp.alignment_model.parameters())
        if flow_loss_fn is not None:
            trainable_params += list(flow_loss_fn.parameters())
    if stage != "alignment" and recon_decoders is not None:
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

    # Run preprocessing sanity check before training starts
    _sanity_check_preprocessing(dataset_train)

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
            prefix_recon_loss_fn=prefix_recon_loss_fn,
        )

        val_stats = evaluate(data_loader_val, contrastive_loss_fn, model, device,
                             epoch=epoch, log_writer=log_writer,
                             recon_decoders=recon_decoders, recon_loss_fn=recon_loss_fn,
                             prefix_recon_loss_fn=prefix_recon_loss_fn, args=args)

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


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    # Load config file if provided
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        # Config overrides argparse defaults, but explicit CLI args take precedence.
        # We re-parse sys.argv to discover which args were explicitly provided.
        parser = get_args_parser()
        cli_specified = {action.dest for action in parser._actions
                         if any(opt in sys.argv for opt in action.option_strings)}
        for section in config.values():
            if isinstance(section, dict):
                for k, v in section.items():
                    if hasattr(args, k) and k not in cli_specified:
                        setattr(args, k, v)

    if args.log_name is not None:
        args.output_dir = os.path.join(args.output_dir, args.log_name)
    if args.log_dir is None:
        args.log_dir = args.output_dir
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.log_name is not None and HAS_WANDB and misc.is_main_process():
        wandb.init(entity="project_vit", project="tvl-stage2", config=args,
                   name=args.log_name, sync_tensorboard=True)

    main(args)
