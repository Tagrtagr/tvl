#!/usr/bin/env python3
"""
Prefix-Based Reconstruction Visualization (OAT-style anytime decoding).

Inspired by Ordered Action Tokenization (OAT), this script visualizes
what information each register token captures by reconstructing with
only the first K tokens. Earlier tokens should capture global structure
(color, shape) while later tokens refine fine-grained details (texture, edges).

This directly addresses the meeting question: "what information are those
register tokens preserving?"

Usage:
    python experiments/stage2/claude_flextok/visualize_prefix_recon.py \
        --checkpoint runs/claude_flextok_recon/checkpoint_best.pth \
        --stage1_checkpoint /path/to/stage1.pth \
        --datasets_dir /path/to/datasets \
        --output_dir visualizations/prefix_recon

    # Compare across different register counts from sweep:
    python experiments/stage2/claude_flextok/visualize_prefix_recon.py \
        --sweep_dir output/sweep_registers \
        --stage1_checkpoint /path/to/stage1.pth \
        --datasets_dir /path/to/datasets \
        --output_dir visualizations/prefix_recon
"""

import argparse
import json
import gc
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
from torch.utils.data import DataLoader

# Add project paths
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "../../.."))
sys.path.insert(0, os.path.join(_here, "../../../tvl_enc"))

from tvl_enc.tacvis import TacVisDataset, RGB_AUGMENTS, TAC_AUGMENTS
from tvl_enc.tvl import TVL, ModalityType
from models.cross_modal_alignment import CrossModalAlignmentModel, Stage2Wrapper
from models.reconstruction_decoder import ReconstructionDecoder
from models.autoregressive_decoder import AutoregressiveDecoder

# Normalization constants
RGB_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
RGB_STD = np.array([0.26862954, 0.26130258, 0.27577711])
TAC_MEAN = np.array([0.29174602, 0.29713256, 0.29104045])
TAC_STD = np.array([0.18764469, 0.19467652, 0.21871583])


def unnormalize(tensor, mean, std):
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def _strip_checkpoint(checkpoint_path):
    """Create a lightweight version of the checkpoint without optimizer/scaler.

    If a stripped version already exists (same path with .stripped.pth),
    returns its path. Otherwise, loads the checkpoint on a machine with
    enough RAM, removes optimizer/scaler, and re-saves it.
    """
    stripped_path = checkpoint_path.replace(".pth", ".stripped.pth")
    if os.path.exists(stripped_path):
        return stripped_path

    print(f"Creating stripped checkpoint (removing optimizer/scaler)...")
    print(f"This is a one-time operation. Loading: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    for key in ["optimizer", "scaler"]:
        if key in ckpt:
            print(f"  Removing '{key}' (freeing memory)")
            del ckpt[key]
    gc.collect()
    torch.save(ckpt, stripped_path)
    del ckpt
    gc.collect()
    print(f"Saved stripped checkpoint: {stripped_path}")
    return stripped_path


def _load_checkpoint_lightweight(checkpoint_path):
    """Load checkpoint, preferring a stripped version if available."""
    # Check for pre-stripped version
    stripped_path = checkpoint_path.replace(".pth", ".stripped.pth")
    load_path = stripped_path if os.path.exists(stripped_path) else checkpoint_path

    ckpt = torch.load(load_path, map_location="cpu")
    # Safety: remove heavy keys if they somehow survived
    for key in ["optimizer", "scaler"]:
        if key in ckpt:
            del ckpt[key]
    gc.collect()
    return ckpt


def load_model_and_decoders(checkpoint_path, stage1_checkpoint, n_registers=32,
                             n_shared=8, hidden_dim=512, device="cuda"):
    """Load Stage 2 model + reconstruction decoders from a checkpoint.

    Model parameters are read from the saved run configuration in
    ckpt["args"] when available, falling back to the function defaults.
    """
    # Selectively load only the keys we need (skip optimizer/scaler to save RAM)
    ckpt = _load_checkpoint_lightweight(checkpoint_path)
    saved_args = ckpt.get("args", {})

    # Read model hyperparameters from checkpoint args, fall back to defaults
    n_registers = saved_args.get("n_registers", n_registers)
    n_shared = saved_args.get("n_shared", n_shared)
    hidden_dim = saved_args.get("hidden_dim", hidden_dim)
    n_layers = saved_args.get("n_layers", 4)
    n_heads = saved_args.get("n_heads", 8)
    tactile_model = saved_args.get("tactile_model", "vit_tiny_patch16_224")
    base_channels = saved_args.get("recon_base_channels", 64)
    n_decoder_layers = saved_args.get("recon_decoder_layers", 2)

    # Extract only what we need from the checkpoint and free it
    alignment_state = ckpt.get("alignment_model", {})
    decoder_states = {}
    for mod_name in [ModalityType.VISION, ModalityType.TACTILE]:
        key = f"recon_decoder_{mod_name}"
        if key not in ckpt:
            raise KeyError(
                f"Checkpoint missing '{key}'. This visualization requires a "
                "reconstruction checkpoint (from Stage 2b or joint training)."
            )
        decoder_states[mod_name] = ckpt[key]
    del ckpt
    gc.collect()

    frozen_encoder = TVL(
        tactile_model=tactile_model,
        active_modalities=[ModalityType.VISION, ModalityType.TACTILE],
    )
    if stage1_checkpoint and os.path.exists(stage1_checkpoint):
        s1_ckpt = torch.load(stage1_checkpoint, map_location="cpu")
        state = s1_ckpt.get("model", s1_ckpt)
        frozen_encoder.load_state_dict(state, strict=False)
        del s1_ckpt, state
        gc.collect()

    # Determine input dims from frozen encoder
    vision_dim = frozen_encoder.clip.visual.output_dim if hasattr(frozen_encoder.clip.visual, "output_dim") else 768
    tactile_dim = frozen_encoder.tactile_encoder.num_classes if frozen_encoder.tactile_encoder.num_classes > 0 else frozen_encoder.tactile_encoder.num_features

    alignment_model = CrossModalAlignmentModel(
        modality_configs={
            ModalityType.VISION: {"input_dim": vision_dim, "feature_type": "pooled"},
            ModalityType.TACTILE: {"input_dim": tactile_dim, "feature_type": "pooled"},
        },
        hidden_dim=hidden_dim, n_registers=n_registers, n_shared=n_shared,
        n_layers=n_layers, n_heads=n_heads,
    )
    model = Stage2Wrapper(frozen_encoder, alignment_model)

    model.alignment_model.load_state_dict(alignment_state, strict=False)
    del alignment_state
    gc.collect()
    model.to(device).eval()

    recon_decoders = {}
    for mod_name in [ModalityType.VISION, ModalityType.TACTILE]:
        state = decoder_states[mod_name]
        # Detect decoder type from saved state: autoregressive decoders have
        # cross-attention keys while convolutional decoders don't.
        is_autoregressive = any("cross_attn" in k or "spatial_queries" in k for k in state)
        if is_autoregressive:
            dec = AutoregressiveDecoder(
                n_registers=n_registers, hidden_dim=hidden_dim,
                base_channels=base_channels, n_decoder_layers=n_decoder_layers,
                n_heads=n_heads,
            )
        else:
            dec = ReconstructionDecoder(
                n_registers=n_registers, hidden_dim=hidden_dim,
                base_channels=base_channels, n_decoder_layers=n_decoder_layers,
                n_heads=n_heads,
            )
        dec.load_state_dict(state)
        dec.to(device).eval()
        recon_decoders[mod_name] = dec
    del decoder_states
    gc.collect()

    return model, recon_decoders


def get_prefix_lengths(n_registers):
    """Get power-of-two prefix lengths up to n_registers."""
    lengths = []
    k = 1
    while k <= n_registers:
        lengths.append(k)
        k *= 2
    if lengths[-1] != n_registers:
        lengths.append(n_registers)
    return lengths


def plot_prefix_reconstruction(model, recon_decoders, dataloader, n_registers,
                                output_dir, n_samples=4, device="cuda"):
    """
    Generate prefix reconstruction grid showing how image quality improves
    as more tokens are used for decoding.

    Rows: samples
    Columns: original, k=1, k=2, k=4, ..., k=n_registers
    """
    batch = next(iter(dataloader))
    for k, v in batch.items():
        if isinstance(v, list):
            v = v[0]
        batch[k] = v.to(device, non_blocking=True)
        if batch[k].dim() > 4:
            batch[k] = batch[k].squeeze(1)

    prefix_lengths = get_prefix_lengths(n_registers)

    with torch.no_grad(), torch.cuda.amp.autocast():
        frozen_features = model.frozen_encoder(batch)
        frozen_features.pop("logit_scale", None)
        frozen_features.pop("logit_bias", None)
        output = model.alignment_model(frozen_features)

    for mod_name, (norm_mean, norm_std, label) in [
        (ModalityType.VISION, (RGB_MEAN, RGB_STD, "Vision")),
        (ModalityType.TACTILE, (TAC_MEAN, TAC_STD, "Tactile")),
    ]:
        all_tokens = output[f"{mod_name}_all_tokens"]
        decoder = recon_decoders[mod_name]
        originals = batch[mod_name]

        # Clamp n_samples to actual batch size
        n_rows = min(n_samples, originals.shape[0])

        # Precompute all prefix reconstructions once (avoid redundant batch decoding)
        prefix_recons = {}
        for k in prefix_lengths:
            with torch.no_grad(), torch.cuda.amp.autocast():
                prefix_recons[k] = decoder.forward_prefix(all_tokens, k=k)

        n_cols = 1 + len(prefix_lengths)  # original + each prefix
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]  # ensure 2D indexing
        fig.suptitle(
            f"OAT-Style Prefix Reconstruction — {label} (n_registers={n_registers})\n"
            "Earlier tokens → global structure, Later tokens → fine details",
            fontsize=13, fontweight="bold",
        )

        for row in range(n_rows):
            # Original
            orig = unnormalize(originals[row].float(), norm_mean, norm_std)
            axes[row, 0].imshow(orig.cpu().permute(1, 2, 0).numpy())
            axes[row, 0].axis("off")
            if row == 0:
                axes[row, 0].set_title("Original", fontsize=10, fontweight="bold")

            # Prefix reconstructions
            for col_idx, k in enumerate(prefix_lengths):
                recon_img = unnormalize(prefix_recons[k][row].float(), norm_mean, norm_std)
                axes[row, col_idx + 1].imshow(recon_img.cpu().permute(1, 2, 0).numpy())
                axes[row, col_idx + 1].axis("off")
                if row == 0:
                    axes[row, col_idx + 1].set_title(f"k={k}", fontsize=10, fontweight="bold")

        plt.tight_layout()
        path = os.path.join(output_dir, f"prefix_recon_{label.lower()}_reg{n_registers}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def plot_prefix_mse_curve(model, recon_decoders, dataloader, n_registers,
                           output_dir, device="cuda"):
    """
    Plot reconstruction MSE as a function of prefix length K.
    OAT predicts monotonically decreasing MSE as K increases.
    """
    prefix_lengths = get_prefix_lengths(n_registers)
    mse_per_k = {mod: {k: [] for k in prefix_lengths}
                 for mod in [ModalityType.VISION, ModalityType.TACTILE]}

    n_batches = 0
    for batch in dataloader:
        if n_batches >= 5:
            break
        for k, v in batch.items():
            if isinstance(v, list):
                v = v[0]
            batch[k] = v.to(device, non_blocking=True)
            if batch[k].dim() > 4:
                batch[k] = batch[k].squeeze(1)

        with torch.no_grad(), torch.cuda.amp.autocast():
            frozen_features = model.frozen_encoder(batch)
            frozen_features.pop("logit_scale", None)
            frozen_features.pop("logit_bias", None)
            output = model.alignment_model(frozen_features)

        for mod_name in [ModalityType.VISION, ModalityType.TACTILE]:
            all_tokens = output[f"{mod_name}_all_tokens"]
            decoder = recon_decoders[mod_name]
            target = batch[mod_name]

            for k in prefix_lengths:
                with torch.no_grad(), torch.cuda.amp.autocast():
                    recon = decoder.forward_prefix(all_tokens, k=k)
                mse = torch.nn.functional.mse_loss(recon.float(), target.float()).item()
                mse_per_k[mod_name][k].append(mse)

        n_batches += 1

    # Average MSE per prefix length
    fig, ax = plt.subplots(figsize=(8, 5))
    for mod_name, color, label in [
        (ModalityType.VISION, "dodgerblue", "Vision"),
        (ModalityType.TACTILE, "orangered", "Tactile"),
    ]:
        k_vals = prefix_lengths
        mean_mse = [np.mean(mse_per_k[mod_name][k]) for k in k_vals]
        ax.plot(k_vals, mean_mse, "-o", color=color, label=label, linewidth=2, markersize=6)
        for k, mse_val in zip(k_vals, mean_mse):
            ax.annotate(f"{mse_val:.4f}", (k, mse_val), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)

    ax.set_xlabel("Prefix Length K (tokens used for decoding)", fontsize=12)
    ax.set_ylabel("Reconstruction MSE", fontsize=12)
    ax.set_title(
        f"OAT-Style Anytime Decoding Quality (n_registers={n_registers})\n"
        "MSE should decrease monotonically as K increases",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(prefix_lengths)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"prefix_mse_curve_reg{n_registers}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser("Prefix Reconstruction Visualization")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single checkpoint to visualize")
    parser.add_argument("--sweep_dir", type=str, default=None,
                        help="Sweep output dir (contains reg8/, reg16/, etc.)")
    parser.add_argument("--stage1_checkpoint", type=str, default=None)
    parser.add_argument("--datasets_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="visualizations/prefix_recon")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--n_registers", type=int, default=32,
                        help="n_registers for single checkpoint mode")
    parser.add_argument("--n_shared", type=int, default=8,
                        help="n_shared for single checkpoint mode")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--strip_checkpoint", action="store_true",
                        help="Only strip optimizer/scaler from checkpoint and exit. "
                             "Run this on a machine with enough RAM first.")

    args = parser.parse_args()

    # Strip mode: create lightweight checkpoint and exit
    if args.strip_checkpoint:
        if not args.checkpoint:
            parser.error("--strip_checkpoint requires --checkpoint")
        _strip_checkpoint(args.checkpoint)
        sys.exit(0)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load validation data
    root_dir = os.path.join(args.datasets_dir, "ssvtp")
    dataset_val = TacVisDataset(
        root_dir=root_dir, split="val",
        transform_rgb=RGB_AUGMENTS, transform_tac=TAC_AUGMENTS,
        modality_types=[ModalityType.VISION, ModalityType.TACTILE],
        text_prompt="This image gives tactile feelings of ",
    )
    loader = DataLoader(dataset_val, batch_size=max(args.n_samples, 16),
                        shuffle=False, num_workers=0)

    if args.checkpoint:
        # Single checkpoint mode
        model, decoders = load_model_and_decoders(
            args.checkpoint, args.stage1_checkpoint,
            n_registers=args.n_registers, n_shared=args.n_shared,
            device=args.device,
        )
        plot_prefix_reconstruction(model, decoders, loader, args.n_registers,
                                    args.output_dir, n_samples=args.n_samples,
                                    device=args.device)
        plot_prefix_mse_curve(model, decoders, loader, args.n_registers,
                               args.output_dir, device=args.device)

    elif args.sweep_dir:
        # Sweep mode: compare across register counts
        sweep_configs = {8: 2, 16: 4, 32: 8, 64: 16}

        for n_reg, n_shared in sweep_configs.items():
            ckpt_path = os.path.join(args.sweep_dir, f"reg{n_reg}", "checkpoint_best.pth")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(args.sweep_dir, f"reg{n_reg}", f"reg{n_reg}", "checkpoint_best.pth")
            if not os.path.exists(ckpt_path):
                print(f"  Skipping reg{n_reg}: no checkpoint found")
                continue

            print(f"\n{'='*50}")
            print(f"  Processing n_registers={n_reg}")
            print(f"{'='*50}")

            model, decoders = load_model_and_decoders(
                ckpt_path, args.stage1_checkpoint,
                n_registers=n_reg, n_shared=n_shared,
                device=args.device,
            )
            plot_prefix_reconstruction(model, decoders, loader, n_reg,
                                        args.output_dir, n_samples=args.n_samples,
                                        device=args.device)
            plot_prefix_mse_curve(model, decoders, loader, n_reg,
                                   args.output_dir, device=args.device)

            # Clean up GPU memory
            del model, decoders
            torch.cuda.empty_cache()

    print(f"\nAll visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
