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

# Normalization constants
RGB_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
RGB_STD = np.array([0.26862954, 0.26130258, 0.27577711])
TAC_MEAN = np.array([0.29174602, 0.29713256, 0.29104045])
TAC_STD = np.array([0.18764469, 0.19467652, 0.21871583])


def unnormalize(tensor, mean, std):
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def load_model_and_decoders(checkpoint_path, stage1_checkpoint, n_registers=32,
                             n_shared=8, hidden_dim=512, device="cuda"):
    """Load Stage 2 model + reconstruction decoders from a checkpoint."""
    frozen_encoder = TVL(
        tactile_model="vit_tiny_patch16_224",
        active_modalities=[ModalityType.VISION, ModalityType.TACTILE],
    )
    if stage1_checkpoint and os.path.exists(stage1_checkpoint):
        ckpt = torch.load(stage1_checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt)
        frozen_encoder.load_state_dict(state, strict=False)

    alignment_model = CrossModalAlignmentModel(
        modality_configs={
            ModalityType.VISION: {"input_dim": 768, "feature_type": "pooled"},
            ModalityType.TACTILE: {"input_dim": 768, "feature_type": "pooled"},
        },
        hidden_dim=hidden_dim, n_registers=n_registers, n_shared=n_shared,
        n_layers=4, n_heads=8,
    )
    model = Stage2Wrapper(frozen_encoder, alignment_model)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.alignment_model.load_state_dict(ckpt.get("alignment_model", {}), strict=False)
    model.to(device).eval()

    recon_decoders = {}
    for mod_name in [ModalityType.VISION, ModalityType.TACTILE]:
        dec = ReconstructionDecoder(
            n_registers=n_registers, hidden_dim=hidden_dim,
            base_channels=64, n_decoder_layers=2, n_heads=8,
        )
        key = f"recon_decoder_{mod_name}"
        if key in ckpt:
            dec.load_state_dict(ckpt[key])
        dec.to(device).eval()
        recon_decoders[mod_name] = dec

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
        batch[k] = v.to(device, non_blocking=True).squeeze()

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

        n_cols = 1 + len(prefix_lengths)  # original + each prefix
        fig, axes = plt.subplots(n_samples, n_cols, figsize=(2.5 * n_cols, 2.5 * n_samples))
        fig.suptitle(
            f"OAT-Style Prefix Reconstruction — {label} (n_registers={n_registers})\n"
            "Earlier tokens → global structure, Later tokens → fine details",
            fontsize=13, fontweight="bold",
        )

        for row in range(n_samples):
            # Original
            orig = unnormalize(originals[row].float(), norm_mean, norm_std)
            ax = axes[row, 0] if n_samples > 1 else axes[0]
            ax.imshow(orig.cpu().permute(1, 2, 0).numpy())
            ax.axis("off")
            if row == 0:
                ax.set_title("Original", fontsize=10, fontweight="bold")

            # Prefix reconstructions
            for col_idx, k in enumerate(prefix_lengths):
                with torch.no_grad(), torch.cuda.amp.autocast():
                    recon = decoder.forward_prefix(all_tokens, k=k)
                recon_img = unnormalize(recon[row].float(), norm_mean, norm_std)

                ax = axes[row, col_idx + 1] if n_samples > 1 else axes[col_idx + 1]
                ax.imshow(recon_img.cpu().permute(1, 2, 0).numpy())
                ax.axis("off")
                if row == 0:
                    ax.set_title(f"k={k}", fontsize=10, fontweight="bold")

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
            batch[k] = v.to(device, non_blocking=True).squeeze()

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

    args = parser.parse_args()
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
                        shuffle=True, num_workers=2)

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
        all_mse_curves = {}

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
