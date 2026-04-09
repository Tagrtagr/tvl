#!/usr/bin/env python3
"""
Data Preprocessing Validation Script.

Diagnoses common data scaling and normalization issues that can cause
reconstruction outputs to appear as solid black/white or washed-out colors.

Checks performed:
  1. Raw pixel value ranges before and after normalization
  2. Correct unnormalization (roundtrip: normalize -> unnormalize -> compare)
  3. Detects pure black (all ~0) or pure white (all ~1) after unnormalization
  4. Validates that 255-scaling is correctly applied (not double-applied or missed)
  5. Visualizes sample images before/after normalization for visual inspection
  6. Checks for orientation issues (90-degree rotation) in tactile images

Usage:
    python experiments/stage2/claude_flextok/validate_data.py \
        --datasets_dir /path/to/datasets

    # With stage1 checkpoint to also validate encoder outputs:
    python experiments/stage2/claude_flextok/validate_data.py \
        --datasets_dir /path/to/datasets \
        --stage1_checkpoint /path/to/stage1.pth
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

# Add parent directories to path
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "../../.."))
sys.path.insert(0, os.path.join(_here, "../../../tvl_enc"))

from tvl_enc.tacvis import TacVisDataset, RGB_AUGMENTS, TAC_AUGMENTS
from tvl_enc.tvl import TVL, ModalityType

# Normalization constants (must match training pipeline)
RGB_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
RGB_STD = np.array([0.26862954, 0.26130258, 0.27577711])
TAC_MEAN = np.array([0.29174602, 0.29713256, 0.29104045])
TAC_STD = np.array([0.18764469, 0.19467652, 0.21871583])


def unnormalize(tensor, mean, std):
    """Unnormalize a (C, H, W) tensor back to [0, 1] range."""
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def check_value_range(tensor, name):
    """Check and report the value range of a tensor."""
    vmin, vmax = tensor.min().item(), tensor.max().item()
    vmean = tensor.mean().item()
    vstd = tensor.std().item()

    print(f"\n  [{name}]")
    print(f"    Range: [{vmin:.4f}, {vmax:.4f}]")
    print(f"    Mean:  {vmean:.4f}")
    print(f"    Std:   {vstd:.4f}")

    # Diagnose issues
    issues = []
    if vmax > 200:
        issues.append(f"VALUES ARE IN [0,255] RANGE - likely missing /255 normalization")
    if abs(vmax) < 0.01 and abs(vmin) < 0.01:
        issues.append("ALL NEAR ZERO - image will appear pure black")
    if vmin > 0.95 and vmax > 0.95:
        issues.append("ALL NEAR 1.0 - image will appear pure white")
    if abs(vmean) < 0.001 and vstd < 0.001:
        issues.append("ZERO TENSOR - no data loaded, check data pipeline")

    # Check if already normalized (expected range for ImageNet-style normalization)
    if -3.0 < vmin < 0.0 and 0.0 < vmax < 3.0:
        print(f"    Status: Looks correctly normalized (typical range for ImageNet-style)")
    elif 0.0 <= vmin and vmax <= 1.0:
        print(f"    Status: Values in [0,1] - may be pre-normalization (before mean/std)")
    elif 0.0 <= vmin and vmax <= 255.0:
        print(f"    Status: Values in [0,255] - RAW pixel values, needs /255 + normalize")

    for issue in issues:
        print(f"    *** WARNING: {issue} ***")

    return issues


def check_unnormalization_roundtrip(tensor, mean, std, name):
    """Verify unnormalization produces valid [0,1] pixel values."""
    unnormed = unnormalize(tensor, mean, std)
    vmin, vmax = unnormed.min().item(), unnormed.max().item()
    # Check how much was clipped
    pre_clamp = tensor * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    n_clipped_low = (pre_clamp < 0).sum().item()
    n_clipped_high = (pre_clamp > 1).sum().item()
    n_total = pre_clamp.numel()

    print(f"\n  [{name} - Unnormalization Check]")
    print(f"    Unnormalized range: [{vmin:.4f}, {vmax:.4f}]")
    print(f"    Clipped below 0: {n_clipped_low}/{n_total} ({100*n_clipped_low/n_total:.1f}%)")
    print(f"    Clipped above 1: {n_clipped_high}/{n_total} ({100*n_clipped_high/n_total:.1f}%)")

    if n_clipped_low / n_total > 0.3 or n_clipped_high / n_total > 0.3:
        print(f"    *** WARNING: >30% pixels clipped! Check normalization constants. ***")


def validate_stage1_checkpoint(checkpoint_path, device="cpu"):
    """Validate that a stage 1 checkpoint loads correctly."""
    print(f"\n{'='*60}")
    print("  Stage 1 Checkpoint Validation")
    print(f"{'='*60}")

    if not checkpoint_path:
        print("  No checkpoint path provided.")
        return False

    if not os.path.exists(checkpoint_path):
        print(f"  *** ERROR: Checkpoint NOT FOUND at: {checkpoint_path} ***")
        print(f"  This will cause the model to use random encoder weights!")
        return False

    print(f"  Path: {checkpoint_path}")
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Check structure
    if isinstance(ckpt, dict):
        print(f"  Keys: {list(ckpt.keys())}")
        if "model" in ckpt:
            state = ckpt["model"]
            print(f"  Model state dict keys: {len(state)} entries")
            # Check for expected keys
            has_clip = any("clip" in k for k in state.keys())
            has_tactile = any("tactile" in k for k in state.keys())
            print(f"  Contains CLIP weights: {has_clip}")
            print(f"  Contains tactile weights: {has_tactile}")
            if not has_tactile:
                print("  *** WARNING: No tactile encoder weights found! ***")
        else:
            print(f"  *** WARNING: No 'model' key found. Direct state dict? ***")
            state = ckpt
    else:
        print(f"  *** WARNING: Checkpoint is not a dict, type: {type(ckpt)} ***")
        return False

    # Try to load into model
    try:
        model = TVL(
            tactile_model="vit_tiny_patch16_224",
            active_modalities=[ModalityType.VISION, ModalityType.TACTILE],
        )
        msg = model.load_state_dict(state, strict=False)
        print(f"  Load result - Missing keys: {len(msg.missing_keys)}")
        print(f"  Load result - Unexpected keys: {len(msg.unexpected_keys)}")
        if msg.missing_keys:
            print(f"  Missing (first 5): {msg.missing_keys[:5]}")
        if msg.unexpected_keys:
            print(f"  Unexpected (first 5): {msg.unexpected_keys[:5]}")
        print("  *** Checkpoint loaded successfully ***")
        return True
    except Exception as e:
        print(f"  *** ERROR loading checkpoint: {e} ***")
        return False


def visualize_samples(batch, output_dir, n_samples=4):
    """Visualize raw data samples for visual inspection of preprocessing."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("Data Preprocessing Validation\n"
                 "Columns: Normalized Vision | Unnormalized Vision | "
                 "Normalized Tactile | Unnormalized Tactile",
                 fontsize=12, fontweight="bold")

    n = min(n_samples, batch[ModalityType.VISION].shape[0])
    for i in range(n):
        # Vision
        vis_norm = batch[ModalityType.VISION][i]
        vis_unnorm = unnormalize(vis_norm, RGB_MEAN, RGB_STD)

        # Show normalized (will look weird but should have structure)
        vis_norm_display = vis_norm.permute(1, 2, 0).numpy()
        vis_norm_display = (vis_norm_display - vis_norm_display.min()) / (vis_norm_display.max() - vis_norm_display.min() + 1e-8)
        axes[i, 0].imshow(vis_norm_display)
        axes[i, 0].set_title(f"Vision Normalized\n[{vis_norm.min():.2f}, {vis_norm.max():.2f}]", fontsize=9)
        axes[i, 0].axis("off")

        # Show unnormalized (should look like natural image)
        axes[i, 1].imshow(vis_unnorm.permute(1, 2, 0).numpy())
        axes[i, 1].set_title(f"Vision Unnormalized\n[{vis_unnorm.min():.2f}, {vis_unnorm.max():.2f}]", fontsize=9)
        axes[i, 1].axis("off")

        # Tactile
        tac_norm = batch[ModalityType.TACTILE][i]
        tac_unnorm = unnormalize(tac_norm, TAC_MEAN, TAC_STD)

        tac_norm_display = tac_norm.permute(1, 2, 0).numpy()
        tac_norm_display = (tac_norm_display - tac_norm_display.min()) / (tac_norm_display.max() - tac_norm_display.min() + 1e-8)
        axes[i, 2].imshow(tac_norm_display)
        axes[i, 2].set_title(f"Tactile Normalized\n[{tac_norm.min():.2f}, {tac_norm.max():.2f}]", fontsize=9)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(tac_unnorm.permute(1, 2, 0).numpy())
        axes[i, 3].set_title(f"Tactile Unnormalized\n[{tac_unnorm.min():.2f}, {tac_unnorm.max():.2f}]", fontsize=9)
        axes[i, 3].axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "data_validation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved validation visualization: {path}")


def check_orientation(batch, output_dir):
    """Check for orientation issues (90-degree rotation) in tactile images."""
    os.makedirs(output_dir, exist_ok=True)

    tac = batch[ModalityType.TACTILE][0]
    tac_unnorm = unnormalize(tac, TAC_MEAN, TAC_STD)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Tactile Orientation Check\n"
                 "If 'Original' looks wrong but a rotation looks correct, "
                 "there's a rotation bug", fontsize=11, fontweight="bold")

    for idx, (angle, label) in enumerate([
        (0, "Original (0°)"),
        (1, "Rotated 90° CW"),
        (2, "Rotated 180°"),
        (3, "Rotated 90° CCW"),
    ]):
        img = torch.rot90(tac_unnorm, k=angle, dims=[1, 2])
        axes[idx].imshow(img.permute(1, 2, 0).numpy())
        axes[idx].set_title(label, fontsize=10)
        axes[idx].axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "tactile_orientation_check.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved orientation check: {path}")


def check_pipeline_roundtrip(output_dir, datasets_dir):
    """End-to-end roundtrip: load raw → preprocess → unnormalize → compare.

    This is the key sanity check from the meeting: if we preprocess a raw
    image and immediately invert the preprocessing, we should get back the
    original (no rotation, no scale change, no color shift).

    Also checks that the tactile tac_padding rotation is handled consistently.
    """
    from tvl_enc.tacvis import (
        RGB_PREPROCESS, TAC_PREPROCESS, tac_padding,
        TacVisDataset,
    )
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)
    print("\n  Loading a raw sample to trace the full pipeline...")

    # Load dataset to get file paths
    root_dir = os.path.join(datasets_dir, "ssvtp")
    dataset = TacVisDataset(
        root_dir=root_dir, split="val",
        transform_rgb=RGB_PREPROCESS, transform_tac=TAC_PREPROCESS,
        modality_types=[ModalityType.VISION, ModalityType.TACTILE],
        text_prompt="This image gives tactile feelings of ",
    )

    # Get first sample through dataset pipeline (deterministic, no augmentation)
    sample = dataset[0]
    vis_tensor = sample[ModalityType.VISION]
    tac_tensor = sample[ModalityType.TACTILE]
    if isinstance(vis_tensor, list):
        vis_tensor = vis_tensor[0]
    if isinstance(tac_tensor, list):
        tac_tensor = tac_tensor[0]
    vis_tensor = vis_tensor.squeeze()
    tac_tensor = tac_tensor.squeeze()

    # Also load the raw images for comparison
    img_path = dataset.paths[0]
    tac_path = dataset.get_tactile_path(img_path)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("End-to-End Pipeline Roundtrip Check\n"
                 "Raw → Preprocess → Unnormalize should ≈ match the raw image",
                 fontsize=13, fontweight="bold")

    # --- Vision row ---
    # Raw image
    try:
        raw_vis = Image.open(img_path).convert("RGB")
        axes[0, 0].imshow(raw_vis)
        axes[0, 0].set_title(f"Raw Vision\n{raw_vis.size}", fontsize=9)
    except Exception as e:
        axes[0, 0].set_title(f"Raw load failed:\n{e}", fontsize=8)
    axes[0, 0].axis("off")

    # Normalized tensor (rescaled for display)
    vis_display = vis_tensor.permute(1, 2, 0).numpy()
    vis_display = (vis_display - vis_display.min()) / (vis_display.max() - vis_display.min() + 1e-8)
    axes[0, 1].imshow(vis_display)
    axes[0, 1].set_title(f"Normalized\nrange=[{vis_tensor.min():.2f}, {vis_tensor.max():.2f}]", fontsize=9)
    axes[0, 1].axis("off")

    # Unnormalized (should look like resized raw)
    vis_unnorm = unnormalize(vis_tensor, RGB_MEAN, RGB_STD)
    axes[0, 2].imshow(vis_unnorm.permute(1, 2, 0).numpy())
    axes[0, 2].set_title(f"Unnormalized\nrange=[{vis_unnorm.min():.2f}, {vis_unnorm.max():.2f}]", fontsize=9)
    axes[0, 2].axis("off")

    # Per-channel stats
    stats_text = "Per-channel (unnorm):\n"
    for c, name in enumerate(["R", "G", "B"]):
        ch = vis_unnorm[c]
        stats_text += f"  {name}: [{ch.min():.3f}, {ch.max():.3f}] μ={ch.mean():.3f}\n"
    axes[0, 3].text(0.1, 0.5, stats_text, fontsize=9, family="monospace",
                     transform=axes[0, 3].transAxes, verticalalignment="center")
    axes[0, 3].set_title("Vision Stats", fontsize=9)
    axes[0, 3].axis("off")

    # --- Tactile row ---
    try:
        raw_tac = Image.open(tac_path).convert("RGB")
        axes[1, 0].imshow(raw_tac)
        axes[1, 0].set_title(f"Raw Tactile\n{raw_tac.size}", fontsize=9)
    except Exception as e:
        axes[1, 0].set_title(f"Raw load failed:\n{e}", fontsize=8)
    axes[1, 0].axis("off")

    tac_display = tac_tensor.permute(1, 2, 0).numpy()
    tac_display = (tac_display - tac_display.min()) / (tac_display.max() - tac_display.min() + 1e-8)
    axes[1, 1].imshow(tac_display)
    axes[1, 1].set_title(f"Normalized\nrange=[{tac_tensor.min():.2f}, {tac_tensor.max():.2f}]", fontsize=9)
    axes[1, 1].axis("off")

    tac_unnorm = unnormalize(tac_tensor, TAC_MEAN, TAC_STD)
    axes[1, 2].imshow(tac_unnorm.permute(1, 2, 0).numpy())
    axes[1, 2].set_title(f"Unnormalized\nrange=[{tac_unnorm.min():.2f}, {tac_unnorm.max():.2f}]", fontsize=9)
    axes[1, 2].axis("off")

    stats_text = "Per-channel (unnorm):\n"
    for c, name in enumerate(["R", "G", "B"]):
        ch = tac_unnorm[c]
        stats_text += f"  {name}: [{ch.min():.3f}, {ch.max():.3f}] μ={ch.mean():.3f}\n"
    stats_text += f"\nNote: tac_padding applies\n90° rotation before transforms"
    axes[1, 3].text(0.1, 0.5, stats_text, fontsize=9, family="monospace",
                     transform=axes[1, 3].transAxes, verticalalignment="center")
    axes[1, 3].set_title("Tactile Stats", fontsize=9)
    axes[1, 3].axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "pipeline_roundtrip.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved pipeline roundtrip: {path}")

    # Print numerical comparison
    print(f"\n  Vision  - normalized: [{vis_tensor.min():.4f}, {vis_tensor.max():.4f}]"
          f"  →  unnormalized: [{vis_unnorm.min():.4f}, {vis_unnorm.max():.4f}]")
    print(f"  Tactile - normalized: [{tac_tensor.min():.4f}, {tac_tensor.max():.4f}]"
          f"  →  unnormalized: [{tac_unnorm.min():.4f}, {tac_unnorm.max():.4f}]")


def main():
    parser = argparse.ArgumentParser("Data Preprocessing Validation")
    parser.add_argument("--datasets_dir", type=str, required=True)
    parser.add_argument("--stage1_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="visualizations/data_validation")
    parser.add_argument("--n_samples", type=int, default=4)
    args = parser.parse_args()

    print("=" * 60)
    print("  Data Preprocessing Validation")
    print("=" * 60)

    # Load dataset
    root_dir = os.path.join(args.datasets_dir, "ssvtp")
    dataset = TacVisDataset(
        root_dir=root_dir, split="val",
        transform_rgb=RGB_AUGMENTS, transform_tac=TAC_AUGMENTS,
        modality_types=[ModalityType.VISION, ModalityType.TACTILE],
        text_prompt="This image gives tactile feelings of ",
    )

    loader = DataLoader(dataset, batch_size=args.n_samples, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    # Squeeze extra dims
    for k, v in batch.items():
        if isinstance(v, list):
            v = v[0]
        batch[k] = v.squeeze()

    # Check 1: Value ranges
    print("\n" + "-" * 40)
    print("  CHECK 1: Value Ranges After Normalization")
    print("-" * 40)
    vis_issues = check_value_range(batch[ModalityType.VISION], "Vision (RGB)")
    tac_issues = check_value_range(batch[ModalityType.TACTILE], "Tactile")

    # Check 2: Unnormalization roundtrip
    print("\n" + "-" * 40)
    print("  CHECK 2: Unnormalization Roundtrip")
    print("-" * 40)
    check_unnormalization_roundtrip(batch[ModalityType.VISION][0], RGB_MEAN, RGB_STD, "Vision")
    check_unnormalization_roundtrip(batch[ModalityType.TACTILE][0], TAC_MEAN, TAC_STD, "Tactile")

    # Check 3: Per-channel statistics (detect if channels are swapped or missing)
    print("\n" + "-" * 40)
    print("  CHECK 3: Per-Channel Statistics")
    print("-" * 40)
    for name, tensor in [("Vision", batch[ModalityType.VISION]), ("Tactile", batch[ModalityType.TACTILE])]:
        print(f"\n  [{name}]")
        for c, ch_name in enumerate(["R/Ch0", "G/Ch1", "B/Ch2"]):
            ch = tensor[:, c]
            print(f"    {ch_name}: mean={ch.mean():.4f}, std={ch.std():.4f}, "
                  f"range=[{ch.min():.4f}, {ch.max():.4f}]")

    # Check 4: Visualize samples
    print("\n" + "-" * 40)
    print("  CHECK 4: Visual Inspection")
    print("-" * 40)
    visualize_samples(batch, args.output_dir, n_samples=args.n_samples)

    # Check 5: Orientation
    print("\n" + "-" * 40)
    print("  CHECK 5: Tactile Orientation")
    print("-" * 40)
    check_orientation(batch, args.output_dir)

    # Check 6: End-to-end pipeline roundtrip (deterministic, no augmentation)
    print("\n" + "-" * 40)
    print("  CHECK 6: Pipeline Roundtrip (preprocess → unnormalize → compare)")
    print("-" * 40)
    check_pipeline_roundtrip(args.output_dir, args.datasets_dir)

    # Check 7: Stage 1 checkpoint
    if args.stage1_checkpoint:
        validate_stage1_checkpoint(args.stage1_checkpoint)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_issues = vis_issues + tac_issues
    if all_issues:
        print("  ISSUES FOUND:")
        for issue in all_issues:
            print(f"    - {issue}")
    else:
        print("  No obvious data scaling issues detected.")
    print(f"\n  Check visualizations in: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
