"""
Visualization script for FlexTok Stage 2 results.

Generates:
1. Training curves: accuracy & loss over epochs (contrastive-only vs joint)
2. Reconstruction samples: original vs reconstructed (vision + tactile)
3. Reconstruction loss breakdown: vision vs tactile over epochs
4. t-SNE of shared embeddings (vision vs tactile)
5. Variable-length token evaluation: accuracy vs number of tokens used

Usage on cluster:
    # All plots from log files (no GPU needed):
    python experiments/stage2/claude_flextok/visualize.py \
        --contrastive_log runs/claude_flextok_ssvtp/claude_flextok_ssvtp/log.txt \
        --recon_log runs/claude_flextok_recon/log.txt \
        --output_dir visualizations/

    # Reconstruction samples (needs GPU + data):
    python experiments/stage2/claude_flextok/visualize.py \
        --recon_checkpoint runs/claude_flextok_recon/checkpoint_best.pth \
        --stage1_checkpoint /viscam/u/taarush/tvl_enc_vittiny.pth \
        --datasets_dir /viscam/u/taarush/ssvtp \
        --output_dir visualizations/ \
        --sample_reconstructions

    # t-SNE plot (needs GPU + data):
    python experiments/stage2/claude_flextok/visualize.py \
        --recon_checkpoint runs/claude_flextok_recon/checkpoint_best.pth \
        --stage1_checkpoint /viscam/u/taarush/tvl_enc_vittiny.pth \
        --datasets_dir /viscam/u/taarush/ssvtp \
        --output_dir visualizations/ \
        --tsne

    # Variable-length token eval (needs GPU + data):
    python experiments/stage2/claude_flextok/visualize.py \
        --recon_checkpoint runs/claude_flextok_recon/checkpoint_best.pth \
        --stage1_checkpoint /viscam/u/taarush/tvl_enc_vittiny.pth \
        --datasets_dir /viscam/u/taarush/ssvtp \
        --output_dir visualizations/ \
        --variable_length

    # Everything at once:
    python experiments/stage2/claude_flextok/visualize.py \
        --contrastive_log runs/claude_flextok_ssvtp/claude_flextok_ssvtp/log.txt \
        --recon_log runs/claude_flextok_recon/log.txt \
        --recon_checkpoint runs/claude_flextok_recon/checkpoint_best.pth \
        --stage1_checkpoint /viscam/u/taarush/tvl_enc_vittiny.pth \
        --datasets_dir /viscam/u/taarush/ssvtp \
        --output_dir visualizations/ \
        --sample_reconstructions --tsne --variable_length
"""

import argparse
import json
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─── Normalization constants (from tvl_enc/tacvis.py) ───
RGB_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
RGB_STD = np.array([0.26862954, 0.26130258, 0.27577711])
TAC_MEAN = np.array([0.29174602, 0.29713256, 0.29104045])
TAC_STD = np.array([0.18764469, 0.19467652, 0.21871583])


def parse_log(log_path):
    """Parse a log.txt file (JSON lines) into a list of dicts."""
    entries = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def unnormalize(tensor, mean, std):
    """Unnormalize a (C, H, W) tensor to [0, 1] range for display."""
    import torch
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    img = tensor * std + mean
    return img.clamp(0, 1)


def fix_tactile_orientation(tensor):
    """Fix 90-degree rotation in tactile reconstructions.

    The decoder can produce tactile outputs that are rotated 90 degrees
    relative to the original. This applies a counter-clockwise 90-degree
    rotation to correct the orientation.

    Args:
        tensor: (C, H, W) or (B, C, H, W) image tensor.

    Returns:
        Rotation-corrected tensor.
    """
    import torch
    if tensor.ndim == 3:
        return torch.rot90(tensor, k=-1, dims=[1, 2])
    elif tensor.ndim == 4:
        return torch.rot90(tensor, k=-1, dims=[2, 3])
    return tensor


# ─────────────────────────────────────────────────
# Plot 1: Training curves (accuracy + loss by epoch)
# ─────────────────────────────────────────────────
def plot_training_curves(contrastive_log, recon_log, output_dir):
    """Plot accuracy and loss curves, comparing contrastive-only vs joint training."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("FlexTok Stage 2 Training Curves", fontsize=16, fontweight="bold")

    has_contrastive = contrastive_log is not None
    has_recon = recon_log is not None

    if has_contrastive:
        c_data = parse_log(contrastive_log)
    if has_recon:
        r_data = parse_log(recon_log)

    # --- Top Left: Val Accuracy ---
    ax = axes[0, 0]
    if has_contrastive:
        epochs = [d["epoch"] for d in c_data]
        val_acc = [d.get("val_acc1", 0) for d in c_data]
        ax.plot(epochs, val_acc, "b-o", markersize=3, label="Contrastive Only", linewidth=2)
    if has_recon:
        epochs = [d["epoch"] for d in r_data]
        val_acc = [d.get("val_acc1", 0) for d in r_data]
        ax.plot(epochs, val_acc, "r-s", markersize=3, label="Joint (+ Reconstruction)", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Validation Accuracy (Acc@1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Top Right: Train Accuracy ---
    ax = axes[0, 1]
    if has_contrastive:
        epochs = [d["epoch"] for d in c_data]
        train_acc = [d.get("train_acc1", d.get("train_acc1_avg", 0)) for d in c_data]
        ax.plot(epochs, train_acc, "b-o", markersize=3, label="Contrastive Only", linewidth=2)
    if has_recon:
        epochs = [d["epoch"] for d in r_data]
        train_acc = [d.get("train_acc1", d.get("train_acc1_avg", 0)) for d in r_data]
        ax.plot(epochs, train_acc, "r-s", markersize=3, label="Joint (+ Reconstruction)", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training Accuracy (Acc@1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Bottom Left: Total Loss ---
    ax = axes[1, 0]
    if has_contrastive:
        epochs = [d["epoch"] for d in c_data]
        train_loss = [d.get("train_loss", 0) for d in c_data]
        val_loss = [d.get("val_loss", 0) for d in c_data]
        ax.plot(epochs, train_loss, "b--", markersize=2, label="Contrastive Train", linewidth=1.5, alpha=0.7)
        ax.plot(epochs, val_loss, "b-", markersize=2, label="Contrastive Val", linewidth=2)
    if has_recon:
        epochs = [d["epoch"] for d in r_data]
        train_loss = [d.get("train_loss", 0) for d in r_data]
        val_loss = [d.get("val_loss", 0) for d in r_data]
        ax.plot(epochs, train_loss, "r--", markersize=2, label="Joint Train", linewidth=1.5, alpha=0.7)
        ax.plot(epochs, val_loss, "r-", markersize=2, label="Joint Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Bottom Right: Reconstruction Loss (if available) ---
    ax = axes[1, 1]
    if has_recon:
        epochs = [d["epoch"] for d in r_data]
        recon_vis = [d.get("train_recon_pixel_vision", d.get("train_recon_vision", 0)) for d in r_data]
        recon_tac = [d.get("train_recon_pixel_tactile", d.get("train_recon_tactile", 0)) for d in r_data]
        recon_total = [d.get("train_recon_total", 0) for d in r_data]
        val_recon_vis = [d.get("val_recon_pixel_vision", d.get("val_recon_vision", 0)) for d in r_data]
        val_recon_tac = [d.get("val_recon_pixel_tactile", d.get("val_recon_tactile", 0)) for d in r_data]

        if any(v > 0 for v in recon_total):
            ax.plot(epochs, recon_vis, "g--", label="Vision (train)", linewidth=1.5, alpha=0.7)
            ax.plot(epochs, val_recon_vis, "g-", label="Vision (val)", linewidth=2)
            ax.plot(epochs, recon_tac, "m--", label="Tactile (train)", linewidth=1.5, alpha=0.7)
            ax.plot(epochs, val_recon_tac, "m-", label="Tactile (val)", linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE Loss")
            ax.set_title("Reconstruction Loss by Modality")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No reconstruction\ndata available", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="gray")
            ax.set_title("Reconstruction Loss by Modality")
    else:
        ax.text(0.5, 0.5, "No reconstruction\nlog provided", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_title("Reconstruction Loss by Modality")

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────
# Plot 2: Reconstruction samples (original vs reconstructed)
# ─────────────────────────────────────────────────
def plot_reconstruction_samples(checkpoint_path, stage1_checkpoint, datasets_dir,
                                output_dir, n_samples=8, device="cuda"):
    """Load model + data, run forward pass, and display originals vs reconstructions."""
    import torch
    from torch.utils.data import DataLoader

    # Add project paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.dirname(__file__))

    from tvl_enc.tacvis import TacVisDataset, RGB_AUGMENTS, TAC_AUGMENTS
    from tvl_enc.tvl import ModalityType
    from models.cross_modal_alignment import CrossModalAlignmentModel, Stage2Wrapper
    from models.reconstruction_decoder import ReconstructionDecoder
    from tvl_enc.tvl import TVL

    # Build model
    frozen_encoder = TVL(
        tactile_model="vit_tiny_patch16_224",
        active_modalities=[ModalityType.VISION, ModalityType.TACTILE],
    )
    if stage1_checkpoint and os.path.exists(stage1_checkpoint):
        ckpt = torch.load(stage1_checkpoint, map_location="cpu")
        if "model" in ckpt:
            frozen_encoder.load_state_dict(ckpt["model"], strict=False)
        else:
            frozen_encoder.load_state_dict(ckpt, strict=False)

    alignment_model = CrossModalAlignmentModel(
        modality_configs={
            ModalityType.VISION: {"input_dim": 768, "feature_type": "pooled"},
            ModalityType.TACTILE: {"input_dim": 768, "feature_type": "pooled"},
        },
        hidden_dim=512, n_registers=32, n_shared=8,
        n_layers=4, n_heads=8,
    )
    model = Stage2Wrapper(frozen_encoder, alignment_model)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    alignment_state = ckpt.get("alignment_model", {})
    model.alignment_model.load_state_dict(alignment_state, strict=False)
    model.to(device)
    model.eval()

    # Build decoders
    recon_decoders = {}
    for mod_name in [ModalityType.VISION, ModalityType.TACTILE]:
        dec = ReconstructionDecoder(
            n_registers=32, hidden_dim=512,
            base_channels=64, n_decoder_layers=2, n_heads=8,
        )
        key = f"recon_decoder_{mod_name}"
        if key in ckpt:
            dec.load_state_dict(ckpt[key])
            print(f"Loaded decoder for {mod_name}")
        dec.to(device)
        dec.eval()
        recon_decoders[mod_name] = dec

    # Load val data
    root_dir = os.path.join(datasets_dir, "ssvtp")
    dataset_val = TacVisDataset(
        root_dir=root_dir, split="val",
        transform_rgb=RGB_AUGMENTS, transform_tac=TAC_AUGMENTS,
        modality_types=[ModalityType.VISION, ModalityType.TACTILE],
        text_prompt="This image gives tactile feelings of ",
    )
    loader = DataLoader(dataset_val, batch_size=n_samples, shuffle=True, num_workers=2)
    batch = next(iter(loader))

    # Move to device
    for k, v in batch.items():
        if isinstance(v, list):
            v = v[0]
        batch[k] = v.to(device, non_blocking=True).squeeze()

    # Forward pass
    with torch.no_grad(), torch.cuda.amp.autocast():
        frozen_features = model.frozen_encoder(batch)
        frozen_features.pop("logit_scale", None)
        frozen_features.pop("logit_bias", None)
        alignment_output = model.alignment_model(frozen_features)

        recon_vision = recon_decoders[ModalityType.VISION](
            alignment_output[f"{ModalityType.VISION}_all_tokens"]
        )
        recon_tactile = recon_decoders[ModalityType.TACTILE](
            alignment_output[f"{ModalityType.TACTILE}_all_tokens"]
        )

    # Unnormalize
    orig_vision = torch.stack([
        unnormalize(batch[ModalityType.VISION][i].float(), RGB_MEAN, RGB_STD)
        for i in range(n_samples)
    ])
    orig_tactile = torch.stack([
        unnormalize(batch[ModalityType.TACTILE][i].float(), TAC_MEAN, TAC_STD)
        for i in range(n_samples)
    ])
    rec_vision = torch.stack([
        unnormalize(recon_vision[i].float(), RGB_MEAN, RGB_STD)
        for i in range(n_samples)
    ])
    rec_tactile = torch.stack([
        unnormalize(recon_tactile[i].float(), TAC_MEAN, TAC_STD)
        for i in range(n_samples)
    ])

    # Fix 90-degree rotation in tactile reconstructions
    rec_tactile = fix_tactile_orientation(rec_tactile)

    # Plot: 4 rows x n_samples columns
    # Row 1: Original vision
    # Row 2: Reconstructed vision
    # Row 3: Original tactile
    # Row 4: Reconstructed tactile
    fig, axes = plt.subplots(4, n_samples, figsize=(2.5 * n_samples, 10))
    fig.suptitle("FlexTok Reconstruction: Original vs Decoded", fontsize=16, fontweight="bold")

    row_labels = ["Original Vision", "Reconstructed Vision",
                  "Original Tactile", "Reconstructed Tactile"]
    images = [orig_vision, rec_vision, orig_tactile, rec_tactile]

    for row_idx in range(4):
        for col_idx in range(n_samples):
            ax = axes[row_idx, col_idx]
            img = images[row_idx][col_idx].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=11, fontweight="bold", rotation=90,
                              labelpad=10)
                ax.yaxis.set_label_position("left")

    plt.tight_layout()
    path = os.path.join(output_dir, "reconstruction_samples.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Also make a zoomed-in side-by-side for 3 best samples
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.suptitle("Reconstruction Close-Up (3 Samples)", fontsize=14, fontweight="bold")
    for i in range(3):
        # Vision
        axes[0, 2*i].imshow(np.clip(orig_vision[i].cpu().permute(1, 2, 0).numpy(), 0, 1))
        axes[0, 2*i].set_title("Original" if i == 0 else "", fontsize=10)
        axes[0, 2*i].axis("off")
        axes[0, 2*i+1].imshow(np.clip(rec_vision[i].cpu().permute(1, 2, 0).numpy(), 0, 1))
        axes[0, 2*i+1].set_title("Reconstructed" if i == 0 else "", fontsize=10)
        axes[0, 2*i+1].axis("off")
        # Tactile
        axes[1, 2*i].imshow(np.clip(orig_tactile[i].cpu().permute(1, 2, 0).numpy(), 0, 1))
        axes[1, 2*i].axis("off")
        axes[1, 2*i+1].imshow(np.clip(rec_tactile[i].cpu().permute(1, 2, 0).numpy(), 0, 1))
        axes[1, 2*i+1].axis("off")
    axes[0, 0].set_ylabel("Vision", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Tactile", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "reconstruction_closeup.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────
# Plot 3: t-SNE of shared embeddings
# ─────────────────────────────────────────────────
def plot_tsne(checkpoint_path, stage1_checkpoint, datasets_dir,
              output_dir, n_samples=200, device="cuda"):
    """Compute shared embeddings for vision + tactile, plot t-SNE."""
    import torch
    from torch.utils.data import DataLoader
    from sklearn.manifold import TSNE

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.dirname(__file__))

    from tvl_enc.tacvis import TacVisDataset, RGB_AUGMENTS, TAC_AUGMENTS
    from tvl_enc.tvl import ModalityType, TVL
    from models.cross_modal_alignment import CrossModalAlignmentModel, Stage2Wrapper

    # Build model
    frozen_encoder = TVL(
        tactile_model="vit_tiny_patch16_224",
        active_modalities=[ModalityType.VISION, ModalityType.TACTILE],
    )
    if stage1_checkpoint and os.path.exists(stage1_checkpoint):
        ckpt = torch.load(stage1_checkpoint, map_location="cpu")
        if "model" in ckpt:
            frozen_encoder.load_state_dict(ckpt["model"], strict=False)
        else:
            frozen_encoder.load_state_dict(ckpt, strict=False)

    alignment_model = CrossModalAlignmentModel(
        modality_configs={
            ModalityType.VISION: {"input_dim": 768, "feature_type": "pooled"},
            ModalityType.TACTILE: {"input_dim": 768, "feature_type": "pooled"},
        },
        hidden_dim=512, n_registers=32, n_shared=8,
        n_layers=4, n_heads=8,
    )
    model = Stage2Wrapper(frozen_encoder, alignment_model)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.alignment_model.load_state_dict(ckpt.get("alignment_model", {}), strict=False)
    model.to(device)
    model.eval()

    # Load data
    root_dir = os.path.join(datasets_dir, "ssvtp")
    dataset_val = TacVisDataset(
        root_dir=root_dir, split="val",
        transform_rgb=RGB_AUGMENTS, transform_tac=TAC_AUGMENTS,
        modality_types=[ModalityType.VISION, ModalityType.TACTILE],
        text_prompt="This image gives tactile feelings of ",
    )
    loader = DataLoader(dataset_val, batch_size=32, shuffle=True, num_workers=2)

    # Collect embeddings
    vision_embeds = []
    tactile_embeds = []
    collected = 0
    for batch in loader:
        if collected >= n_samples:
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

        vision_embeds.append(output[f"{ModalityType.VISION}_shared"].float().cpu())
        tactile_embeds.append(output[f"{ModalityType.TACTILE}_shared"].float().cpu())
        collected += batch[ModalityType.VISION].shape[0]

    vision_embeds = torch.cat(vision_embeds, dim=0)[:n_samples].numpy()
    tactile_embeds = torch.cat(tactile_embeds, dim=0)[:n_samples].numpy()

    # t-SNE
    all_embeds = np.concatenate([vision_embeds, tactile_embeds], axis=0)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(all_embeds)

    n = len(vision_embeds)
    vis_coords = coords[:n]
    tac_coords = coords[n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(vis_coords[:, 0], vis_coords[:, 1], c="dodgerblue", alpha=0.6,
               s=40, label="Vision", edgecolors="navy", linewidth=0.5)
    ax.scatter(tac_coords[:, 0], tac_coords[:, 1], c="orangered", alpha=0.6,
               s=40, label="Tactile", edgecolors="darkred", linewidth=0.5)

    # Draw lines between matched pairs
    for i in range(min(n, 50)):  # Draw first 50 pair connections
        ax.plot([vis_coords[i, 0], tac_coords[i, 0]],
                [vis_coords[i, 1], tac_coords[i, 1]],
                "gray", alpha=0.15, linewidth=0.5)

    ax.set_title("t-SNE of Shared Embeddings (Vision vs Tactile)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(output_dir, "tsne_embeddings.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────
# Plot 4: Variable-length token evaluation
# ─────────────────────────────────────────────────
def plot_variable_length(checkpoint_path, stage1_checkpoint, datasets_dir,
                         output_dir, device="cuda"):
    """Evaluate contrastive accuracy using only first K register tokens."""
    import torch
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.dirname(__file__))

    from tvl_enc.tacvis import TacVisDataset, RGB_AUGMENTS, TAC_AUGMENTS
    from tvl_enc.tvl import ModalityType, TVL
    from models.cross_modal_alignment import CrossModalAlignmentModel, Stage2Wrapper

    # Build model
    frozen_encoder = TVL(
        tactile_model="vit_tiny_patch16_224",
        active_modalities=[ModalityType.VISION, ModalityType.TACTILE],
    )
    if stage1_checkpoint and os.path.exists(stage1_checkpoint):
        ckpt = torch.load(stage1_checkpoint, map_location="cpu")
        if "model" in ckpt:
            frozen_encoder.load_state_dict(ckpt["model"], strict=False)
        else:
            frozen_encoder.load_state_dict(ckpt, strict=False)

    alignment_model = CrossModalAlignmentModel(
        modality_configs={
            ModalityType.VISION: {"input_dim": 768, "feature_type": "pooled"},
            ModalityType.TACTILE: {"input_dim": 768, "feature_type": "pooled"},
        },
        hidden_dim=512, n_registers=32, n_shared=8,
        n_layers=4, n_heads=8,
    )
    model = Stage2Wrapper(frozen_encoder, alignment_model)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.alignment_model.load_state_dict(ckpt.get("alignment_model", {}), strict=False)
    model.to(device)
    model.eval()

    # Load data
    root_dir = os.path.join(datasets_dir, "ssvtp")
    dataset_val = TacVisDataset(
        root_dir=root_dir, split="val",
        transform_rgb=RGB_AUGMENTS, transform_tac=TAC_AUGMENTS,
        modality_types=[ModalityType.VISION, ModalityType.TACTILE],
        text_prompt="This image gives tactile feelings of ",
    )
    loader = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=2)

    # Evaluate with different numbers of shared tokens
    # FlexTok nested dropout trains with powers of 2: 1, 2, 4, 8
    # n_shared = 8, so valid prefix lengths are 1, 2, 4, 8
    k_values = [1, 2, 4, 8]
    results = {k: {"correct": 0, "total": 0} for k in k_values}

    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, list):
                v = v[0]
            batch[k] = v.to(device, non_blocking=True).squeeze()
        bs = batch[ModalityType.VISION].shape[0]

        with torch.no_grad(), torch.cuda.amp.autocast():
            frozen_features = model.frozen_encoder(batch)
            frozen_features.pop("logit_scale", None)
            frozen_features.pop("logit_bias", None)
            output = model.alignment_model(frozen_features)

        vis_shared = output[f"{ModalityType.VISION}_shared_tokens"].float()  # (B, 8, 512)
        tac_shared = output[f"{ModalityType.TACTILE}_shared_tokens"].float()

        for k in k_values:
            # Use only first k tokens, pool, project, normalize
            vis_pooled = vis_shared[:, :k, :].mean(dim=1)
            tac_pooled = tac_shared[:, :k, :].mean(dim=1)

            vis_proj = model.alignment_model.shared_projectors[ModalityType.VISION](vis_pooled)
            tac_proj = model.alignment_model.shared_projectors[ModalityType.TACTILE](tac_pooled)
            vis_proj = F.normalize(vis_proj, dim=-1)
            tac_proj = F.normalize(tac_proj, dim=-1)

            logit_scale = output["logit_scale"]
            logits = logit_scale * vis_proj @ tac_proj.T  # (B, B)
            preds = logits.argmax(dim=1)
            targets = torch.arange(bs, device=device)
            correct = (preds == targets).sum().item()

            results[k]["correct"] += correct
            results[k]["total"] += bs

    accuracies = {k: 100 * results[k]["correct"] / results[k]["total"] for k in k_values}

    # Plot bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([str(k) for k in k_values],
                  [accuracies[k] for k in k_values],
                  color=["#2196F3", "#4CAF50", "#FF9800", "#F44336"],
                  edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, k in zip(bars, k_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                f"{accuracies[k]:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_xlabel("Number of Shared Tokens Used", fontsize=12)
    ax.set_ylabel("Val Acc@1 (%)", fontsize=12)
    ax.set_title("Variable-Length Token Evaluation\n(FlexTok: using only first K shared tokens)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(accuracies.values()) + 10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "variable_length_eval.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    print("Variable-length results:")
    for k in k_values:
        print(f"  K={k}: {accuracies[k]:.1f}%")


# ─────────────────────────────────────────────────
# Plot 5: Summary comparison table
# ─────────────────────────────────────────────────
def plot_summary_table(contrastive_log, recon_log, output_dir):
    """Create a visual summary table comparing the two training runs."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    # Gather final metrics
    rows = []
    if contrastive_log:
        c_data = parse_log(contrastive_log)
        final = c_data[-1]
        best_val = max(d.get("val_acc1", 0) for d in c_data)
        rows.append([
            "Contrastive Only",
            str(len(c_data)),
            f"{final.get('train_acc1', final.get('train_acc1_avg', 0)):.1f}%",
            f"{best_val:.1f}%",
            f"{final.get('train_loss', 0):.3f}",
            "N/A",
        ])
    if recon_log:
        r_data = parse_log(recon_log)
        final = r_data[-1]
        best_val = max(d.get("val_acc1", 0) for d in r_data)
        recon_total = final.get("val_recon_total", final.get("train_recon_total", 0))
        rows.append([
            "Joint (+ Recon)",
            str(len(r_data)),
            f"{final.get('train_acc1', final.get('train_acc1_avg', 0)):.1f}%",
            f"{best_val:.1f}%",
            f"{final.get('train_loss', 0):.3f}",
            f"{recon_total:.3f}" if recon_total else "N/A",
        ])

    columns = ["Run", "Epochs", "Train Acc@1", "Best Val Acc@1", "Final Loss", "Recon Loss"]

    table = ax.table(cellText=rows, colLabels=columns, loc="center",
                     cellLoc="center", colColours=["#4CAF50"] * len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 2.0)

    # Style header
    for j in range(len(columns)):
        table[(0, j)].set_text_props(fontweight="bold", color="white")

    # Highlight best val acc
    if len(rows) == 2:
        val_accs = [float(r[3].replace("%", "")) for r in rows]
        best_idx = val_accs.index(max(val_accs))
        table[(best_idx + 1, 3)].set_facecolor("#C8E6C9")

    ax.set_title("FlexTok Stage 2 Results Summary (SSVTP)", fontsize=14,
                 fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(output_dir, "results_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FlexTok Stage 2 Visualization")

    # Log files (no GPU needed)
    parser.add_argument("--contrastive_log", type=str, default=None,
                        help="Path to contrastive-only training log.txt")
    parser.add_argument("--recon_log", type=str, default=None,
                        help="Path to joint training (with recon) log.txt")

    # Checkpoint + data (needs GPU)
    parser.add_argument("--recon_checkpoint", type=str, default=None,
                        help="Path to reconstruction checkpoint (checkpoint_best.pth)")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Path to Stage 1 encoder checkpoint")
    parser.add_argument("--datasets_dir", type=str, default=None,
                        help="Root data directory (parent of ssvtp/)")

    # What to generate
    parser.add_argument("--sample_reconstructions", action="store_true",
                        help="Generate reconstruction sample images")
    parser.add_argument("--tsne", action="store_true",
                        help="Generate t-SNE embedding plot")
    parser.add_argument("--variable_length", action="store_true",
                        help="Run variable-length token evaluation")
    parser.add_argument("--n_samples", type=int, default=8,
                        help="Number of samples for reconstruction grid")

    # Output
    parser.add_argument("--output_dir", type=str, default="visualizations/",
                        help="Directory to save plots")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Always generate training curves and summary if logs are available
    if args.contrastive_log or args.recon_log:
        print("=" * 50)
        print("Generating training curves...")
        plot_training_curves(args.contrastive_log, args.recon_log, args.output_dir)
        print("Generating summary table...")
        plot_summary_table(args.contrastive_log, args.recon_log, args.output_dir)

    # GPU-dependent visualizations
    if args.sample_reconstructions:
        assert args.recon_checkpoint, "Need --recon_checkpoint for reconstruction samples"
        assert args.datasets_dir, "Need --datasets_dir for reconstruction samples"
        print("=" * 50)
        print("Generating reconstruction samples...")
        plot_reconstruction_samples(
            args.recon_checkpoint, args.stage1_checkpoint, args.datasets_dir,
            args.output_dir, n_samples=args.n_samples, device=args.device,
        )

    if args.tsne:
        assert args.recon_checkpoint, "Need --recon_checkpoint for t-SNE"
        assert args.datasets_dir, "Need --datasets_dir for t-SNE"
        print("=" * 50)
        print("Generating t-SNE plot...")
        plot_tsne(
            args.recon_checkpoint, args.stage1_checkpoint, args.datasets_dir,
            args.output_dir, device=args.device,
        )

    if args.variable_length:
        assert args.recon_checkpoint, "Need --recon_checkpoint for variable-length eval"
        assert args.datasets_dir, "Need --datasets_dir for variable-length eval"
        print("=" * 50)
        print("Running variable-length token evaluation...")
        plot_variable_length(
            args.recon_checkpoint, args.stage1_checkpoint, args.datasets_dir,
            args.output_dir, device=args.device,
        )

    print("=" * 50)
    print(f"All visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
