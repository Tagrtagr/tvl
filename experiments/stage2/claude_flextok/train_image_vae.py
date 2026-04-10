"""
Train a simple VAE on Stage-2 datasets for a selected modality.

Usage:
    # Train on RGB images only
    python experiments/stage2/claude_flextok/train_image_vae.py \
        --vae_modality vision \
        --datasets_dir /path/to/datasets \
        --datasets ssvtp hct \
        --output_dir ./output/vae_vision

    # Train on tactile images only
    python experiments/stage2/claude_flextok/train_image_vae.py \
        --vae_modality tactile \
        --datasets_dir /path/to/datasets \
        --datasets ssvtp hct \
        --output_dir ./output/vae_tactile
"""

import datetime
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torchvision.utils import save_image

import sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "../../.."))

from tvl_enc.tvl import ModalityType
from tvl_enc.tacvis import (
    TacVisDataset, TacVisDatasetV2,
    RGB_AUGMENTS, TAC_AUGMENTS, TAC_AUGMENTS_BG,
    RGB_MEAN, RGB_STD, TAC_MEAN, TAC_STD, TAC_MEAN_BG, TAC_STD_BG,
)
from models.simple_vae import ConvVAE



def _unnormalize(x: torch.Tensor, mod_name: str, args) -> torch.Tensor:
    if mod_name == ModalityType.VISION:
        mean = torch.tensor(RGB_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(RGB_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    else:
        if args.subtract_background == "background":
            mean_np = TAC_MEAN_BG
            std_np = TAC_STD_BG
        else:
            mean_np = TAC_MEAN
            std_np = TAC_STD
        mean = torch.tensor(mean_np, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(std_np, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return x * std + mean


def build_datasets(args):
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


def kld_loss(mu, logvar):
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def step_vae(model, x, kl_weight):
    recon, mu, logvar = model(x)
    rec_loss = F.mse_loss(recon, x)
    kl = kld_loss(mu, logvar)
    loss = rec_loss + kl_weight * kl
    return loss, rec_loss, kl, recon


def _extract_modality_batch(batch, modality):
    """
    Return a tensor batch for a modality from collated TacVis samples.
    TacVis datasets wrap image tensors in a singleton list, so DataLoader
    often yields `batch[modality]` as a list/tuple instead of a tensor.
    """
    if modality not in batch:
        return None

    value = batch[modality]
    if isinstance(value, torch.Tensor):
        return value

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        if len(value) == 1 and isinstance(value[0], torch.Tensor):
            return value[0]
        if all(isinstance(v, torch.Tensor) for v in value):
            return torch.stack(list(value), dim=0)
    return None


def main(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    vae_modality = ModalityType.VISION if args.vae_modality == "vision" else ModalityType.TACTILE
    print(f"Training VAE modality: {vae_modality}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_train, dataset_val = build_datasets(args)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    model = ConvVAE(
        in_channels=3,
        latent_channels=args.latent_channels,
        scaling_factor=args.vae_scaling_factor,
        hf_hub_path=args.vae_hf_hub_path,
        sample_posterior=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_rec = 0.0
        train_kl = 0.0
        n_train = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x = _extract_modality_batch(batch, vae_modality)
            if x is None:
                continue

            x = x.to(device, non_blocking=True)
            loss_acc, rec_acc, kl_acc, _ = step_vae(model, x, args.kl_weight)

            loss_acc.backward()
            optimizer.step()

            bsz = x.shape[0]
            train_loss += loss_acc.item() * bsz
            train_rec += rec_acc.item() * bsz
            train_kl += kl_acc.item() * bsz
            n_train += bsz

        model.eval()
        val_loss = 0.0
        val_rec = 0.0
        val_kl = 0.0
        n_val = 0

        with torch.no_grad():
            for b_idx, batch in enumerate(val_loader):
                x = _extract_modality_batch(batch, vae_modality)
                if x is None:
                    continue

                x = x.to(device, non_blocking=True)
                loss_acc, rec_acc, kl_acc, recon = step_vae(model, x, args.kl_weight)
                bsz = x.shape[0]

                if b_idx == 0 and epoch % max(args.save_every, 1) == 0:
                    vis_dir = Path(args.output_dir) / "reconstructions"
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    x_disp = torch.clamp(_unnormalize(x, vae_modality, args), 0.0, 1.0)
                    r_disp = torch.clamp(_unnormalize(recon, vae_modality, args), 0.0, 1.0)
                    save_image(x_disp[:16], vis_dir / f"epoch_{epoch:04d}_{vae_modality}_target.png", nrow=4)
                    save_image(r_disp[:16], vis_dir / f"epoch_{epoch:04d}_{vae_modality}_recon.png", nrow=4)

                val_loss += loss_acc.item() * bsz
                val_rec += rec_acc.item() * bsz
                val_kl += kl_acc.item() * bsz
                n_val += bsz

        train_stats = {
            "loss": train_loss / max(n_train, 1),
            "recon": train_rec / max(n_train, 1),
            "kl": train_kl / max(n_train, 1),
        }
        val_stats = {
            "loss": val_loss / max(n_val, 1),
            "recon": val_rec / max(n_val, 1),
            "kl": val_kl / max(n_val, 1),
        }

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"train_loss={train_stats['loss']:.6f} "
            f"val_loss={val_stats['loss']:.6f} "
            f"val_recon={val_stats['recon']:.6f} "
            f"val_kl={val_stats['kl']:.6f}"
        )

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "train": train_stats,
            "val": val_stats,
        }
        torch.save(ckpt, Path(args.output_dir) / "checkpoint_latest.pth")
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(ckpt, Path(args.output_dir) / "checkpoint_best.pth")

        with open(Path(args.output_dir) / "log.txt", "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in val_stats.items()}}) + "\n")

    elapsed = time.time() - start
    print(f"Done. Total time: {str(datetime.timedelta(seconds=int(elapsed)))}")


def _cfg_to_args(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return SimpleNamespace(**cfg_dict)


@hydra.main(config_path="configs", config_name="config_image_vae", version_base="1.3")
def hydra_main(cfg: DictConfig):
    args = _cfg_to_args(cfg)
    main(args)


if __name__ == "__main__":
    hydra_main()
