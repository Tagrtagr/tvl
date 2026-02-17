"""
Stage 2 Training Script: Cross-Modality Alignment with Register Tokens.
Approach: FlexTok-inspired register tokens (Claude Code)

Trains learnable register token modules on top of frozen Stage-1 encoders
to align vision and tactile representations in a shared latent space.

Usage:
    # Single GPU
    python experiments/stage2/claude_flextok/train.py \
        --stage1_checkpoint /path/to/stage1/checkpoint.pth \
        --datasets_dir /path/to/datasets \
        --output_dir ./output/stage2

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 experiments/stage2/claude_flextok/train.py \
        --stage1_checkpoint /path/to/stage1/checkpoint.pth \
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
)

from models.cross_modal_alignment import CrossModalAlignmentModel, Stage2Wrapper
from losses.alignment_loss import CrossModalAlignmentLoss
from losses.flow_matching import FlowMatchingAlignmentLoss

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


def build_model(args, device):
    """Build frozen encoder + Stage 2 alignment model."""
    # Build frozen Stage 1 encoder
    frozen_encoder = TVL(
        tactile_model=args.tactile_model,
        active_modalities=[ModalityType.VISION, ModalityType.TACTILE],
    )

    # Load Stage 1 checkpoint if provided
    if args.stage1_checkpoint and os.path.exists(args.stage1_checkpoint):
        print(f"Loading Stage 1 checkpoint from {args.stage1_checkpoint}")
        ckpt = torch.load(args.stage1_checkpoint, map_location="cpu")
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
        ModalityType.VISION: {"input_dim": vision_dim, "feature_type": "pooled"},
        ModalityType.TACTILE: {"input_dim": tactile_dim, "feature_type": "pooled"},
    }

    # Build Stage 2 alignment model
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
):
    model.train()
    # Keep frozen encoder in eval mode
    model_inner = _unwrap(model)
    model_inner.frozen_encoder.eval()

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

        with torch.cuda.amp.autocast():
            # Get frozen features
            with torch.no_grad():
                frozen_features = model_inner.encode(samples)

            # Forward through alignment model
            alignment_output = model_inner.alignment_model(frozen_features)

            # Contrastive + preservation loss
            preservation_projectors = model_inner.alignment_model.preservation_projectors
            loss_dict = contrastive_loss_fn(
                alignment_output,
                frozen_features=frozen_features,
                preservation_projectors=preservation_projectors,
                output_dict=True,
            )
            loss = loss_dict["total_loss"]

            # Flow matching loss (if enabled)
            if flow_loss_fn is not None:
                flow_dict = flow_loss_fn(alignment_output, output_dict=True)
                flow_loss = flow_dict.get("flow_total", torch.tensor(0.0, device=device))
                loss = loss + args.flow_weight * flow_loss
                loss_dict["flow_loss"] = flow_loss

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter

        # Collect trainable parameters
        trainable_params = list(model_inner.alignment_model.parameters())
        if flow_loss_fn is not None:
            trainable_params += list(flow_loss_fn.parameters())

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
):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"
    model.eval()
    model_inner = _unwrap(model)

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

        loss = loss_dict["total_loss"]
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

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

    # Build model
    model = build_model(args, device)
    model_without_ddp = model

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

    # Optimizer (only optimize Stage 2 parameters)
    alignment_params = list(model_without_ddp.alignment_model.parameters())

    # Build loss
    contrastive_loss_fn, flow_loss_fn = build_loss(args, device)
    if flow_loss_fn is not None:
        alignment_params += list(flow_loss_fn.parameters())

    # Effective batch size for LR scaling
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"actual lr: {args.lr:.2e}")
    print(f"effective batch size: {eff_batch_size}")

    optimizer = torch.optim.AdamW(alignment_params, lr=args.lr, betas=(0.9, 0.95),
                                   weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()

    # Resume
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        model_without_ddp.alignment_model.load_state_dict(ckpt.get("alignment_model", {}), strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            loss_scaler.load_state_dict(ckpt["scaler"])
        args.start_epoch = ckpt.get("epoch", 0) + 1

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_acc1 = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, contrastive_loss_fn, flow_loss_fn, data_loader_train,
            optimizer, device, epoch, loss_scaler, args, log_writer=log_writer,
        )

        val_stats = evaluate(data_loader_val, contrastive_loss_fn, model, device,
                             epoch=epoch, log_writer=log_writer)

        acc1 = val_stats.get("acc1", 0.0)
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

            if acc1 > best_acc1:
                best_acc1 = acc1
                if misc.is_main_process():
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
        # Config overrides defaults but CLI overrides config
        # (simple merge for now)
        for section in config.values():
            if isinstance(section, dict):
                for k, v in section.items():
                    if hasattr(args, k) and getattr(args, k) is None:
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
