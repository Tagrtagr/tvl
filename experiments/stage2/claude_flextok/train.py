"""
Stage 2 Training: Cross-Modality Alignment with FlexTok Register Tokens.

Two-stage protocol (following FlexTok, Bachmann et al. 2025):
  stage=alignment    Train register tokens + autoregressive probe jointly.
  stage=reconstruction  Freeze alignment, train autoregressive decoder.

Launch (Hydra config overrides use key=value syntax):
  python experiments/stage2/claude_flextok/train.py \\
      stage1_checkpoint=/path/to/tvl.pth datasets_dir=/path/to/data

  # Stage 2b: frozen encoder, AR decoder
  python experiments/stage2/claude_flextok/train.py \\
      stage=reconstruction model=reconstruction losses=reconstruction \\
      alignment_checkpoint=./output/stage2a/best.pth ...

  # Multi-GPU
  torchrun --nproc_per_node=4 experiments/stage2/claude_flextok/train.py ...

  # Resume
  python experiments/stage2/claude_flextok/train.py resume=auto ...
"""

import warnings
warnings.filterwarnings("ignore")

import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "../../.."))
sys.path.insert(0, os.path.join(_here, "../../../tvl_enc"))

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched

from tvl_enc.tvl import ModalityType
from tvl_enc.tacvis import RGB_MEAN, RGB_STD, TAC_MEAN, TAC_STD, TAC_MEAN_BG, TAC_STD_BG

from losses.flow_reconstruction_loss import FlowReconstructionLoss
from models.flow_matching_decoder import FlowMatchingReconstructionDecoder
from pipeline.stage2_pipeline import DecoderCollection

from build import build_datasets, build_pipeline, build_decoders, build_losses

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _unnormalize_for_display(x: torch.Tensor, mod_name: str, subtract_bg: str) -> torch.Tensor:
    """Map dataset-normalized tensor back to [0, 1]-ish pixel space."""
    if mod_name == ModalityType.VISION:
        mean_np, std_np = RGB_MEAN, RGB_STD
    elif mod_name == ModalityType.TACTILE:
        mean_np, std_np = (TAC_MEAN_BG, TAC_STD_BG) if subtract_bg == "background" else (TAC_MEAN, TAC_STD)
    else:
        return x
    mean = torch.tensor(mean_np, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std  = torch.tensor(std_np,  device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return x * std + mean


def _compute_recon_loss(cfg, alignment_output, samples, recon_decoders, recon_loss_fn):
    """Gather tokens/targets and call the reconstruction loss.

    Handles both FlowReconstructionLoss (needs unnormalized pixel targets)
    and pixel-space losses (take normalized inputs directly).
    Returns the loss dict, or {} if no modality tokens are available.
    """
    is_flow = isinstance(recon_loss_fn, FlowReconstructionLoss)
    all_tokens, targets, prefix_tokens = {}, {}, {}
    for mod in [ModalityType.VISION, ModalityType.TACTILE]:
        if f"{mod}_all_tokens" not in alignment_output or mod not in samples:
            continue
        all_tokens[mod] = alignment_output[f"{mod}_all_tokens"]
        if f"{mod}_prefix_tokens" in alignment_output:
            prefix_tokens[mod] = alignment_output[f"{mod}_prefix_tokens"]
        targets[mod] = (
            _unnormalize_for_display(samples[mod], mod, cfg.subtract_background)
            if is_flow else samples[mod]
        )
    if not all_tokens:
        return {}
    if is_flow:
        return recon_loss_fn(all_tokens, targets, recon_decoders, output_dict=True,
                             prefix_tokens=prefix_tokens)
    recons = {m: recon_decoders[m](all_tokens[m]) for m in all_tokens}
    return recon_loss_fn(recons, targets)


def _save_recon_images(batch_idx, alignment_output, batch, recon_decoders,
                       output_dir, epoch, stage, cfg):
    if (recon_decoders is None or output_dir is None or epoch is None
            or not cfg.save_recon_images or cfg.recon_save_every <= 0
            or epoch % cfg.recon_save_every != 0
            or batch_idx >= cfg.recon_save_max_batches):
        return

    root = Path(output_dir) / "reconstructions" / stage / f"epoch_{epoch:04d}" / f"batch_{batch_idx:03d}"
    root.mkdir(parents=True, exist_ok=True)

    for mod in [ModalityType.VISION, ModalityType.TACTILE]:
        if f"{mod}_all_tokens" not in alignment_output or mod not in batch:
            continue
        tokens = alignment_output[f"{mod}_all_tokens"]
        decoder = recon_decoders.get(mod)
        if decoder is None:
            continue
        is_flow_dec = isinstance(_unwrap(decoder), FlowMatchingReconstructionDecoder)
        if is_flow_dec:
            recons = _unwrap(decoder).sample(tokens, n_steps=cfg.flow_recon_steps)
            # Flow decoder outputs are already in [0,1] pixel space (trained on unnormalized targets)
            recons = torch.clamp(recons, 0.0, 1.0)
        else:
            recons = decoder(tokens)
            recons = torch.clamp(_unnormalize_for_display(recons, mod, cfg.subtract_background), 0.0, 1.0)
        targets = torch.clamp(_unnormalize_for_display(batch[mod], mod, cfg.subtract_background), 0.0, 1.0)
        save_image(targets, root / f"{mod}_target.png", nrow=8)
        save_image(recons,  root / f"{mod}_recon.png",  nrow=8)


def _resolve_resume_checkpoint(resume, output_dir: str) -> Optional[str]:
    resume = str(resume or "").strip()
    if not resume:
        return None
    if os.path.isfile(resume):
        return resume
    if resume.lower() in {"1", "true", "yes", "y", "auto"}:
        for name in ("checkpoint_latest.pth", "checkpoint_best.pth"):
            p = os.path.join(output_dir, name)
            if os.path.isfile(p):
                return p
        print(f"[resume] no checkpoint found in {output_dir}")
    return None


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, contrastive_loss_fn, data_loader,
                    optimizer, device, epoch, loss_scaler, cfg, sched,
                    log_writer=None, recon_decoders=None, recon_loss_fn=None,
                    has_alignment_loss=True):
    model.train()
    model_inner = _unwrap(model)
    model_inner.modality_encoder.eval()
    if recon_decoders:
        for dec in recon_decoders.values():
            dec.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    accum_iter = cfg.accum_iter
    stage = cfg.stage

    # Disable nested dropout when there is no alignment objective:
    # the decoder needs fully-populated tokens to learn fine details.
    _apply_nd = False if (stage == "reconstruction" or not has_alignment_loss) else None

    # Loss weight for the decoder (probe during alignment, full decoder during reconstruction)
    decoder_weight = (
        cfg.losses.probe.weight if stage == "alignment"
        else cfg.losses.recon_decoder.weight
    )

    optimizer.zero_grad()
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, 20, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, sched)

        for k, v in samples.items():
            samples[k] = (v[0] if isinstance(v, list) else v).to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                frozen_features = model_inner.encode(samples)

            if stage == "reconstruction":
                with torch.no_grad():
                    alignment_output = model_inner.cross_modality_encoder(
                        frozen_features, apply_nested_dropout=False)
            else:
                alignment_output = model_inner.cross_modality_encoder(
                    frozen_features, apply_nested_dropout=_apply_nd)

            loss = torch.tensor(0.0, device=device)
            loss_dict = {}

            if stage != "reconstruction" and contrastive_loss_fn is not None:
                preservation_projectors = model_inner.cross_modality_encoder.preservation_projectors
                align_dict = contrastive_loss_fn(
                    alignment_output, frozen_features=frozen_features,
                    preservation_projectors=preservation_projectors,
                    input_images=samples, output_dict=True,
                )
                loss = loss + cfg.losses.contrastive.weight * align_dict["total_loss"]
                loss_dict.update(align_dict)

            if recon_decoders is not None and recon_loss_fn is not None:
                recon_dict = _compute_recon_loss(cfg, alignment_output, samples, recon_decoders, recon_loss_fn)
                if recon_dict:
                    loss = loss + decoder_weight * recon_dict["recon_total"]
                    loss_dict.update({k: v for k, v in recon_dict.items() if isinstance(v, torch.Tensor)})

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        trainable = [p for p in (
            list(model_inner.cross_modality_encoder.parameters())
            + [p for dec in (recon_decoders or {}).values() for p in dec.parameters()]
        ) if p.requires_grad]
        loss_scaler(loss, optimizer, parameters=trainable,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value, lr=optimizer.param_groups[0]["lr"])
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                metric_logger.update(**{k: v.item()})

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            t = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value, t)
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    log_writer.add_scalar(f"train_{k}", v.item(), t)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, contrastive_loss_fn, model, device,
             epoch, log_writer, recon_decoders, recon_loss_fn, cfg,
             stage, output_dir, has_alignment_loss):
    metric_logger = misc.MetricLogger(delimiter="  ")
    model.eval()
    model_inner = _unwrap(model)
    if recon_decoders:
        for dec in recon_decoders.values():
            dec.eval()

    _apply_nd = False if (stage == "reconstruction" or not has_alignment_loss) else None
    debug = cfg.debug_single_sample
    decoder_weight = (
        cfg.losses.probe.weight if stage == "alignment"
        else cfg.losses.recon_decoder.weight
    )

    for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, 10, "Test:")):
        batch_size = None
        for k, v in batch.items():
            v = (v[0] if isinstance(v, list) else v)
            if debug and isinstance(v, torch.Tensor):
                v = v[:1]
            batch[k] = v.to(device, non_blocking=True)
            if isinstance(v, torch.Tensor):
                batch_size = v.shape[0]

        with torch.cuda.amp.autocast():
            frozen_features = model_inner.encode(batch)
            alignment_output = model_inner.cross_modality_encoder(
                frozen_features, apply_nested_dropout=_apply_nd)
            preservation_projectors = model_inner.cross_modality_encoder.preservation_projectors

            loss_dict = {}
            alignment_total = torch.tensor(0.0, device=device)

            if contrastive_loss_fn is not None:
                contrastive_dict = contrastive_loss_fn(
                    alignment_output, frozen_features=frozen_features,
                    preservation_projectors=preservation_projectors,
                    input_images=batch, output_dict=True,
                )
                w = cfg.losses.contrastive.weight
                contrastive_dict["total_loss"] = w * contrastive_dict["total_loss"]
                loss_dict.update(contrastive_dict)
                alignment_total = contrastive_dict["total_loss"]

            loss_dict["total_loss"] = alignment_total

            if recon_decoders is not None and recon_loss_fn is not None:
                recon_dict = _compute_recon_loss(cfg, alignment_output, batch, recon_decoders, recon_loss_fn)
                loss_dict.update({k: v for k, v in recon_dict.items() if isinstance(v, torch.Tensor)})

            _save_recon_images(batch_idx, alignment_output, batch, recon_decoders,
                               output_dir, epoch, stage, cfg)

        base  = loss_dict.get("total_loss", torch.tensor(0.0, device=device))
        recon = loss_dict.get("recon_total")
        loss  = base + decoder_weight * recon if recon is not None else base
        metric_logger.update(loss=loss.item())

        for key in ("acc1_avg", "acc5_avg"):
            val = loss_dict.get(key)
            if val is not None and batch_size:
                metric_logger.meters.setdefault(key[:-4], misc.SmoothedValue())
                metric_logger.meters[key[:-4]].update(val.item(), n=batch_size)

        for k, v in loss_dict.items():
            if k not in ("total_loss", "acc1_avg", "acc5_avg") and isinstance(v, torch.Tensor):
                metric_logger.update(**{k: v.item()})

        if debug:
            break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if log_writer is not None and epoch is not None:
        for k, m in metric_logger.meters.items():
            log_writer.add_scalar(f"val_{k}", m.global_avg, epoch)
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run(cfg: DictConfig):
    # Distributed init requires a mutable namespace (misc.init_distributed_mode writes to it).
    dist = SimpleNamespace(
        distributed=False, gpu=None, rank=0,
        dist_on_itp=cfg.dist_on_itp, dist_url=cfg.dist_url,
        world_size=cfg.world_size, local_rank=cfg.local_rank,
    )
    misc.init_distributed_mode(dist)
    torch.backends.cudnn.deterministic = True

    output_dir = os.path.join(cfg.output_dir, cfg.log_name) if cfg.log_name else cfg.output_dir
    log_dir    = cfg.log_dir or output_dir

    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device)
    seed = cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Datasets & dataloaders
    dataset_train, dataset_val = build_datasets(cfg)
    num_tasks, global_rank = misc.get_world_size(), misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    sampler_val = (
        torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks,
                                             rank=global_rank, shuffle=False)
        if cfg.dist_eval else torch.utils.data.SequentialSampler(dataset_val)
    )
    num_workers = cfg.debug_num_workers if cfg.debug_single_sample else cfg.num_workers
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=cfg.batch_size, num_workers=num_workers,
        pin_memory=cfg.pin_mem, drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=cfg.batch_size, num_workers=num_workers,
        pin_memory=cfg.pin_mem, drop_last=False,
    )
    if len(dataset_train) == 1 or cfg.debug_single_sample:
        print("[single-sample] Reusing training dataloader for evaluation")
        data_loader_val = data_loader_train

    # Build model
    model = build_pipeline(cfg, device)
    model_without_ddp = model
    stage = cfg.stage

    if stage == "reconstruction":
        print(f"Loading alignment checkpoint: {cfg.alignment_checkpoint}")
        ckpt = torch.load(cfg.alignment_checkpoint, map_location="cpu")
        state = ckpt.get("cross_modality_encoder", ckpt.get("alignment_model"))
        model_without_ddp.cross_modality_encoder.load_state_dict(state, strict=False)
        for p in model_without_ddp.cross_modality_encoder.parameters():
            p.requires_grad = False
        n_frozen = sum(p.numel() for p in model_without_ddp.cross_modality_encoder.parameters())
        print(f"Froze alignment model ({n_frozen:,} params)")

    # Decoders
    alignment_hidden_dim = model_without_ddp.cross_modality_encoder.hidden_dim
    recon_decoders, recon_loss_fn = build_decoders(cfg, device, alignment_hidden_dim)
    model_without_ddp.decoder_collection = DecoderCollection(recon_decoders)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")

    if dist.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        for mod in list(recon_decoders.keys()):
            recon_decoders[mod] = torch.nn.parallel.DistributedDataParallel(
                recon_decoders[mod], device_ids=[dist.gpu])

    # Alignment losses
    contrastive_loss_fn = build_losses(
        cfg, device, embed_dim=model_without_ddp.cross_modality_encoder.hidden_dim)
    has_alignment_loss = contrastive_loss_fn is not None
    if not has_alignment_loss:
        print("[train] No alignment loss — nested dropout disabled for reconstruction quality")

    # Optimizer
    trainable_params = []
    if stage != "reconstruction":
        trainable_params += list(model_without_ddp.cross_modality_encoder.parameters())
    for dec in recon_decoders.values():
        trainable_params += list(dec.parameters())
    trainable_params = [p for p in trainable_params if p.requires_grad]

    eff_batch = cfg.batch_size * cfg.accum_iter * misc.get_world_size()
    lr = cfg.lr if cfg.lr is not None else cfg.blr * eff_batch / 256
    print(f"lr: {lr:.2e}  (base {cfg.blr:.2e} × batch {eff_batch} / 256)  stage={stage}")

    # sched is a plain namespace so lr_sched.adjust_learning_rate can read lr/warmup/epochs
    sched = SimpleNamespace(lr=lr, min_lr=cfg.min_lr, warmup_epochs=cfg.warmup_epochs, epochs=cfg.epochs)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.95),
                                   weight_decay=cfg.weight_decay)
    loss_scaler = NativeScaler()

    # TensorBoard
    log_writer = None
    if misc.is_main_process() and log_dir and HAS_TENSORBOARD:
        os.makedirs(log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_dir)

    # Resume
    start_epoch = cfg.start_epoch
    resume_ckpt = _resolve_resume_checkpoint(cfg.resume, output_dir)
    if resume_ckpt:
        print(f"Resuming from {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        if stage != "reconstruction":
            state = ckpt.get("cross_modality_encoder", ckpt.get("alignment_model", {}))
            if state:
                msg = model_without_ddp.cross_modality_encoder.load_state_dict(state, strict=False)
                print(f"[resume] alignment (missing={len(msg.missing_keys)}, "
                      f"unexpected={len(msg.unexpected_keys)})")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            loss_scaler.load_state_dict(ckpt["scaler"])
        for mod, dec in recon_decoders.items():
            key = f"recon_decoder_{mod}"
            if key in ckpt:
                msg = dec.load_state_dict(ckpt[key], strict=False)
                print(f"[resume] decoder {mod} (missing={len(msg.missing_keys)}, "
                      f"unexpected={len(msg.unexpected_keys)})")
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[resume] start_epoch={start_epoch}")

    # Training loop
    print(f"Start training for {cfg.epochs} epochs")
    start_time = time.time()
    best_acc1, best_recon_loss = 0.0, float("inf")

    for epoch in range(start_epoch, cfg.epochs):
        if dist.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, contrastive_loss_fn, data_loader_train,
            optimizer, device, epoch, loss_scaler, cfg, sched,
            log_writer=log_writer, recon_decoders=recon_decoders,
            recon_loss_fn=recon_loss_fn, has_alignment_loss=has_alignment_loss,
        )
        val_stats = evaluate(
            data_loader_val, contrastive_loss_fn, model, device,
            epoch=epoch, log_writer=log_writer, recon_decoders=recon_decoders,
            recon_loss_fn=recon_loss_fn, cfg=cfg, stage=stage,
            output_dir=output_dir, has_alignment_loss=has_alignment_loss,
        )

        acc1 = val_stats.get("acc1", 0.0)
        recon_val = val_stats.get("recon_total", val_stats.get("recon_pixel_avg", float("inf")))
        if stage == "reconstruction":
            print(f"Epoch {epoch}: val_recon_loss={recon_val:.6f}")
        else:
            print(f"Epoch {epoch}: val_loss={val_stats.get('loss', 0):.4f}, val_acc1={acc1:.1f}%")

        if output_dir:
            save_dict = {
                "cross_modality_encoder": model_without_ddp.cross_modality_encoder.state_dict(),
                "alignment_model":        model_without_ddp.cross_modality_encoder.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scaler":     loss_scaler.state_dict(),
                "epoch":      epoch,
                "full_config": OmegaConf.to_container(cfg, resolve=True),
            }
            for mod, dec in recon_decoders.items():
                save_dict[f"recon_decoder_{mod}"] = dec.state_dict()

            is_best = (recon_val < best_recon_loss) if stage == "reconstruction" else (acc1 > best_acc1)
            if is_best:
                best_recon_loss = min(best_recon_loss, recon_val)
                best_acc1 = max(best_acc1, acc1)
            if misc.is_main_process():
                if is_best:
                    torch.save(save_dict, os.path.join(output_dir, "checkpoint_best.pth"))
                torch.save(save_dict, os.path.join(output_dir, "checkpoint_latest.pth"))

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                     **{f"val_{k}": v for k, v in val_stats.items()},
                     "epoch": epoch}
        if output_dir and misc.is_main_process():
            if log_writer:
                log_writer.flush()
            with open(os.path.join(output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    print(f"Training time: {datetime.timedelta(seconds=int(time.time() - start_time))}")
    print(f"Best val acc@1: {best_acc1:.1f}%")


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    output_dir = os.path.join(cfg.output_dir, cfg.log_name) if cfg.log_name else cfg.output_dir
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if misc.is_main_process():
            OmegaConf.save(cfg, os.path.join(output_dir, "config_full.yaml"), resolve=True)

    if cfg.log_name and HAS_WANDB and misc.is_main_process():
        wandb.init(
            entity="projects-wpl", project="tvl-stage2",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.log_name, sync_tensorboard=True,
        )

    run(cfg)


if __name__ == "__main__":
    main()
