# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from timm.data.loader import MultiEpochsDataLoader
import timm.optim.optim_factory as optim_factory

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from tvl_enc.transformer_utils import handle_flash_attn
from tvl_enc import tvl 
from tvl_enc.tvl import ModalityType
from tvl_enc.tacvis import TacVisDataset, TacVisDatasetV2, RGB_AUGMENTS, TAC_AUGMENTS, TAC_AUGMENTS_BG, TAC_AUGMENTS_BG_CJ

from experiments.stage2.tvl_stage2 import TVLStage2
from experiments.stage2.loss_stage2 import Stage2Loss
from experiments.stage2.engine_stage2 import train_one_epoch, evaluate

import wandb


def get_args_parser():
    parser = argparse.ArgumentParser('Stage 2 Cross-Modality Alignment Training', add_help=False)
    
    # Stage 1 checkpoint
    parser.add_argument('--stage1_checkpoint', type=str, required=True,
                        help='Path to Stage 1 model checkpoint')
    
    # Model architecture
    parser.add_argument('--num_register_tokens', type=int, default=4,
                        help='Total number of register tokens (default: 4)')
    parser.add_argument('--num_overlapping_tokens', type=int, default=3,
                        help='Number of tokens for overlapping information (default: 3)')
    parser.add_argument('--register_token_dim', type=int, default=768,
                        help='Dimension of register tokens (default: 768)')
    parser.add_argument('--num_cross_attn_layers', type=int, default=2,
                        help='Number of cross-attention transformer layers (default: 2)')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--mlp_ratio', type=float, default=4.0,
                        help='MLP expansion ratio (default: 4.0)')
    parser.add_argument('--drop_rate', type=float, default=0.0,
                        help='Dropout rate (default: 0.0)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0,
                        help='Attention dropout rate (default: 0.0)')
    parser.add_argument('--drop_path_rate', type=float, default=0.0,
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--nested_dropout_rate', type=float, default=0.5,
                        help='Base dropout rate for nested dropout (default: 0.5)')
    
    # Loss function
    parser.add_argument('--complementary_reg_weight', type=float, default=0.01,
                        help='Weight for complementary token regularization (default: 0.01)')
    parser.add_argument('--disable_vision_text_loss', action='store_true', default=False,
                        help='Disable vision-text loss')
    parser.add_argument('--disable_tactile_text_loss', action='store_true', default=False,
                        help='Disable tactile-text loss')
    
    # Training parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    
    # Dataset parameters
    parser.add_argument("--datasets_dir", type=str, default="./.datasets",
                        help="Directory containing the datasets")
    parser.add_argument("--datasets", type=str, nargs="+", default=["ssvtp", "hct"], 
                        choices=["ssvtp", "hct"],
                        help="Datasets to use for training and validation")
    parser.add_argument("--importance_sampling", nargs="+", type=float, default=None, 
                        help="Percentage of data draw from each dataset")
    parser.add_argument("--use_not_contact", action="store_true", default=False, 
                        help="Use not contact data (from tacvis v2 dataset)")
    parser.add_argument("--percent_not_contact", type=float, default=0.1, 
                        help="Percentage of not contact data to use")
    parser.add_argument("--color_jitter", action="store_true", default=False, 
                        help="Apply color jitter to the image modality")
    parser.add_argument("--randomize_crop", action="store_true", default=False, 
                        help="Apply randomize_crop to the image modality")
    parser.add_argument("--subtract_background", type=str, default=None, 
                        choices=[None, "mean", "median", "background"],
                        help="Subtract tactile by background")
    parser.add_argument('--shuffle_text', action="store_true", 
                        help="Shuffle adjectives when feeding into the text modality")
    parser.add_argument('--no_text_prompt', action="store_true", 
                        help="Do not use text prompt when feeding into the text modality")
    parser.add_argument('--replace_synonyms', action="store_true", 
                        help="Replace synonyms when feeding into the text modality")
    parser.add_argument('--keep_k_synonyms', type=int, default=None, 
                        help="Use top k synonyms")
    
    # Stage 1 model parameters (needed to load checkpoint)
    parser.add_argument('--tactile_model', type=str, default='vit_tiny_patch16_224', 
                        choices=["vit_base_patch16_224", "vit_small_patch16_224", "vit_tiny_patch16_224", "resnet18"],
                        help="Tactile encoder model used in Stage 1")
    parser.add_argument('--common_latent_dim', type=int, default=None, 
                        help="Common latent dimension used in Stage 1")
    
    # Output parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--log_name', default=None, type=str)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--multi_epochs_dataloader', action='store_true', 
                        help='Use MultiEpochsDataLoader')
    parser.add_argument("--active_modality_names", nargs="+", type=str, 
                        default=["vision", "tactile", "text"], 
                        choices=["vision", "text", "audio", "thermal", "depth", "imu", "tactile"],
                        help="Modalities that are used for training")
    
    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--enable_flash_attention2', action='store_true', default=False, 
                        help="Use flash attention 2")
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument(
        '--find_unused_parameters', action='store_true',
        help="distributed ddp find unused parameters")
    parser.set_defaults(find_unused_parameters=False)
    
    return parser


def main(args):
    misc.init_distributed_mode(args)
    torch.backends.cudnn.determinstic = True

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    
    handle_flash_attn(args)

    # handle the active modalities
    modality_types = []
    modalities = ["vision", "text", "tactile"]
    for modality_name in args.active_modality_names:
        if modality_name in modalities:
            modality_type = getattr(ModalityType, modality_name.upper())
            modality_types.append(modality_type)
        else:
            raise ValueError(f"Unknown modality name: {modality_name}")
    
    # Load datasets (same as Stage 1)
    dataset_train = []
    dataset_val = []
    if args.importance_sampling:
        assert len(args.datasets) == len(args.importance_sampling), \
            "importance_sampling must have the same length as datasets"
        sampling_ratios = []
    
    print("datasets: ", args.datasets)
    if args.no_text_prompt:
        prompt = ""
    else:
        prompt = "This image gives tactile feelings of "
    
    if "ssvtp" in args.datasets:
        if args.subtract_background is None:
            tac_augments = TAC_AUGMENTS
        elif args.subtract_background == "background":
            if args.color_jitter:
                tac_augments = TAC_AUGMENTS_BG_CJ
            else:
                tac_augments = TAC_AUGMENTS_BG
        else:
            raise ValueError(f"Unknown subtract_background: {args.subtract_background}")
        
        root_dir = os.path.join(args.datasets_dir, "ssvtp")
        dataset_train.append(TacVisDataset(
                root_dir=root_dir, split="train",
                transform_rgb=RGB_AUGMENTS,
                transform_tac=tac_augments,
                modality_types=modality_types,
                shuffle_text=args.shuffle_text, 
                text_prompt=prompt,
                replace_synonyms=args.replace_synonyms,
                keep_k_synonyms=args.keep_k_synonyms,
            )
        )
        dataset_val.append(TacVisDataset(
                root_dir=root_dir, split="val",
                transform_rgb=RGB_AUGMENTS,
                transform_tac=tac_augments,
                modality_types=modality_types,
                shuffle_text=False, 
                text_prompt=prompt,
                replace_synonyms=False,
                keep_k_synonyms=args.keep_k_synonyms,
            )
        )

        if args.importance_sampling:
            sampling_ratios.append(args.importance_sampling[args.datasets.index("ssvtp")])

    if "hct" in args.datasets:
        dataset_dirs = []
        for i in os.listdir(os.path.join(args.datasets_dir, "hct")):
            sub_dir = os.path.join(args.datasets_dir, "hct", i)
            if os.path.isdir(sub_dir) and os.path.exists(os.path.join(sub_dir, "contact.json")):
                dataset_dirs.append(sub_dir)
        print("dataset_dirs: ", dataset_dirs)
        if args.subtract_background is None:
            tac_augments = TAC_AUGMENTS
        else:
            tac_augments = None
        dataset_train.append(TacVisDatasetV2(
                root_dir=dataset_dirs, split="train",
                transform_rgb=RGB_AUGMENTS,
                modality_types=modality_types,
                randomize_crop=args.randomize_crop, 
                transform_tac=tac_augments,
                use_not_contact=args.use_not_contact,
                replace_synonyms=args.replace_synonyms,
                shuffle_text=args.shuffle_text, 
                text_prompt=prompt,
                percent_not_contact=args.percent_not_contact,
            )
        )
        dataset_val.append(TacVisDatasetV2(
                root_dir=dataset_dirs, split="val",
                transform_rgb=RGB_AUGMENTS,
                modality_types=modality_types,
                transform_tac=tac_augments,
                use_not_contact=False,
                replace_synonyms=args.replace_synonyms,
                shuffle_text=False,
                text_prompt=prompt, 
            )
        )

        if args.importance_sampling:
            sampling_ratios.append(args.importance_sampling[args.datasets.index("hct")])
    
    if len(args.datasets) == 1:
        dataset_train = dataset_train[0]
        dataset_val = dataset_val[0]
    else:
        dataset_train = ConcatDataset(dataset_train)
        dataset_val = ConcatDataset(dataset_val)

    # Setup data loaders
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    if args.distributed:
        if args.importance_sampling:
            print("Using importance sampler with ratios: ", sampling_ratios)
            from util.datasets import DistributedImportanceSampler
            sampler_train = DistributedImportanceSampler(
                dataset_train, sampling_ratios=sampling_ratios, num_replicas=num_tasks, 
                rank=global_rank, shuffle=True, 
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    dataloader_cls = MultiEpochsDataLoader if args.multi_epochs_dataloader else torch.utils.data.DataLoader

    data_loader_train = dataloader_cls(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = dataloader_cls(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    # Load Stage 1 model
    print(f"Loading Stage 1 model from {args.stage1_checkpoint}")
    stage1_model = tvl.TVL(
        tactile_model=args.tactile_model, 
        active_modalities=modality_types, 
        common_latent_dim=args.common_latent_dim
    )
    
    # Load Stage 1 checkpoint
    checkpoint = torch.load(args.stage1_checkpoint, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load Stage 1 weights
    missing_keys, unexpected_keys = stage1_model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    
    # Ensure Stage 1 is frozen
    stage1_model.eval()
    for param in stage1_model.parameters():
        param.requires_grad = False
    
    # Create Stage 2 model
    model = TVLStage2(
        stage1_model=stage1_model,
        num_register_tokens=args.num_register_tokens,
        num_overlapping_tokens=args.num_overlapping_tokens,
        register_token_dim=args.register_token_dim,
        num_cross_attn_layers=args.num_cross_attn_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        nested_dropout_rate=args.nested_dropout_rate,
    )
    
    # Create loss function
    loss_fn = Stage2Loss(
        active_modalities=modality_types,
        num_overlapping_tokens=args.num_overlapping_tokens,
        complementary_reg_weight=args.complementary_reg_weight,
        disable_vision_text_loss=args.disable_vision_text_loss,
        disable_tactile_text_loss=args.disable_tactile_text_loss,
    )
    
    model.to(device)
    loss_fn.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], 
            find_unused_parameters=args.find_unused_parameters
        )
        model_without_ddp = model.module
    
    # Setup optimizer - only optimize Stage 2 parameters (register tokens and cross-attention layers)
    # Stage 1 parameters are frozen
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy_1 = 0.0
    max_accuracy_5 = 0.0
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, loss_fn, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        test_stats = evaluate(data_loader_val, loss_fn, model, device, epoch=epoch, log_writer=log_writer)

        print(f"Acc@1 the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        print(f"Acc@5 the network on the {len(dataset_val)} test images: {test_stats['acc5']:.1f}%")
        print(f"Validation Loss: {test_stats['loss']:.4f}")

        if args.output_dir:
            if test_stats["acc1"] >= max_accuracy_1:
                max_accuracy_1 = test_stats["acc1"]
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, metric="acc1")
            if test_stats["acc5"] >= max_accuracy_5:
                max_accuracy_5 = test_stats["acc5"]
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, metric="acc5")
            # save latest model
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, save_latest_model_only=True)
    
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.log_name is not None: 
        args.output_dir = os.path.join(args.output_dir, args.log_name)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
