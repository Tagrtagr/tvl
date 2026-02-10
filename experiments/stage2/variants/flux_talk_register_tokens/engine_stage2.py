<<<<<<< Current (Your changes)
=======
"""
Stage 2 Training Engine: FluxTalk Register Tokens Implementation

Created by: Cursor (Composer)
Model: Composer (Cursor's AI coding assistant)
Date: February 7, 2026
Experiment ID: flux_talk_register_tokens
Version: 1.0

Training and evaluation loops for Stage 2 cross-modality alignment.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
"""
import math
import sys
from typing import Iterable
import numpy as np

import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../'))
import util.misc as misc
import util.lr_sched as lr_sched

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: Iterable, 
    optimizer: torch.optim.Optimizer,
    device: torch.device, 
    epoch: int, 
    loss_scaler,
    log_writer=None,
    args=None
):
    """
    Training loop for Stage 2 model.
    
    Handles nested dropout and logs overlapping token alignment metrics.
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # samples is a dictionary 
        for k, v in samples.items():
            if isinstance(v, list):
                v = v[0]
            samples[k] = v.to(device, non_blocking=True).squeeze()

        # Forward pass through Stage 2 model
        with torch.cuda.amp.autocast():
            out_dict = model(samples, apply_nested_dropout=True)
            
            # Add logit scale from Stage 1 features if available
            if 'logit_scale' not in out_dict:
                if 'stage1_features' in out_dict and 'logit_scale' in out_dict['stage1_features']:
                    out_dict['logit_scale'] = out_dict['stage1_features']['logit_scale']
                else:
                    # Use default logit scale
                    out_dict['logit_scale'] = torch.tensor(1.0 / 0.07, device=device, dtype=torch.float32)
            
            loss_dict = loss_fn(out_dict, logit_scale=out_dict["logit_scale"], output_dict_format=True)

        # Extract total loss
        loss = loss_dict.pop("total_loss")
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        # log all keys in loss_dict
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                metric_logger.update(**{k: v.item()})
            else:
                metric_logger.update(**{k: v})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                loss_dict[k] = misc.all_reduce_mean(v.item())
            else:
                loss_dict[k] = v

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            # log all keys in loss_dict
            for k, v in loss_dict.items():
                log_writer.add_scalar(f"train_{k}", v, epoch_1000x)
            
            # Log nested dropout mask statistics if available
            if 'nested_dropout_mask' in out_dict and out_dict['nested_dropout_mask'] is not None:
                mask = out_dict['nested_dropout_mask']
                tokens_kept = mask.float().mean(dim=1).mean()  # Average tokens kept per sample
                log_writer.add_scalar('train_tokens_kept_ratio', tokens_kept.item(), epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    data_loader, 
    loss_fn, 
    model, 
    device, 
    epoch=None, 
    log_writer=None
):
    """
    Evaluation loop for Stage 2 model.
    
    Tracks complementary token statistics and overlapping token alignment metrics.
    """
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        # samples is a dictionary 
        for k, v in batch.items():
            if isinstance(v, list):
                v = v[0]
            batch[k] = v.to(device, non_blocking=True).squeeze()
            batch_size = v.shape[0]

        # compute output
        with torch.cuda.amp.autocast():
            output = model(batch, apply_nested_dropout=False)  # No nested dropout during eval
            
            # Add logit scale
            if 'logit_scale' not in output:
                if 'stage1_features' in output and 'logit_scale' in output['stage1_features']:
                    output['logit_scale'] = output['stage1_features']['logit_scale']
                else:
                    output['logit_scale'] = torch.tensor(1.0 / 0.07, device=device, dtype=torch.float32)
            
            loss_dict = loss_fn(output, logit_scale=output["logit_scale"], output_dict_format=True)

        loss = loss_dict.pop("total_loss")
        acc1 = loss_dict.pop("average_acc1", torch.tensor(0.0))
        acc5 = loss_dict.pop("average_acc5", torch.tensor(0.0))

        metric_logger.update(loss=loss.item())
        if isinstance(acc1, torch.Tensor):
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        else:
            metric_logger.meters['acc1'].update(acc1, n=batch_size)
        if isinstance(acc5, torch.Tensor):
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        else:
            metric_logger.meters['acc5'].update(acc5, n=batch_size)
        
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                metric_logger.update(**{k: v.item()})
            else:
                metric_logger.update(**{k: v})
        
        # Track complementary token statistics
        if 'complementary_tokens' in output:
            complementary_tokens = output['complementary_tokens']
            if isinstance(complementary_tokens, dict):
                for modality, tokens in complementary_tokens.items():
                    # tokens: (batch_size, num_complementary_tokens, token_dim)
                    token_norms = tokens.norm(dim=-1).mean().item()
                    metric_logger.update(**{f'complementary_norm_{modality}': token_norms})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if log_writer is not None and epoch is not None: 
        for k, meter in metric_logger.meters.items():
            log_writer.add_scalar(f"val_{k}", meter.global_avg, epoch)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
>>>>>>> Incoming (Background Agent changes)
