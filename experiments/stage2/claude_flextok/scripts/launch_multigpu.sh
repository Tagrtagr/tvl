#!/bin/bash
#SBATCH --job-name=stage2a-4gpu
#SBATCH --partition=viscam
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stage2a_4gpu_%j.out
#SBATCH --error=logs/stage2a_4gpu_%j.err

# =============================================================================
# Stage 2a: Multi-GPU Alignment Training (4x GPU with DDP)
# =============================================================================

set -e

STAGE1_CKPT="${STAGE1_CKPT:-/viscam/u/taarush/tvl_enc_vittiny.pth}"
DATASETS_DIR="${DATASETS_DIR:-/viscam/u/taarush}"
OUTPUT_DIR="experiments/stage2/runs/stage2a_4gpu"
LOG_NAME="stage2a_4gpu"
NUM_GPUS=4

cd "$(dirname "$0")/../../../.."
mkdir -p "$OUTPUT_DIR" logs

echo "Multi-GPU training with $NUM_GPUS GPUs"

torchrun --nproc_per_node=$NUM_GPUS \
    experiments/stage2/claude_flextok/train.py \
    --stage alignment \
    --stage1_checkpoint "$STAGE1_CKPT" \
    --tactile_model vit_tiny_patch16_224 \
    --datasets_dir "$DATASETS_DIR" \
    --datasets ssvtp hct \
    --output_dir "$OUTPUT_DIR" \
    --log_name "$LOG_NAME" \
    --hidden_dim 512 \
    --n_registers 32 \
    --n_shared 8 \
    --n_layers 4 \
    --n_heads 8 \
    --loss_type contrastive \
    --contrastive_weight 1.0 \
    --preservation_weight 0.5 \
    --epochs 100 \
    --batch_size 64 \
    --blr 3e-4 \
    --warmup_epochs 10 \
    --weight_decay 0.05 \
    --num_workers 10 \
    --seed 42
