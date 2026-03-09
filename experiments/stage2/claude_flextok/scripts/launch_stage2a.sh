#!/bin/bash
#SBATCH --job-name=stage2a-alignment
#SBATCH --partition=viscam
#SBATCH --account=viscam
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage2a_%j.out
#SBATCH --error=logs/stage2a_%j.err
#SBATCH --chdir=/viscam/u/taarush/tvl

# =============================================================================
# Stage 2a: Alignment Training (Register Tokens)
#
# Train register tokens with contrastive + preservation loss.
# Frozen Stage 1 encoders; only alignment model parameters are learned.
#
# Usage:
#   sbatch experiments/stage2/claude_flextok/scripts/launch_stage2a.sh
#   # or interactively:
#   srun --gres=gpu:1 --mem=32G --time=12:00:00 --pty bash
#   bash experiments/stage2/claude_flextok/scripts/launch_stage2a.sh
# =============================================================================

set -e

# ---- Paths (edit these for your setup) ----
STAGE1_CKPT="/viscam/u/taarush/tvl_enc_vittiny.pth"
DATASETS_DIR="/viscam/u/taarush"
OUTPUT_DIR="experiments/stage2/runs/stage2a_alignment"
LOG_NAME="stage2a_alignment"

# ---- Verify paths ----
if [ ! -f "$STAGE1_CKPT" ]; then
    echo "ERROR: Stage 1 checkpoint not found at $STAGE1_CKPT"
    echo "Please update STAGE1_CKPT in this script."
    exit 1
fi

if [ ! -d "$DATASETS_DIR/ssvtp" ]; then
    echo "WARNING: ssvtp dataset not found at $DATASETS_DIR/ssvtp"
    echo "Please update DATASETS_DIR in this script."
fi

# ---- Setup ----
# SBATCH --chdir sets working directory to repo root
mkdir -p "$OUTPUT_DIR" logs

echo "========================================="
echo "Stage 2a: Alignment Training"
echo "Stage 1 checkpoint: $STAGE1_CKPT"
echo "Datasets dir: $DATASETS_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "========================================="

# ---- Launch ----
python experiments/stage2/claude_flextok/train.py \
    --stage alignment \
    --stage1_checkpoint "$STAGE1_CKPT" \
    --tactile_model vit_tiny_patch16_224 \
    --datasets_dir "$DATASETS_DIR" \
    --datasets ssvtp \
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
    --batch_size 256 \
    --blr 3e-4 \
    --warmup_epochs 10 \
    --weight_decay 0.05 \
    --num_workers 10 \
    --seed 42

echo "Stage 2a training complete. Best checkpoint: $OUTPUT_DIR/$LOG_NAME/checkpoint_best.pth"
