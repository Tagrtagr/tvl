#!/bin/bash
#SBATCH --job-name=stage2b-recon
#SBATCH --partition=viscam
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage2b_%j.out
#SBATCH --error=logs/stage2b_%j.err

# =============================================================================
# Stage 2b: Reconstruction Decoder Training
#
# Freeze alignment model from Stage 2a, train autoregressive decoder only.
#
# Usage:
#   sbatch experiments/stage2/claude_flextok/scripts/launch_stage2b.sh
# =============================================================================

set -e

# ---- Paths (edit these for your setup) ----
STAGE1_CKPT="${STAGE1_CKPT:-/viscam/u/taarush/tvl_enc_vittiny.pth}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-experiments/stage2/runs/stage2a_alignment/checkpoint_best.pth}"
DATASETS_DIR="${DATASETS_DIR:-/viscam/u/taarush}"
OUTPUT_DIR="experiments/stage2/runs/stage2b_recon"
LOG_NAME="stage2b_recon"

# ---- Setup ----
cd "$(dirname "$0")/../../../.."
mkdir -p "$OUTPUT_DIR" logs

# ---- Verify paths (after cd so relative paths resolve) ----
if [ ! -f "$ALIGNMENT_CKPT" ]; then
    echo "ERROR: Alignment checkpoint not found at $ALIGNMENT_CKPT"
    echo "Run Stage 2a first, then update ALIGNMENT_CKPT."
    exit 1
fi

echo "========================================="
echo "Stage 2b: Reconstruction Decoder Training"
echo "Alignment checkpoint: $ALIGNMENT_CKPT"
echo "========================================="

# ---- Launch ----
python experiments/stage2/claude_flextok/train.py \
    --stage reconstruction \
    --alignment_checkpoint "$ALIGNMENT_CKPT" \
    --stage1_checkpoint "$STAGE1_CKPT" \
    --tactile_model vit_tiny_patch16_224 \
    --datasets_dir "$DATASETS_DIR" \
    --datasets ssvtp \
    --output_dir "$OUTPUT_DIR" \
    --log_name "$LOG_NAME" \
    --decoder_type autoregressive \
    --hidden_dim 512 \
    --n_registers 32 \
    --n_shared 8 \
    --n_layers 4 \
    --n_heads 8 \
    --recon_base_channels 64 \
    --recon_decoder_layers 2 \
    --recon_loss_type mse \
    --reconstruction_weight 1.0 \
    --use_prefix_recon \
    --prefix_recon_weight 0.5 \
    --epochs 100 \
    --batch_size 256 \
    --blr 3e-4 \
    --warmup_epochs 10 \
    --num_workers 10 \
    --seed 42

echo "Stage 2b training complete. Best checkpoint: $OUTPUT_DIR/$LOG_NAME/checkpoint_best.pth"
