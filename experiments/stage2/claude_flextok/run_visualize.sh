#!/bin/bash
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=prefix_recon_vis
#SBATCH --output=visualize_prefix_recon_%j.log

cd /viscam/u/taarush/tvl

python experiments/stage2/claude_flextok/visualize_prefix_recon.py \
    --checkpoint experiments/stage2/runs/stage2b_recon/stage2b_recon/checkpoint_best.pth \
    --stage1_checkpoint /viscam/u/taarush/tvl_enc_vittiny.pth \
    --datasets_dir /viscam/u/taarush \
    --n_samples 4 \
    --output_dir visualizations/stage2b_prefix_with_s1
