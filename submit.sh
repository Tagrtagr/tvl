#!/usr/bin/env bash
#SBATCH --job-name=tvl_train
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_ydu_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


source ~/.bashrc
conda activate tvl

python experiments/stage2/claude_flextok/train.py --config-name config --resume \
  stage=alignment \
  stage1_checkpoint=ckpts/tvl_enc_vittiny.pth \
  datasets_dir=/n/holylabs/ydu_lab/Lab/pwu/Projects/tactile_encoder/tvl/dataset/tvl_dataset \
  datasets=[hct,ssvtp] \
  output_dir=./output/no_perservation_loss