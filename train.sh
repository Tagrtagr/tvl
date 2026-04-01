python experiments/stage2/claude_flextok/train.py \
  --stage alignment \
  --stage1_checkpoint ckpts/tvl_enc_vittiny.pth \
  --datasets_dir /n/holylabs/ydu_lab/Lab/pwu/Projects/tactile_encoder/tvl/dataset/tvl_dataset \
  --datasets ssvtp hct \
  --output_dir ./output/stage2a

bash experiments/stage2/train_alignment_then_recon.sh \
  --stage1_checkpoint ckpts/tvl_enc_vittiny.pth \
  --datasets_dir dataset/tvl_dataset \
  --datasets ssvtp hct \
  --output_root output/stage2_flextok_slurm \
  --slurm \
  --slurm_partition kempner_h100 \
  --slurm_account kempner_ydu_lab \
  --slurm_time 24:00:00 \
  --slurm_gpus 1 \
  --slurm_cpus_per_task 16 \
  --slurm_mem 256G \
  --slurm_job_name tvl_stage2 \
  --slurm_output slurm-%j.out \
  --slurm_error slurm-%j.err