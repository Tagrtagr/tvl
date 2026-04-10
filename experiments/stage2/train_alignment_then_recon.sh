#!/usr/bin/env bash
set -euo pipefail

# Two-stage Stage2 training:
#   1) alignment: train register/alignment model only
#   2) reconstruction: freeze alignment, train decoder only
#
# Example:
#   bash experiments/stage2/train_alignment_then_recon.sh \
#     --stage1_checkpoint ckpts/tvl_enc_vittiny.pth \
#     --datasets_dir /path/to/tvl_dataset \
#     --datasets ssvtp hct \
#     --output_root ./output/stage2_flextok

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_PY="experiments/stage2/claude_flextok/train.py"
ORIGINAL_ARGS=("$@")

# Conda environment for training (Slurm jobs must activate inside the batch step).
# Override or disable: TVL_CONDA_ENV=mycenv or TVL_CONDA_ENV="" to skip.
TVL_CONDA_ENV="${TVL_CONDA_ENV:-tvl}"
# Optional: path to conda.sh if `conda` is not on PATH in non-interactive shells (e.g. .../miniconda3/etc/profile.d/conda.sh)
CONDA_SH_PATH="${CONDA_SH_PATH:-}"

conda_activate_tvl() {
  local env_name="${1:-${TVL_CONDA_ENV}}"
  [[ -z "${env_name}" ]] && return 0
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${env_name}" ]]; then
    return 0
  fi
  if [[ -n "${CONDA_SH_PATH}" && -f "${CONDA_SH_PATH}" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_SH_PATH}"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "ERROR: Cannot activate conda env '${env_name}': conda not found. Install conda or set CONDA_SH_PATH." >&2
    return 1
  fi
  conda activate "${env_name}"
}

sbatch_wrap_cmd() {
  # Single bash command for sbatch --wrap=... (conda activate + script + safely quoted args).
  local script_abs="$1"
  shift
  local w="set -euo pipefail"
  if [[ -n "${TVL_CONDA_ENV}" ]]; then
    if [[ -n "${CONDA_SH_PATH}" && -f "${CONDA_SH_PATH}" ]]; then
      w+="; source $(printf '%q' "${CONDA_SH_PATH}")"
    else
      w+='; eval "$(conda shell.bash hook)"'
    fi
    w+="; conda activate $(printf '%q' "${TVL_CONDA_ENV}")"
  fi
  w+="; bash $(printf '%q' "${script_abs}")"
  local a
  for a in "$@"; do
    w+=" $(printf '%q' "${a}")"
  done
  printf '%s' "${w}"
}

STAGE1_CHECKPOINT=""
DATASETS_DIR=""
DATASETS=("ssvtp" "hct")
OUTPUT_ROOT="./output/stage2_flextok"

# Common training defaults (override via flags below if needed)
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS_ALIGN="${EPOCHS_ALIGN:-100}"
EPOCHS_RECON="${EPOCHS_RECON:-100}"
NUM_WORKERS="${NUM_WORKERS:-10}"
TAC_MODEL="${TAC_MODEL:-vit_tiny_patch16_224}"

# Reconstruction options
DECODER_TYPE="${DECODER_TYPE:-autoregressive}"   # autoregressive | conv
RECON_WEIGHT="${RECON_WEIGHT:-1.0}"
RECON_LOSS_TYPE="${RECON_LOSS_TYPE:-mse}"        # mse | l1 | smooth_l1
RECON_BASE_CHANNELS="${RECON_BASE_CHANNELS:-64}"
RECON_DECODER_LAYERS="${RECON_DECODER_LAYERS:-2}"

# Optional extras
SUBTRACT_BACKGROUND="${SUBTRACT_BACKGROUND:-}"   # "" | "background" | "mean" | "median" (train.py supports None/background)

# Slurm options (optional). If provided, the script will submit itself to Slurm.
SLURM_PARTITION="${SLURM_PARTITION:-}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_ACCOUNT_CHOICE="${SLURM_ACCOUNT_CHOICE:-}"
SLURM_ACCOUNT_CHOICES="${SLURM_ACCOUNT_CHOICES:-}"
SLURM_TIME="${SLURM_TIME:-}"
SLURM_GPUS="${SLURM_GPUS:-}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-}"
SLURM_MEM="${SLURM_MEM:-}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-}"
SLURM_EXCLUDE="${SLURM_EXCLUDE:-}"
SLURM_QOS="${SLURM_QOS:-}"
SLURM_OUTPUT="${SLURM_OUTPUT:-}"
SLURM_ERROR="${SLURM_ERROR:-}"
SLURM_ADDITIONAL_ARGS=( )

usage() {
  cat <<'EOF'
Usage:
  bash experiments/stage2/train_alignment_then_recon.sh \
    --stage1_checkpoint PATH \
    --datasets_dir PATH \
    [--datasets ssvtp hct] \
    [--output_root PATH]

Environment overrides (optional):
  TVL_CONDA_ENV=tvl              # conda env for training; set empty to skip activation
  CONDA_SH_PATH=/path/to/conda.sh  # if conda is not on PATH in batch/non-interactive bash
  PYTHON_BIN=python
  BATCH_SIZE=256
  EPOCHS_ALIGN=100
  EPOCHS_RECON=100
  NUM_WORKERS=10
  TAC_MODEL=vit_tiny_patch16_224
  DECODER_TYPE=autoregressive|conv
  RECON_WEIGHT=1.0
  RECON_LOSS_TYPE=mse|l1|smooth_l1
  RECON_BASE_CHANNELS=64
  RECON_DECODER_LAYERS=2
  SUBTRACT_BACKGROUND=background

Slurm submission (optional):
  --slurm                         Submit via sbatch using current flags
  --slurm_partition PARTITION
  --slurm_account ACCOUNT
  --slurm_account_choice NAME     pick from SLURM_ACCOUNT_CHOICES (env)
  --slurm_time TIME               e.g. 12:00:00
  --slurm_gpus N                  e.g. 1
  --slurm_cpus_per_task N         e.g. 16
  --slurm_mem MEM                 e.g. 64G
  --slurm_job_name NAME
  --slurm_exclude NODES
  --slurm_qos QOS
  --slurm_output PATH             e.g. slurm-%j.out (default if --slurm)
  --slurm_error PATH              e.g. slurm-%j.err (default if --slurm)
  --slurm_arg "--constraint=a100" (repeatable)
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

SUBMIT_SLURM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage1_checkpoint)
      STAGE1_CHECKPOINT="$2"; shift 2;;
    --datasets_dir)
      DATASETS_DIR="$2"; shift 2;;
    --datasets)
      shift
      DATASETS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        DATASETS+=("$1")
        shift
      done
      ;;
    --output_root|--output_dir|--output)
      OUTPUT_ROOT="$2"; shift 2;;

    --slurm)
      SUBMIT_SLURM=1; shift;;
    --slurm_partition)
      SLURM_PARTITION="$2"; shift 2;;
    --slurm_account)
      SLURM_ACCOUNT="$2"; shift 2;;
    --slurm_account_choice)
      SLURM_ACCOUNT_CHOICE="$2"; shift 2;;
    --slurm_time)
      SLURM_TIME="$2"; shift 2;;
    --slurm_gpus)
      SLURM_GPUS="$2"; shift 2;;
    --slurm_cpus_per_task)
      SLURM_CPUS_PER_TASK="$2"; shift 2;;
    --slurm_mem)
      SLURM_MEM="$2"; shift 2;;
    --slurm_job_name)
      SLURM_JOB_NAME="$2"; shift 2;;
    --slurm_exclude)
      SLURM_EXCLUDE="$2"; shift 2;;
    --slurm_qos)
      SLURM_QOS="$2"; shift 2;;
    --slurm_output)
      SLURM_OUTPUT="$2"; shift 2;;
    --slurm_error)
      SLURM_ERROR="$2"; shift 2;;
    --slurm_arg)
      SLURM_ADDITIONAL_ARGS+=("$2"); shift 2;;

    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${STAGE1_CHECKPOINT}" ]]; then
  echo "ERROR: --stage1_checkpoint is required" >&2
  exit 2
fi
if [[ -z "${DATASETS_DIR}" ]]; then
  echo "ERROR: --datasets_dir is required" >&2
  exit 2
fi

# If --slurm was requested, submit and exit before running locally.
if [[ "${SUBMIT_SLURM}" -eq 1 ]]; then
  SLURM_JOB_NAME="${SLURM_JOB_NAME:-tvl_stage2}"
  SLURM_OUTPUT="${SLURM_OUTPUT:-slurm-%j.out}"
  SLURM_ERROR="${SLURM_ERROR:-slurm-%j.err}"

  SBATCH_ARGS=("--job-name=${SLURM_JOB_NAME}" "--output=${SLURM_OUTPUT}" "--error=${SLURM_ERROR}")
  if [[ -n "${SLURM_PARTITION}" ]]; then SBATCH_ARGS+=("--partition=${SLURM_PARTITION}"); fi

  if [[ -n "${SLURM_ACCOUNT_CHOICE}" ]]; then
    IFS=',' read -r -a _SLURM_ACCOUNTS <<< "${SLURM_ACCOUNT_CHOICES}"
    ACCOUNT_MATCH=""
    for acct in "${_SLURM_ACCOUNTS[@]}"; do
      acct_trimmed="${acct## }"
      acct_trimmed="${acct_trimmed%% }"
      if [[ "${acct_trimmed}" == "${SLURM_ACCOUNT_CHOICE}" ]]; then
        ACCOUNT_MATCH="${acct_trimmed}"
        break
      fi
    done
    if [[ -z "${ACCOUNT_MATCH}" ]]; then
      echo "ERROR: --slurm_account_choice '${SLURM_ACCOUNT_CHOICE}' not in SLURM_ACCOUNT_CHOICES='${SLURM_ACCOUNT_CHOICES}'" >&2
      exit 2
    fi
    SLURM_ACCOUNT="${ACCOUNT_MATCH}"
  fi

  if [[ -n "${SLURM_ACCOUNT}" ]]; then SBATCH_ARGS+=("--account=${SLURM_ACCOUNT}"); fi
  if [[ -n "${SLURM_TIME}" ]]; then SBATCH_ARGS+=("--time=${SLURM_TIME}"); fi
  if [[ -n "${SLURM_GPUS}" ]]; then SBATCH_ARGS+=("--gres=gpu:${SLURM_GPUS}"); fi
  if [[ -n "${SLURM_CPUS_PER_TASK}" ]]; then SBATCH_ARGS+=("--cpus-per-task=${SLURM_CPUS_PER_TASK}"); fi
  if [[ -n "${SLURM_MEM}" ]]; then SBATCH_ARGS+=("--mem=${SLURM_MEM}"); fi
  if [[ -n "${SLURM_EXCLUDE}" ]]; then SBATCH_ARGS+=("--exclude=${SLURM_EXCLUDE}"); fi
  if [[ -n "${SLURM_QOS}" ]]; then SBATCH_ARGS+=("--qos=${SLURM_QOS}"); fi
  if [[ "${#SLURM_ADDITIONAL_ARGS[@]}" -gt 0 ]]; then SBATCH_ARGS+=("${SLURM_ADDITIONAL_ARGS[@]}"); fi

  # Strip --slurm* flags and their values when resubmitting.
  CLEAN_ARGS=()
  SKIP_NEXT=0
  for arg in "${ORIGINAL_ARGS[@]}"; do
    if [[ "${SKIP_NEXT}" -eq 1 ]]; then
      SKIP_NEXT=0
      continue
    fi
    case "$arg" in
      --slurm)
        ;;
      --slurm_partition|--slurm_account|--slurm_account_choice|--slurm_time|--slurm_gpus|--slurm_cpus_per_task|--slurm_mem|--slurm_job_name|--slurm_exclude|--slurm_qos|--slurm_output|--slurm_error|--slurm_arg)
        SKIP_NEXT=1
        ;;
      *)
        CLEAN_ARGS+=("$arg");;
    esac
  done

  _SCRIPT_FILE="${BASH_SOURCE[0]}"
  _SCRIPT_DIR="$(cd "$(dirname "${_SCRIPT_FILE}")" && pwd)"
  SCRIPT_ABS="${_SCRIPT_DIR}/$(basename "${_SCRIPT_FILE}")"

  if [[ -n "${TVL_CONDA_ENV}" ]]; then
    echo "Activating conda env '${TVL_CONDA_ENV}' before sbatch..."
    conda_activate_tvl
  fi

  WRAP_CMD="$(sbatch_wrap_cmd "${SCRIPT_ABS}" "${CLEAN_ARGS[@]}")"
  echo "Submitting to Slurm: ${SBATCH_ARGS[*]}"
  sbatch "${SBATCH_ARGS[@]}" --wrap="${WRAP_CMD}"
  exit 0
fi

conda_activate_tvl

ALIGN_OUT="${OUTPUT_ROOT}/stage2a_alignment"
RECON_OUT="${OUTPUT_ROOT}/stage2b_reconstruction"
ALIGN_CKPT="${ALIGN_OUT}/checkpoint_best.pth"

mkdir -p "${ALIGN_OUT}" "${RECON_OUT}"

DATASETS_YAML="[$(IFS=,; echo "${DATASETS[*]}")]"

ALIGN_ARGS=(
  stage=alignment
  stage1_checkpoint="${STAGE1_CHECKPOINT}"
  datasets_dir="${DATASETS_DIR}"
  datasets="${DATASETS_YAML}"
  tactile_model="${TAC_MODEL}"
  batch_size="${BATCH_SIZE}"
  epochs="${EPOCHS_ALIGN}"
  num_workers="${NUM_WORKERS}"
  output_dir="${ALIGN_OUT}"
  loss_type=contrastive
  save_recon_images=true
)

if [[ -n "${SUBTRACT_BACKGROUND}" ]]; then
  ALIGN_ARGS+=(subtract_background="${SUBTRACT_BACKGROUND}")
fi

echo "============================================================"
echo "Stage 2a: ALIGNMENT"
echo "Output: ${ALIGN_OUT}"
echo "============================================================"
"${PYTHON_BIN}" "${TRAIN_PY}" --config-name config "${ALIGN_ARGS[@]}"

if [[ ! -f "${ALIGN_CKPT}" ]]; then
  echo "ERROR: alignment checkpoint not found at: ${ALIGN_CKPT}" >&2
  echo "Check your run output_dir and whether checkpoint_best.pth was written." >&2
  exit 3
fi

RECON_ARGS=(
  stage=reconstruction
  alignment_checkpoint="${ALIGN_CKPT}"
  stage1_checkpoint="${STAGE1_CHECKPOINT}"
  datasets_dir="${DATASETS_DIR}"
  datasets="${DATASETS_YAML}"
  tactile_model="${TAC_MODEL}"
  decoder_type="${DECODER_TYPE}"
  use_prefix_recon=true
  prefix_recon_weight=0.5
  reconstruction_weight="${RECON_WEIGHT}"
  recon_loss_type="${RECON_LOSS_TYPE}"
  recon_base_channels="${RECON_BASE_CHANNELS}"
  recon_decoder_layers="${RECON_DECODER_LAYERS}"
  batch_size="${BATCH_SIZE}"
  epochs="${EPOCHS_RECON}"
  num_workers="${NUM_WORKERS}"
  output_dir="${RECON_OUT}"
  save_recon_images=true
)

if [[ -n "${SUBTRACT_BACKGROUND}" ]]; then
  RECON_ARGS+=(subtract_background="${SUBTRACT_BACKGROUND}")
fi

echo "============================================================"
echo "Stage 2b: RECONSTRUCTION"
echo "Alignment checkpoint: ${ALIGN_CKPT}"
echo "Output: ${RECON_OUT}"
echo "============================================================"
"${PYTHON_BIN}" "${TRAIN_PY}" --config-name config "${RECON_ARGS[@]}"

echo "Done."
