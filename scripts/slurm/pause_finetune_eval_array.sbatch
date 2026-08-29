#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:4
#SBATCH --requeue
set -euo pipefail

: "${PAUSE_CAMPAIGN_DIR:?PAUSE_CAMPAIGN_DIR is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
PAUSE_WORKSPACE=${PAUSE_WORKSPACE:-/home/wangpch/TG-Interpolation}
PAUSE_SCRIPT_ROOT=${PAUSE_SCRIPT_ROOT:-${PAUSE_WORKSPACE}/scripts}

# Slurm owns CUDA_VISIBLE_DEVICES for this array element. Training and
# evaluation share the same allocation and checkpoint, so no job dependency is
# required and an element is independently retryable.
bash "${PAUSE_SCRIPT_ROOT}/pause_eval/run_finetune.sh" \
  "${PAUSE_CAMPAIGN_DIR}" "${SLURM_ARRAY_TASK_ID}"
exec bash "${PAUSE_SCRIPT_ROOT}/pause_eval/run_finetune_eval.sh" \
  "${PAUSE_CAMPAIGN_DIR}" "${SLURM_ARRAY_TASK_ID}"
