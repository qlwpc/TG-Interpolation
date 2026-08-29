#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:4
#SBATCH --requeue
set -euo pipefail

: "${PAUSE_CAMPAIGN_DIR:?PAUSE_CAMPAIGN_DIR is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
PAUSE_WORKSPACE=${PAUSE_WORKSPACE:-/home/wangpch/TG-Interpolation}
PAUSE_SCRIPT_ROOT=${PAUSE_SCRIPT_ROOT:-${PAUSE_WORKSPACE}/scripts}

# Do not override CUDA_VISIBLE_DEVICES here. Slurm assigns an independent GPU
# set to every array element and exports the correct visibility mask.
exec bash "${PAUSE_SCRIPT_ROOT}/pause_eval/run_finetune_eval.sh" \
  "${PAUSE_CAMPAIGN_DIR}" "${SLURM_ARRAY_TASK_ID}"
