#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:4
#SBATCH --requeue
set -euo pipefail

: "${PAUSE1_CAMPAIGN_DIR:?PAUSE1_CAMPAIGN_DIR is required}"
: "${PAUSE2_CAMPAIGN_DIR:?PAUSE2_CAMPAIGN_DIR is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
PAUSE_WORKSPACE=${PAUSE_WORKSPACE:-/home/wangpch/TG-Interpolation}
PAUSE_SCRIPT_ROOT=${PAUSE_SCRIPT_ROOT:-${PAUSE_WORKSPACE}/scripts}
PAIR_MODE=${PAUSE_DUAL_PAIR_MODE:-0}

if ! [[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]] || \
   ! [[ "${PAIR_MODE}" =~ ^[01]$ ]]; then
  exit 22
fi

run_xsum() {
  local model_label=$1
  local campaign=$2
  local run_index=$3
  echo "DUAL_XSUM_START model=${model_label} run_index=${run_index} campaign=${campaign}"
  bash "${PAUSE_SCRIPT_ROOT}/pause_eval/run_finetune.sh" \
    "${campaign}" "${run_index}"
  bash "${PAUSE_SCRIPT_ROOT}/pause_eval/run_finetune_eval.sh" \
    "${campaign}" "${run_index}"
  echo "DUAL_XSUM_DONE model=${model_label} run_index=${run_index} campaign=${campaign}"
}

if [ "${PAIR_MODE}" -eq 1 ]; then
  # Shanghai's QOS counts every array element and permits only five submitted
  # jobs per user. One element therefore owns one seed and runs both models
  # serially on the same four GPUs. Alternate order avoids a model-order bias.
  if [ "${SLURM_ARRAY_TASK_ID}" -gt 4 ]; then
    exit 22
  fi
  RUN_INDEX=${SLURM_ARRAY_TASK_ID}
  if [ $((RUN_INDEX % 2)) -eq 0 ]; then
    run_xsum pause1 "${PAUSE1_CAMPAIGN_DIR}" "${RUN_INDEX}"
    run_xsum pause2 "${PAUSE2_CAMPAIGN_DIR}" "${RUN_INDEX}"
  else
    run_xsum pause2 "${PAUSE2_CAMPAIGN_DIR}" "${RUN_INDEX}"
    run_xsum pause1 "${PAUSE1_CAMPAIGN_DIR}" "${RUN_INDEX}"
  fi
  exit 0
fi

# Generic mode interleaves models: 0/1 are Pause-1/Pause-2 seed 0, 2/3
# are seed 1, and so on. Use this 10-element form only where QOS permits it.
if [ "${SLURM_ARRAY_TASK_ID}" -gt 9 ]; then
  exit 22
fi
RUN_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
if [ $((SLURM_ARRAY_TASK_ID % 2)) -eq 0 ]; then
  run_xsum pause1 "${PAUSE1_CAMPAIGN_DIR}" "${RUN_INDEX}"
else
  run_xsum pause2 "${PAUSE2_CAMPAIGN_DIR}" "${RUN_INDEX}"
fi
