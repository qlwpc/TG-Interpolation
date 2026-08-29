#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:8
#SBATCH --requeue
set -euo pipefail
: "${PAUSE_CAMPAIGN_DIR:?PAUSE_CAMPAIGN_DIR is required}"
: "${PAUSE_PACKED_TASK:?PAUSE_PACKED_TASK is required}"
PAUSE_WORKSPACE=${PAUSE_WORKSPACE:-/home/wangpch/TG-Interpolation}
PAUSE_SCRIPT_ROOT=${PAUSE_SCRIPT_ROOT:-${PAUSE_WORKSPACE}/scripts}
bash "${PAUSE_SCRIPT_ROOT}/pause_eval/run_eval_only_packed.sh" \
  "${PAUSE_CAMPAIGN_DIR}" "${PAUSE_PACKED_TASK}"
