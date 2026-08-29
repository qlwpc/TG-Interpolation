#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH --requeue
set -euo pipefail
: "${PAUSE_CAMPAIGN_DIR:?PAUSE_CAMPAIGN_DIR is required}"
: "${BASE_INDEX:?BASE_INDEX is required}"
PAUSE_WORKSPACE=${PAUSE_WORKSPACE:-/home/wangpch/TG-Interpolation}
PAUSE_SCRIPT_ROOT=${PAUSE_SCRIPT_ROOT:-${PAUSE_WORKSPACE}/scripts}
bash "${PAUSE_SCRIPT_ROOT}/pause_eval/run_base_eval.sh" \
  "${PAUSE_CAMPAIGN_DIR}" "${BASE_INDEX}"
