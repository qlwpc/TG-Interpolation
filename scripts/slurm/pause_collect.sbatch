#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:30:00
set -euo pipefail
: "${PAUSE_CAMPAIGN_DIR:?PAUSE_CAMPAIGN_DIR is required}"
PAUSE_WORKSPACE=${PAUSE_WORKSPACE:-/home/wangpch/TG-Interpolation}
PAUSE_SCRIPT_ROOT=${PAUSE_SCRIPT_ROOT:-${PAUSE_WORKSPACE}/scripts}
PAUSE_RUNTIME_BIN=${PAUSE_RUNTIME_BIN:-/home/wangpch/.conda/envs/LLM/bin}
export PATH=${PAUSE_RUNTIME_BIN}:${PATH}
export PYTHONPATH=${PAUSE_WORKSPACE}${PYTHONPATH:+:${PYTHONPATH}}
cd "${PAUSE_WORKSPACE}"
python "${PAUSE_SCRIPT_ROOT}/pause_eval_campaign.py" verify --campaign-dir "${PAUSE_CAMPAIGN_DIR}"
python "${PAUSE_SCRIPT_ROOT}/pause_eval_campaign.py" collect --campaign-dir "${PAUSE_CAMPAIGN_DIR}"
grep -q 'Canonical metric records: 13/13' "${PAUSE_CAMPAIGN_DIR}/REPORT.md"
grep -q 'All requested canonical metrics are present and finite.' \
  "${PAUSE_CAMPAIGN_DIR}/REPORT.md"
touch "${PAUSE_CAMPAIGN_DIR}/CAMPAIGN_DONE"
