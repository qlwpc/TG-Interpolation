#!/usr/bin/env bash
set -euo pipefail

REMOTE_CAMPAIGN=${1:?usage: sync_results_from_remote.sh REMOTE_CAMPAIGN LOCAL_CAMPAIGN}
LOCAL_CAMPAIGN=${2:?usage: sync_results_from_remote.sh REMOTE_CAMPAIGN LOCAL_CAMPAIGN}
REMOTE_HOST=${PAUSE_REMOTE_HOST:-SIST}
WORKSPACE=${PAUSE_WORKSPACE:-/home/wangpch/TG-Interpolation}
COLLECTOR_PYTHON=${PAUSE_COLLECTOR_PYTHON:-/home/wangpch/.conda/envs/LLM/bin/python}

mkdir -p "${LOCAL_CAMPAIGN}/runs" "${LOCAL_CAMPAIGN}/sist_slurm"

# Results and validation metadata only. Never copy model/optimizer/trainer state
# into the local campaign and never overwrite local finetune checkpoints.
rsync -a --prune-empty-dirs \
  --include='*/' --include='eval.log' --include='EVAL_DONE' \
  --include='checkpoint_health.json' --include='final_model.sha256' \
  --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_CAMPAIGN}/runs/" "${LOCAL_CAMPAIGN}/runs/"
rsync -a --prune-empty-dirs --include='*.out' --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_CAMPAIGN}/slurm/" "${LOCAL_CAMPAIGN}/sist_slurm/"

"${COLLECTOR_PYTHON}" "${WORKSPACE}/scripts/pause_eval_campaign.py" collect \
  --campaign-dir "${LOCAL_CAMPAIGN}"
