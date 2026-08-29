#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN=${1:?usage: submit_finetune_eval_array.sh CAMPAIGN TASK}
TASK=${2:?usage: submit_finetune_eval_array.sh CAMPAIGN TASK}
WORKSPACE=${PAUSE_WORKSPACE:-/home/wangpch/TG-Interpolation}
SCRIPT_ROOT=${PAUSE_SCRIPT_ROOT:-${WORKSPACE}/scripts}
RUNTIME_BIN=${PAUSE_RUNTIME_BIN:-/home/wangpch/.conda/envs/LLM/bin}
SBATCH_BIN=${PAUSE_SBATCH_BIN:-sbatch}
PARTITION=${PAUSE_SLURM_PARTITION:-ShangHAI}
ACCOUNT=${PAUSE_SLURM_ACCOUNT:-tukw-ShangHAI}
GPU_TYPE=${PAUSE_SLURM_GPU_TYPE:-NVIDIARTX6000D}
MAX_PARALLEL=${PAUSE_ARRAY_MAX_PARALLEL:-2}
GPUS_PER_RUN=${PAUSE_GPUS_PER_RUN:-4}
TOTAL_GPU_BUDGET=${PAUSE_TOTAL_GPU_BUDGET:-8}

if [ "${TASK}" != xsum ] && [ "${TASK}" != boolq ]; then
  exit 22
fi
for value in "${MAX_PARALLEL}" "${GPUS_PER_RUN}" "${TOTAL_GPU_BUDGET}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    exit 22
  fi
done
if [ $((MAX_PARALLEL * GPUS_PER_RUN)) -gt "${TOTAL_GPU_BUDGET}" ]; then
  echo "array concurrency exceeds total GPU budget" >&2
  exit 22
fi
if [ "${GPUS_PER_RUN}" -ne 4 ]; then
  echo "finetune configs currently require four GPUs per seed" >&2
  exit 22
fi

mapfile -t INDICES < <(
  awk -F $'\t' -v task="${TASK}" 'NR > 1 && $3 == task {print $1}' \
    "${CAMPAIGN}/finetune_runs.tsv"
)
if [ "${#INDICES[@]}" -ne 5 ] || [ $((INDICES[4] - INDICES[0])) -ne 4 ]; then
  echo "expected five contiguous ${TASK} indices" >&2
  exit 22
fi

if [ "${TASK}" = xsum ]; then
  EVAL_BATCH=1
  LOG_PREDICTIONS=0
else
  EVAL_BATCH=${PAUSE_BOOLQ_EVAL_BATCH:-32}
  LOG_PREDICTIONS=1
fi
mkdir -p "${CAMPAIGN}/slurm"

"${SBATCH_BIN}" --parsable \
  --partition="${PARTITION}" --account="${ACCOUNT}" \
  --array="${INDICES[0]}-${INDICES[4]}%${MAX_PARALLEL}" \
  --gres="gpu:${GPU_TYPE}:${GPUS_PER_RUN}" \
  --cpus-per-task=$((GPUS_PER_RUN * 2)) --mem=80G --time=48:00:00 \
  --job-name="$(basename "${CAMPAIGN}")-${TASK}-ft-eval" \
  --output="${CAMPAIGN}/slurm/${TASK}-ft-eval-%A_%a.out" \
  --export=ALL,OLMO_LOG_XSUM_PREDICTIONS="${LOG_PREDICTIONS}",PAUSE_CAMPAIGN_DIR="${CAMPAIGN}",PAUSE_WORKSPACE="${WORKSPACE}",PAUSE_SCRIPT_ROOT="${SCRIPT_ROOT}",PAUSE_RUNTIME_BIN="${RUNTIME_BIN}",PAUSE_EVAL_NPROC=4,PAUSE_EVAL_GLOBAL_TRAIN_BATCH=40,PAUSE_EVAL_BATCH_OVERRIDE="${EVAL_BATCH}" \
  "${SCRIPT_ROOT}/slurm/pause_finetune_eval_array.sbatch"
