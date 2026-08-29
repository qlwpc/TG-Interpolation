#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN=${1:?usage: submit_base_jobs.sh CAMPAIGN}
WORKSPACE=${PAUSE_WORKSPACE:-/home/wangpch/TG-Interpolation}
SCRIPT_ROOT=${PAUSE_SCRIPT_ROOT:-${WORKSPACE}/scripts}
RUNTIME_BIN=${PAUSE_RUNTIME_BIN:-/home/wangpch/.conda/envs/LLM/bin}
SBATCH_BIN=${PAUSE_SBATCH_BIN:-sbatch}
PARTITION=${PAUSE_SLURM_PARTITION:-ShangHAI}
ACCOUNT=${PAUSE_SLURM_ACCOUNT:-tukw-ShangHAI}
GPU_TYPE=${PAUSE_SLURM_GPU_TYPE:-NVIDIARTX6000D}
TOTAL_GPU_BUDGET=${PAUSE_TOTAL_GPU_BUDGET:-8}

if ! [[ "${TOTAL_GPU_BUDGET}" =~ ^[1-9][0-9]*$ ]]; then
  exit 22
fi
mapfile -t ROWS < <(tail -n +2 "${CAMPAIGN}/base_runs.tsv")
if [ "${#ROWS[@]}" -ne 3 ]; then
  echo "expected three base evaluation rows" >&2
  exit 22
fi
requested=0
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r _ _ _ _ _ _ gpus <<< "${row}"
  gpus=${gpus%$'\r'}
  requested=$((requested + gpus))
done
if [ "${requested}" -gt "${TOTAL_GPU_BUDGET}" ]; then
  echo "base jobs request ${requested} GPUs, exceeding budget ${TOTAL_GPU_BUDGET}" >&2
  exit 22
fi
mkdir -p "${CAMPAIGN}/slurm"

for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r index run_name task _ _ run_dir gpus <<< "${row}"
  gpus=${gpus%$'\r'}
  if [ -f "${run_dir}/EVAL_DONE" ]; then
    echo "SKIP task=${task} reason=EVAL_DONE"
    continue
  fi
  job_id=$("${SBATCH_BIN}" --parsable \
    --partition="${PARTITION}" --account="${ACCOUNT}" \
    --gres="gpu:${GPU_TYPE}:${gpus}" --cpus-per-task=$((gpus * 2)) \
    --mem=80G --time=12:00:00 --job-name="${run_name}" \
    --output="${CAMPAIGN}/slurm/base-${task}-%j.out" \
    --export=ALL,PAUSE_CAMPAIGN_DIR="${CAMPAIGN}",PAUSE_WORKSPACE="${WORKSPACE}",PAUSE_SCRIPT_ROOT="${SCRIPT_ROOT}",PAUSE_RUNTIME_BIN="${RUNTIME_BIN}",BASE_INDEX="${index}" \
    "${SCRIPT_ROOT}/slurm/pause_base_eval.sbatch")
  printf "SUBMITTED task=%s job_id=%s gpus=%s\n" "${task}" "${job_id}" "${gpus}"
done
