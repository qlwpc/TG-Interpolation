#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN=${1:?usage: run_eval_only_packed.sh CAMPAIGN TASK}
TASK=${2:?usage: run_eval_only_packed.sh CAMPAIGN TASK}
WORKSPACE=${PAUSE_WORKSPACE:-/home/wangpch/TG-Interpolation}
SCRIPT_ROOT=${PAUSE_SCRIPT_ROOT:-${WORKSPACE}/scripts}
PARALLEL=${PAUSE_PACKED_PARALLEL:-2}
GPUS_PER_RUN=${PAUSE_GPUS_PER_RUN:-4}
SELECTED=${PAUSE_SELECTED_INDICES:-}

if [ "${TASK}" != boolq ] && [ "${TASK}" != xsum ]; then
  exit 22
fi
if ! [[ "${PARALLEL}" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "${GPUS_PER_RUN}" =~ ^[1-9][0-9]*$ ]]; then
  exit 22
fi

mapfile -t TASK_INDICES < <(
  awk -F $'\t' -v task="${TASK}" 'NR > 1 && $3 == task {print $1}' \
    "${CAMPAIGN}/finetune_runs.tsv"
)
if [ "${#TASK_INDICES[@]}" -ne 5 ]; then
  exit 22
fi

INDICES=()
if [ -n "${SELECTED}" ]; then
  NORMALIZED_SELECTED=${SELECTED//:/,}
  IFS=',' read -r -a REQUESTED <<< "${NORMALIZED_SELECTED}"
  for index in "${REQUESTED[@]}"; do
    if ! [[ "${index}" =~ ^[0-9]+$ ]]; then
      exit 22
    fi
    found=0
    for candidate in "${TASK_INDICES[@]}"; do
      if [ "${index}" = "${candidate}" ]; then
        found=1
        break
      fi
    done
    if [ "${found}" -ne 1 ]; then
      exit 22
    fi
    INDICES+=("${index}")
  done
else
  INDICES=("${TASK_INDICES[@]}")
fi

run_one() {
  local index=$1
  local slot=$2
  local first_gpu=$((slot * GPUS_PER_RUN))
  local last_gpu=$((first_gpu + GPUS_PER_RUN - 1))
  local gpu_list
  gpu_list=$(seq -s, "${first_gpu}" "${last_gpu}")
  export CUDA_VISIBLE_DEVICES=${gpu_list}
  export PAUSE_SKIP_CAMPAIGN_VERIFY=1
  echo "PACKED_EVAL_START task=${TASK} index=${index} slot=${slot} gpus=${gpu_list}"
  bash "${SCRIPT_ROOT}/pause_eval/run_finetune_eval.sh" "${CAMPAIGN}" "${index}"
  echo "PACKED_EVAL_DONE task=${TASK} index=${index} slot=${slot} gpus=${gpu_list}"
}

for ((offset = 0; offset < ${#INDICES[@]}; offset += PARALLEL)); do
  pids=()
  for ((slot = 0; slot < PARALLEL; slot += 1)); do
    position=$((offset + slot))
    if [ "${position}" -ge "${#INDICES[@]}" ]; then
      break
    fi
    run_one "${INDICES[${position}]}" "${slot}" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [ "${failed}" -ne 0 ]; then
    exit 1
  fi
done

all_done=1
for index in "${TASK_INDICES[@]}"; do
  row=$(sed -n "$((index + 2))p" "${CAMPAIGN}/finetune_runs.tsv")
  IFS=$'\t' read -r _ _ _ _ _ _ run_dir _ _ <<< "${row}"
  if [ ! -f "${run_dir}/EVAL_DONE" ]; then
    all_done=0
    break
  fi
done
if [ "${all_done}" -eq 1 ]; then
  touch "${CAMPAIGN}/PACKED_EVAL_${TASK^^}_DONE"
else
  echo "PACKED_EVAL_SUBSET_DONE task=${TASK} selected=${SELECTED:-all}"
fi
