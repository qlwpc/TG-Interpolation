#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN=${1:?usage: run_downstream_packed.sh CAMPAIGN TASK}
TASK=${2:?usage: run_downstream_packed.sh CAMPAIGN TASK}
WORKSPACE=${PAUSE_WORKSPACE:-/home/wangpch/TG-Interpolation}
SCRIPT_ROOT=${PAUSE_SCRIPT_ROOT:-${WORKSPACE}/scripts}
PARALLEL=${PAUSE_PACKED_PARALLEL:-2}
GPUS_PER_RUN=${PAUSE_GPUS_PER_RUN:-4}

if [ "${TASK}" != boolq ] && [ "${TASK}" != xsum ]; then
  exit 22
fi
if ! [[ "${PARALLEL}" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "${GPUS_PER_RUN}" =~ ^[1-9][0-9]*$ ]]; then
  exit 22
fi

mapfile -t INDICES < <(
  awk -F $'\t' -v task="${TASK}" 'NR > 1 && $3 == task {print $1}' \
    "${CAMPAIGN}/finetune_runs.tsv"
)
if [ "${#INDICES[@]}" -ne 5 ]; then
  exit 22
fi

run_one() {
  local index=$1
  local slot=$2
  local first_gpu=$((slot * GPUS_PER_RUN))
  local last_gpu=$((first_gpu + GPUS_PER_RUN - 1))
  local gpu_list
  gpu_list=$(seq -s, "${first_gpu}" "${last_gpu}")
  export CUDA_VISIBLE_DEVICES=${gpu_list}
  echo "PACKED_START task=${TASK} index=${index} slot=${slot} gpus=${gpu_list}"
  bash "${SCRIPT_ROOT}/pause_eval/run_finetune.sh" "${CAMPAIGN}" "${index}"
  bash "${SCRIPT_ROOT}/pause_eval/run_finetune_eval.sh" "${CAMPAIGN}" "${index}"
  echo "PACKED_DONE task=${TASK} index=${index} slot=${slot} gpus=${gpu_list}"
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

touch "${CAMPAIGN}/PACKED_${TASK^^}_DONE"
