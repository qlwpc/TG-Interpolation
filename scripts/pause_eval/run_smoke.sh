#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN=${1:?usage: run_smoke.sh CAMPAIGN}
WORKSPACE=${PAUSE_WORKSPACE:-/home/wangpch/TG-Interpolation}
SCRIPT_ROOT=${PAUSE_SCRIPT_ROOT:-${WORKSPACE}/scripts}
RUNTIME_BIN=${PAUSE_RUNTIME_BIN:-/home/wangpch/.conda/envs/LLM/bin}
CAMPAIGN_DRIVER=${PAUSE_CAMPAIGN_DRIVER:-${SCRIPT_ROOT}/pause_eval_campaign.py}
SMOKE_TRAIN_STEPS=${PAUSE_SMOKE_TRAIN_STEPS:-1}
export PATH=${RUNTIME_BIN}:${PATH}
SMOKE_ROOT=${CAMPAIGN}/smoke
mkdir -p "${SMOKE_ROOT}" "${SMOKE_ROOT}/mpl"
exec > >(tee -a "${SMOKE_ROOT}/smoke.log") 2>&1
export PYTHONPATH=${WORKSPACE}${PYTHONPATH:+:${PYTHONPATH}}
export MPLCONFIGDIR=${SMOKE_ROOT}/mpl
export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${WORKSPACE}"

if [ -f "${SMOKE_ROOT}/SMOKE_DONE" ]; then
  exit 0
fi
python "${CAMPAIGN_DRIVER}" verify --campaign-dir "${CAMPAIGN}"
nvidia-smi

# One real optimizer step for each finetuning data path. These runs prove that
# SEP-expanded XSum and BoolQ batches reach backward/update before the full array
# is released by the afterok dependency.
for TASK_TO_SMOKE in xsum boolq; do
  INDEX=$(awk -F $'\t' -v task="${TASK_TO_SMOKE}" \
    'NR > 1 && $3 == task {print $1; exit}' "${CAMPAIGN}/finetune_runs.tsv")
  test -n "${INDEX}"
  ROW=$(sed -n "$((INDEX + 2))p" "${CAMPAIGN}/finetune_runs.tsv")
  IFS=$'\t' read -r _ RUN_NAME TASK SEED CHECKPOINT CONFIG RUN_DIR SAVE EVAL_BATCH <<< "${ROW}"
  TASK_ROOT=${SMOKE_ROOT}/${TASK}
  mkdir -p "${TASK_ROOT}/train" "${TASK_ROOT}/eval"
  PORT=$((31000 + INDEX + ${SLURM_JOB_ID:-1000} % 10000))
  "${RUNTIME_BIN}/torchrun" --master_port="${PORT}" --nproc-per-node=4 \
    "${SCRIPT_ROOT}/train.py" "${CONFIG}" \
    --run_name="${RUN_NAME}_smoke_train" --workspace="${WORKSPACE}" \
    --save_folder="${TASK_ROOT}/train" --save_overwrite=true --try_load_latest_save=false \
    --load_path="${CHECKPOINT}" --stop_after="${SMOKE_TRAIN_STEPS}" --evaluators='[]'
  test -e "${TASK_ROOT}/train/latest-unsharded"

  if [ "${TASK}" = xsum ]; then
    EVALUATORS="[{label: xsum, type: rouge, device_eval_batch_size: ${EVAL_BATCH}, subset_num_batches: 2}]"
  else
    EVALUATORS="[{label: boolq, type: downstream, device_eval_batch_size: ${EVAL_BATCH}, subset_num_batches: 2}]"
  fi
  PORT=$((32000 + INDEX + ${SLURM_JOB_ID:-1000} % 10000))
  "${RUNTIME_BIN}/torchrun" --master_port="${PORT}" --nproc-per-node=4 \
    "${SCRIPT_ROOT}/train.py" "${CONFIG}" \
    --run_name="${RUN_NAME}_smoke_eval" --workspace="${WORKSPACE}" \
    --save_folder="${TASK_ROOT}/eval" --save_overwrite=true --load_path="${CHECKPOINT}" \
    --evaluators="${EVALUATORS}" --eval_on_load=true --eval_return=true --eval_no_save=true \
    --max_duration=0 --stop_at=0 --reset_optimizer_state=true --reset_trainer_state=true
done

grep -q '^    R-AVG=' "${SMOKE_ROOT}/smoke.log"
grep -q 'eval/downstream/boolq_acc__' "${SMOKE_ROOT}/smoke.log"
touch "${SMOKE_ROOT}/SMOKE_DONE"
