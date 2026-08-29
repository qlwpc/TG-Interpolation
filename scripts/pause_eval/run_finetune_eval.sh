#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN=${1:?usage: run_finetune_eval.sh CAMPAIGN INDEX}
INDEX=${2:?usage: run_finetune_eval.sh CAMPAIGN INDEX}
WORKSPACE=${PAUSE_WORKSPACE:-/home/wangpch/TG-Interpolation}
SCRIPT_ROOT=${PAUSE_SCRIPT_ROOT:-${WORKSPACE}/scripts}
RUNTIME_BIN=${PAUSE_RUNTIME_BIN:-/home/wangpch/.conda/envs/LLM/bin}
CAMPAIGN_DRIVER=${PAUSE_CAMPAIGN_DRIVER:-${SCRIPT_ROOT}/pause_eval_campaign.py}
XSUM_PIPELINE_VERSION=2
export PATH=${RUNTIME_BIN}:${PATH}
ROW=$(sed -n "$((INDEX + 2))p" "${CAMPAIGN}/finetune_runs.tsv")
IFS=$'\t' read -r _ RUN_NAME TASK SEED CHECKPOINT CONFIG RUN_DIR SAVE EVAL_BATCH <<< "${ROW}"
EVAL_BATCH=${EVAL_BATCH%$'\r'}
EVAL_BATCH=${PAUSE_EVAL_BATCH_OVERRIDE:-${EVAL_BATCH}}
EVAL_NPROC=${PAUSE_EVAL_NPROC:-4}
EVAL_GLOBAL_TRAIN_BATCH=${PAUSE_EVAL_GLOBAL_TRAIN_BATCH:-$((EVAL_NPROC * 10))}
if ! [[ "${EVAL_BATCH}" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "${EVAL_NPROC}" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "${EVAL_GLOBAL_TRAIN_BATCH}" =~ ^[1-9][0-9]*$ ]] || \
   [ $((EVAL_GLOBAL_TRAIN_BATCH % EVAL_NPROC)) -ne 0 ]; then
  exit 22
fi
test -f "${RUN_DIR}/TRAIN_DONE"
FINAL=$(cat "${RUN_DIR}/final_checkpoint.txt")
test -f "${FINAL}/model.pt"
if [ "${TASK}" = xsum ]; then
  CONTRACT=${RUN_DIR}/training_contract.json
  CONTRACT_PATH=${CONTRACT} EXPECTED_VERSION=${XSUM_PIPELINE_VERSION} \
    "${RUNTIME_BIN}/python" -c \
    'import json, os; p=json.load(open(os.environ["CONTRACT_PATH"])); assert p["xsum_pipeline_version"] == int(os.environ["EXPECTED_VERSION"])'
fi
mkdir -p "${RUN_DIR}/eval_tmp" "${RUN_DIR}/mpl-eval"
exec > >(tee -a "${RUN_DIR}/eval.log") 2>&1
export PYTHONPATH=${WORKSPACE}${PYTHONPATH:+:${PYTHONPATH}}
export MPLCONFIGDIR=${RUN_DIR}/mpl-eval
export WANDB_MODE=disabled
cd "${WORKSPACE}"

if [ -f "${RUN_DIR}/EVAL_DONE" ]; then
  exit 0
fi
if [ "${PAUSE_SKIP_CAMPAIGN_VERIFY:-0}" = 1 ]; then
  echo "CAMPAIGN_VERIFY skipped_by_verified_migration"
elif [ -f "${CAMPAIGN}/smoke/SMOKE_DONE" ]; then
  echo "CAMPAIGN_VERIFY inherited_from_smoke"
else
  python "${CAMPAIGN_DRIVER}" verify --campaign-dir "${CAMPAIGN}"
fi
if [ "${TASK}" = xsum ]; then
  EVALUATORS="[{label: xsum, type: rouge, device_eval_batch_size: ${EVAL_BATCH}}]"
else
  EVALUATORS="[{label: boolq, type: downstream, device_eval_batch_size: ${EVAL_BATCH}}]"
fi
PORT=$((21000 + (${SLURM_ARRAY_JOB_ID:-1000} + ${SLURM_ARRAY_TASK_ID:-INDEX}) % 30000))
"${RUNTIME_BIN}/torchrun" --master_port="${PORT}" --nproc-per-node="${EVAL_NPROC}" \
  "${SCRIPT_ROOT}/train.py" "${CONFIG}" \
  --run_name="${RUN_NAME}_eval" --workspace="${WORKSPACE}" --save_folder="${RUN_DIR}/eval_tmp" \
  --save_overwrite=true --load_path="${FINAL}" --distributed_strategy=ddp \
  --activation_checkpointing=null --evaluators="${EVALUATORS}" --eval_on_load=true \
  --eval_return=true --eval_no_save=true --max_duration=0 --stop_at=0 \
  --global_train_batch_size="${EVAL_GLOBAL_TRAIN_BATCH}" \
  --device_eval_batch_size="${EVAL_BATCH}" --eval_subset_num_batches=-1 \
  --reset_optimizer_state=true --reset_trainer_state=true
if [ "${TASK}" = xsum ]; then
  grep -q '^    R-AVG=' "${RUN_DIR}/eval.log"
else
  grep -q 'eval/downstream/boolq_acc__' "${RUN_DIR}/eval.log"
fi
touch "${RUN_DIR}/EVAL_DONE"
