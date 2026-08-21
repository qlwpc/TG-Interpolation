#!/bin/bash
set -euo pipefail

INDEX=${1:?usage: run_smoke.sh SMOKE_INDEX}
WORKSPACE=/home/wangpch/TG-Interpolation
ROOT=${WORKSPACE}/artifacts/experiment/scaleup_nonfineweb_multiseed_20260815
ROW=$(sed -n "$((INDEX + 2))p" "${ROOT}/smoke.tsv")
if [ -z "${ROW}" ]; then
    echo "SMOKE_INDEX_OUT_OF_RANGE index=${INDEX}" >&2
    exit 2
fi
IFS=$'\t' read -r _ RUN_NAME MODEL_ID PAPER_NAME SCALE GRAMMAR TASK SEED CHECKPOINT CONFIG RUN_DIR SAVE PRECISION STRATEGY MICROBATCH EVAL_BATCH PAPER_REFERENCE <<< "${ROW}"

SMOKE_ROOT=/tmp/scaleup_nonfineweb_multiseed_20260815/smoke/${MODEL_ID}
case "${SMOKE_ROOT}" in
    /tmp/scaleup_nonfineweb_multiseed_20260815/smoke/*) ;;
    *) echo "UNSAFE_SMOKE_PATH ${SMOKE_ROOT}" >&2; exit 3 ;;
esac
rm -rf "${SMOKE_ROOT}"
mkdir -p "${SMOKE_ROOT}" "${ROOT}/smoke_logs"
exec > >(tee -a "${ROOT}/smoke_logs/${MODEL_ID}.log") 2>&1
cd "${WORKSPACE}"

export PYTHONPATH=${WORKSPACE}
export MPLCONFIGDIR=${SMOKE_ROOT}/mpl
export WANDB_MODE=disabled
export OLMO_FLEX_ATTENTION_NUM_STAGES=1
export OLMO_FLEX_ATTENTION_PAD_TO_MULTIPLE=128
TORCHRUN=/home/wangpch/.conda/envs/LLM/bin/torchrun

echo "SMOKE_START index=${INDEX} model=${MODEL_ID} scale=${SCALE} strategy=${STRATEGY} $(date --iso-8601=seconds)"
PORT=$((16000 + (${SLURM_ARRAY_JOB_ID:-1000} + ${SLURM_ARRAY_TASK_ID:-INDEX}) % 35000))
"${TORCHRUN}" \
    --master_port="${PORT}" \
    --nproc-per-node=4 \
    scripts/train.py \
    "${CONFIG}" \
    --run_name="smoke_${MODEL_ID}" \
    --workspace="${WORKSPACE}" \
    --save_folder="${SMOKE_ROOT}/checkpoints" \
    --save_overwrite=true \
    --try_load_latest_save=false \
    --load_path="${CHECKPOINT}" \
    --max_duration=1 \
    --stop_after=1 \
    --evaluators='[]' \
    --eval_no_save=true \
    --save_data_indices=false \
    --wandb.mode=disabled

touch "${ROOT}/smoke_logs/${MODEL_ID}.done"
rm -rf "${SMOKE_ROOT}"
echo "SMOKE_DONE index=${INDEX} model=${MODEL_ID} $(date --iso-8601=seconds)"
