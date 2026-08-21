#!/bin/bash
set -euo pipefail

INDEX=${1:?usage: run_finetune.sh RUN_INDEX}
WORKSPACE=/home/wangpch/TG-Interpolation
ROOT=${WORKSPACE}/artifacts/experiment/scaleup_nonfineweb_multiseed_20260815
ROW=$(sed -n "$((INDEX + 2))p" "${ROOT}/runs.tsv")
if [ -z "${ROW}" ]; then
    echo "RUN_INDEX_OUT_OF_RANGE index=${INDEX}" >&2
    exit 2
fi
IFS=$'\t' read -r _ RUN_NAME MODEL_ID PAPER_NAME SCALE GRAMMAR TASK SEED CHECKPOINT CONFIG RUN_DIR SAVE PRECISION STRATEGY MICROBATCH EVAL_BATCH PAPER_REFERENCE <<< "${ROW}"

mkdir -p "${RUN_DIR}" "${SAVE}" "${RUN_DIR}/mpl-train"
exec > >(tee -a "${RUN_DIR}/train.log") 2>&1
cd "${WORKSPACE}"

export PYTHONPATH=${WORKSPACE}
export MPLCONFIGDIR=${RUN_DIR}/mpl-train
export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com
export OLMO_FLEX_ATTENTION_NUM_STAGES=1
# PyTorch 2.7.1 / Triton 3.3.1 can access out of bounds in the
# FlexAttention backward tail when Q/KV length is not divisible by 128.
# Pad only the FlexAttention computation and slice back before logits.
export OLMO_FLEX_ATTENTION_PAD_TO_MULTIPLE=128
TORCHRUN=/home/wangpch/.conda/envs/LLM/bin/torchrun

TRAIN_OVERRIDES=()
if [ -n "${DEVICE_TRAIN_MICROBATCH_SIZE_OVERRIDE:-}" ]; then
    if ! [[ "${DEVICE_TRAIN_MICROBATCH_SIZE_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
        echo "INVALID_DEVICE_TRAIN_MICROBATCH_SIZE_OVERRIDE value=${DEVICE_TRAIN_MICROBATCH_SIZE_OVERRIDE}" >&2
        exit 22
    fi
    TRAIN_OVERRIDES+=("--device_train_microbatch_size=${DEVICE_TRAIN_MICROBATCH_SIZE_OVERRIDE}")
fi

echo "TRAIN_START index=${INDEX} run=${RUN_NAME} model=${MODEL_ID} task=${TASK} seed=${SEED} scale=${SCALE} $(date --iso-8601=seconds)"
if [ "${#TRAIN_OVERRIDES[@]}" -gt 0 ]; then
    echo "TRAIN_OVERRIDES ${TRAIN_OVERRIDES[*]}"
fi
hostname
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader

if [ "${TASK}" = xsum ]; then
    echo "15ba739782e8829b2c6d15ccb71898156e02798dc20b7a614d91702213f2c5ad  ${WORKSPACE}/dataset/Xsum/xsum_train.txt" | sha256sum -c -
else
    echo "5a0cc1d6cb971a7a177b74bde27b8355de4b0f0e4d86d0a8435ec92cfeb63ba6  ${WORKSPACE}/dataset/SuperGLUE/BoolQ/train.jsonl" | sha256sum -c -
fi
test -f "${CHECKPOINT}/model.pt"
test -f "${CONFIG}"

if [ -f "${RUN_DIR}/TRAIN_DONE" ] && [ -f "$(cat "${RUN_DIR}/final_checkpoint.txt" 2>/dev/null)/model.pt" ]; then
    echo "TRAIN_ALREADY_COMPLETE run=${RUN_NAME}"
    exit 0
fi

PORT=$((12000 + (${SLURM_ARRAY_JOB_ID:-1000} + ${SLURM_ARRAY_TASK_ID:-INDEX}) % 40000))
"${TORCHRUN}" \
    --master_port="${PORT}" \
    --nproc-per-node=4 \
    scripts/train.py \
    "${CONFIG}" \
    --run_name="${RUN_NAME}" \
    --workspace="${WORKSPACE}" \
    --save_folder="${SAVE}" \
    --save_overwrite=true \
    --try_load_latest_save=true \
    --load_path="${CHECKPOINT}" \
    --evaluators='[]' \
    --wandb.mode=disabled \
    "${TRAIN_OVERRIDES[@]}"

FINAL=$(readlink -f "${SAVE}/latest-unsharded" 2>/dev/null || true)
if [ -z "${FINAL}" ] || [ ! -f "${FINAL}/model.pt" ]; then
    FINAL=$(find "${SAVE}" -maxdepth 1 -type d -name 'step*-unsharded' -printf '%f\t%p\n' | sort -t $'\t' -k1,1V | tail -n 1 | cut -f2-)
fi
if [ -z "${FINAL}" ] || [ ! -f "${FINAL}/model.pt" ]; then
    echo "FINAL_CHECKPOINT_MISSING run=${RUN_NAME} save=${SAVE}" >&2
    exit 21
fi

printf '%s\n' "${FINAL}" > "${RUN_DIR}/final_checkpoint.txt"
sha256sum "${FINAL}/model.pt" > "${RUN_DIR}/final_model.sha256"

# Downstream evaluation reloads only model.pt and config.yaml.  Keeping the
# optimizer/trainer state for all 200 runs would require substantially more
# local scratch space than the campaign has available, so compact a checkpoint
# only after training and the model checksum have both succeeded.
test -f "${FINAL}/config.yaml"
rm -f -- "${FINAL}/optim.pt" "${FINAL}/train.pt"
touch "${RUN_DIR}/TRAIN_DONE"
echo "TRAIN_DONE index=${INDEX} run=${RUN_NAME} checkpoint=${FINAL} $(date --iso-8601=seconds)"
