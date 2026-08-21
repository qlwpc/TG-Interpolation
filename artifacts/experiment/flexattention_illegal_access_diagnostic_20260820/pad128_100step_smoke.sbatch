#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=1M
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:4
#SBATCH --job-name=flex-pad128-100step
#SBATCH --output=/home/wangpch/TG-Interpolation/artifacts/experiment/flexattention_illegal_access_diagnostic_20260820/slurm-%j.out

set -euo pipefail

WORKSPACE=/home/wangpch/TG-Interpolation
DIAG_ROOT=${WORKSPACE}/artifacts/experiment/flexattention_illegal_access_diagnostic_20260820
CONFIG=${WORKSPACE}/artifacts/experiment/scaleup_nonfineweb_multiseed_20260815/runs/tg_100m_xsum_seed6198/train_config.yaml
CHECKPOINT=${WORKSPACE}/saved_models/TG_test/step55457-unsharded
JOB_TAG=${SLURM_JOB_ID:-manual}
SCRATCH=/tmp/flexattention_pad128_100step/${JOB_TAG}
TORCHRUN=/home/wangpch/.conda/envs/LLM/bin/torchrun

mkdir -p "${DIAG_ROOT}" "${SCRATCH}/torchinductor" "${SCRATCH}/checkpoints" "${SCRATCH}/mpl"
exec > >(tee "${DIAG_ROOT}/pad128-100step-${JOB_TAG}.log") 2>&1
cd "${WORKSPACE}"

export PYTHONPATH=${WORKSPACE}
export MPLCONFIGDIR=${SCRATCH}/mpl
export WANDB_MODE=disabled
export OLMO_FLEX_ATTENTION_NUM_STAGES=1
export OLMO_FLEX_ATTENTION_PAD_TO_MULTIPLE=128
export TORCHINDUCTOR_CACHE_DIR=${SCRATCH}/torchinductor

echo "FLEX_PAD128_100STEP_START job=${JOB_TAG} time=$(date --iso-8601=seconds)"
hostname
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader
/home/wangpch/.conda/envs/LLM/bin/python -c 'import torch, triton; print(f"torch={torch.__version__} cuda={torch.version.cuda} triton={triton.__version__}")'

PORT=$((26000 + JOB_TAG % 20000))
set +e
"${TORCHRUN}" \
    --master_port="${PORT}" \
    --nproc-per-node=4 \
    scripts/train.py \
    "${CONFIG}" \
    --run_name=flexdiag_pad128_100step_tg100m_xsum_seed6198 \
    --workspace="${WORKSPACE}" \
    --save_folder="${SCRATCH}/checkpoints" \
    --save_overwrite=true \
    --try_load_latest_save=false \
    --load_path="${CHECKPOINT}" \
    --evaluators='[]' \
    --wandb.mode=disabled \
    --device_train_microbatch_size=5 \
    --console_log_interval=1 \
    --save_interval_unsharded=1000000 \
    --max_duration=100
STATUS=$?
set -e

echo "FLEX_PAD128_100STEP_END job=${JOB_TAG} status=${STATUS} time=$(date --iso-8601=seconds)"
exit "${STATUS}"
