#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=1M
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:4
#SBATCH --job-name=flexdiag-exact
#SBATCH --output=/home/wangpch/TG-Interpolation/artifacts/experiment/flexattention_illegal_access_diagnostic_20260820/slurm-%j.out

set -euo pipefail

WORKSPACE=/home/wangpch/TG-Interpolation
DIAG_ROOT=${WORKSPACE}/artifacts/experiment/flexattention_illegal_access_diagnostic_20260820
CONFIG=${WORKSPACE}/artifacts/experiment/scaleup_nonfineweb_multiseed_20260815/runs/tg_100m_xsum_seed6198/train_config.yaml
CHECKPOINT=${WORKSPACE}/saved_models/TG_test/step55457-unsharded
JOB_TAG=${SLURM_JOB_ID:-manual}
SCRATCH=/tmp/flexattention_illegal_access_diagnostic_20260820/${JOB_TAG}
TORCHRUN=/home/wangpch/.conda/envs/LLM/bin/torchrun

mkdir -p "${DIAG_ROOT}" "${SCRATCH}/torchinductor" "${SCRATCH}/checkpoints" "${SCRATCH}/mpl"
exec > >(tee "${DIAG_ROOT}/exact-repro-${JOB_TAG}.log") 2>&1
cd "${WORKSPACE}"

export PYTHONPATH=${WORKSPACE}
export MPLCONFIGDIR=${SCRATCH}/mpl
export WANDB_MODE=disabled
export OLMO_FLEX_ATTENTION_NUM_STAGES=1
export OLMO_FLEX_ATTENTION_DIAGNOSTICS=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCHINDUCTOR_CACHE_DIR=${SCRATCH}/torchinductor

echo "FLEX_DIAG_JOB_START job=${JOB_TAG} time=$(date --iso-8601=seconds)"
hostname
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader
/home/wangpch/.conda/envs/LLM/bin/python -c 'import torch, triton; print(f"torch={torch.__version__} cuda={torch.version.cuda} triton={triton.__version__}")'

PORT=$((22000 + JOB_TAG % 20000))
set +e
"${TORCHRUN}" \
    --master_port="${PORT}" \
    --nproc-per-node=4 \
    scripts/train.py \
    "${CONFIG}" \
    --run_name=flexdiag_exact_tg100m_xsum_seed6198 \
    --workspace="${WORKSPACE}" \
    --save_folder="${SCRATCH}/checkpoints" \
    --save_overwrite=true \
    --try_load_latest_save=false \
    --load_path="${CHECKPOINT}" \
    --evaluators='[]' \
    --wandb.mode=disabled \
    --device_train_microbatch_size=5 \
    --max_duration=1
STATUS=$?
set -e

echo "FLEX_DIAG_JOB_END job=${JOB_TAG} status=${STATUS} time=$(date --iso-8601=seconds)"
exit "${STATUS}"
