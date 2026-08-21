#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem-per-cpu=2G
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=flex-pad128-parity
#SBATCH --output=/home/wangpch/TG-Interpolation/artifacts/experiment/flexattention_illegal_access_diagnostic_20260820/pad128-parity-%j.log

set -euo pipefail

WORKSPACE=/home/wangpch/TG-Interpolation
DIAG_ROOT=${WORKSPACE}/artifacts/experiment/flexattention_illegal_access_diagnostic_20260820
JOB_TAG=${SLURM_JOB_ID:-manual}
SCRATCH=/tmp/flexattention_pad128_parity/${JOB_TAG}

mkdir -p "${SCRATCH}/torchinductor" "${SCRATCH}/mpl"
cd "${WORKSPACE}"

export PYTHONPATH=${WORKSPACE}
export MPLCONFIGDIR=${SCRATCH}/mpl
export TORCHINDUCTOR_CACHE_DIR=${SCRATCH}/torchinductor
export OLMO_FLEX_ATTENTION_NUM_STAGES=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1

echo "FLEX_PAD128_PARITY_START job=${JOB_TAG} time=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader
/home/wangpch/.conda/envs/LLM/bin/python -c 'import torch, triton; print(f"torch={torch.__version__} cuda={torch.version.cuda} triton={triton.__version__}")'
/home/wangpch/.conda/envs/LLM/bin/python "${DIAG_ROOT}/validate_pad128_correctness.py"
echo "FLEX_PAD128_PARITY_END job=${JOB_TAG} status=0 time=$(date --iso-8601=seconds)"
