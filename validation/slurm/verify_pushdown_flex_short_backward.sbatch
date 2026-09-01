#!/bin/bash
#SBATCH --job-name=push-flex-short
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=validation/slurm/pushdown-flex-short-backward_%j.out
#SBATCH --error=validation/slurm/pushdown-flex-short-backward_%j.err

set -uo pipefail

REPO=/home/wangpch/TG-Interpolation
FLEX_ENV_NAME=${FLEX_ENV_NAME:-LLM}
FLEX_LABEL=${FLEX_LABEL:-baseline}

cd "$REPO" || exit 2
source /data/software/anaconda3/etc/profile.d/conda.sh
conda activate "$FLEX_ENV_NAME"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/push-flex-xdg-${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/push-flex-inductor-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/push-flex-triton-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1

echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname)"
echo "environment=${FLEX_ENV_NAME}"
echo "python=$(command -v python)"
python -c 'import torch, triton; print(f"torch={torch.__version__} triton={triton.__version__}")'

python -m pytest -q \
    tests/test_pushdown_flex_parity.py::test_pushdown_flex_backward \
    --junitxml="validation/slurm/results/pushdown-flex-short-backward-${FLEX_LABEL}_${SLURM_JOB_ID}.xml"
