#!/bin/bash
#SBATCH --job-name=flex128-bwd
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1M
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=validation/slurm/flex-attention-128_%j.out
#SBATCH --error=validation/slurm/flex-attention-128_%j.err

set -uo pipefail

REPO=/home/wangpch/TG-Interpolation
FLEX_ENV_NAME=${FLEX_ENV_NAME:-LLM}
FLEX_LABEL=${FLEX_LABEL:-baseline}

cd "$REPO" || exit 2
source /data/software/anaconda3/etc/profile.d/conda.sh
conda activate "$FLEX_ENV_NAME"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/flex128-xdg-${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/flex128-inductor-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/flex128-triton-${SLURM_JOB_ID}"

echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname)"
echo "environment=${FLEX_ENV_NAME}"
echo "python=$(command -v python)"

args=(--label "$FLEX_LABEL")
if [[ -n "${FLEX_NUM_STAGES:-}" ]]; then
    args+=(--num-stages "$FLEX_NUM_STAGES")
fi
python validation/slurm/verify_flex_attention_128.py "${args[@]}"
