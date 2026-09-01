#!/bin/bash
#SBATCH --job-name=attention-router
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=validation/slurm/attention-router_%j.out
#SBATCH --error=validation/slurm/attention-router_%j.err

set -uo pipefail

REPO=/home/wangpch/TG-Interpolation
cd "$REPO" || exit 2
source /data/software/anaconda3/etc/profile.d/conda.sh
conda activate LLM

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/attention-router-xdg-${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/attention-router-inductor-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/attention-router-triton-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
export OLMO_ATTENTION_ROUTER_DIAGNOSTICS=1

pytest -q \
    tests/test_attention_backend_router.py \
    tests/test_pushdown_flex_parity.py::test_pushdown_flex_backward \
    --junitxml="validation/slurm/results/attention-router_${SLURM_JOB_ID}.xml"
