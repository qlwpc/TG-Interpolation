#!/bin/bash
#SBATCH --job-name=real-tg-pad128
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=validation/slurm/real-tg-pad128_%j.out
#SBATCH --error=validation/slurm/real-tg-pad128_%j.err

set -uo pipefail

REPO=/home/wangpch/TG-Interpolation
cd "$REPO" || exit 2
source /data/software/anaconda3/etc/profile.d/conda.sh
conda activate LLM

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/real-tg-pad128-xdg-${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/real-tg-pad128-inductor-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/real-tg-pad128-triton-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1

python validation/slurm/verify_real_tg_pad128.py
