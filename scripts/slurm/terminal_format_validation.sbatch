#!/bin/bash
#SBATCH --job-name=terminal-format-validation
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=artifacts/experiment/terminal-format-validation-20260828/slurm-%x-%j.log
#SBATCH --error=artifacts/experiment/terminal-format-validation-20260828/slurm-%x-%j.log

set -euo pipefail

ROOT=/home/wangpch/TG-Interpolation
NPROC=${NPROC:-4}
if [[ $# -lt 1 ]]; then
  echo "usage: sbatch terminal_format_validation.sbatch CONFIG [TRAIN_OVERRIDES...]" >&2
  exit 2
fi

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR=/tmp/mpl-terminal-format-validation
export TOKENIZERS_PARALLELISM=false
export OLMO_BOOTSTRAP_DUMP=${OLMO_BOOTSTRAP_DUMP:-1}

/home/wangpch/.conda/envs/LLM/bin/torchrun \
  --standalone --nnodes=1 --nproc-per-node="$NPROC" \
  scripts/train.py "$@"
