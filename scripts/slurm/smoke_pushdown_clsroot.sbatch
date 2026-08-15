#!/bin/bash
#SBATCH --job-name=pushdown-clsroot-smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/pushdown_clsroot_smoke_%j.log

set -euo pipefail

cd /home/wangpch/TG-Interpolation

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH=/home/wangpch/TG-Interpolation${PYTHONPATH:+:$PYTHONPATH}

/home/wangpch/.conda/envs/LLM/bin/torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  scripts/train.py train_configs/pushdown_smoke.yaml
