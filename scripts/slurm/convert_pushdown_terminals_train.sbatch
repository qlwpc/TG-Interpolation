#!/bin/bash
#SBATCH --job-name=pushdown-terminals
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pushdown_terminals_convert_%j.log

set -euo pipefail

cd /home/wangpch/TG-Interpolation

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

/home/wangpch/.conda/envs/LLM/bin/python \
  scripts/convert_treereg_to_pushdown_terminals.py \
  --source-dir dataset/bbc-news/parse_aligned/train_treereg \
  --out-dir dataset/bbc-news/parse_aligned/train_pushdown_unary_terminals \
  --direction right \
  --workers 16 \
  --min-free-disk-gb 256
