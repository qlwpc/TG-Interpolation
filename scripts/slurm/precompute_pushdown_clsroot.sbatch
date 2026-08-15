#!/bin/bash
#SBATCH --job-name=pushdown-clsroot-data
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
# The normal QOS caps each job at two CPUs. Keep one CPU free for the parent
# process/I/O while one pinned parser worker performs the sequential parse.
#SBATCH --cpus-per-task=2
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pushdown_clsroot_preprocess_%j.log

set -euo pipefail

cd /home/wangpch/TG-Interpolation

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

/home/wangpch/.conda/envs/LLM/bin/python scripts/precompute_pushdown_unary.py \
  --split train \
  --tree-dir dataset/tree \
  --tokenizer dataset/bbc-news/TG_GPT2_tokenizer.json \
  --out-dir dataset/bbc-news/parse_aligned/train_pushdown_unary \
  --direction right \
  --root-token-id 50260 \
  --sentence-eos-token-id 50256 \
  --workers 1 \
  --max-workers 1 \
  --scan-workers 1 \
  --max-spans 4096 \
  --load-tree-to-ram \
  --min-free-memory-gb 64 \
  --min-free-disk-gb 256
