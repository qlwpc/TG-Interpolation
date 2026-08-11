#!/bin/bash
#SBATCH --job-name=gpst-bbc-unsup
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1M
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --output=logs/gpst_bbc_pretrain_%j.log

# GPST unsupervised pretrain, aligned with train_configs/terminal.yaml (OLMo-300M):
#   - data:    BBC tree-stream → lazy (terminal + sentence boundaries)
#   - model:   768/12/12, vocab 50320, swiglu/rms/flash (RoPE off, GPST constraint)
#   - budget:  num_samples 2.08M × 8GPU × batch32 × seq2048 ≈ 4.26B tokens (0.42 ep, matches baseline)
# CLAUDE.md: local cluster requires -c 1 --mem-per-cpu=1M; GPU via --gres; do NOT set CUDA_VISIBLE_DEVICES manually.

set -euo pipefail

cd /home/wangpch/TG-Interpolation

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/wangpch/TG-Interpolation${PYTHONPATH:+:$PYTHONPATH}

CORPUS_PATH="${1:-corpus/bbc-tree.lazy}"
OUTPUT_DIR="${2:-saved_models/gpst-bbc-unsup}"
NUM_SAMPLES="${3:-2079744}"

torchrun --standalone --nnodes=1 --nproc-per-node=8 \
    scripts/gpst/run_gpst.py \
    --unsupervised \
    --corpus_path "$CORPUS_PATH" \
    --r2d2_config_path olmo/gpst/data/en_config/r2d2_256_4_1.json \
    --gpt_config_path olmo/gpst/data/gpt2-bbc/config.json \
    --vocab_dir olmo/gpst/data/gpt2-small \
    --output_dir "$OUTPUT_DIR" \
    --backbone olmo \
    --batch_size 32 \
    --accumulation_steps 1 \
    --num_samples "$NUM_SAMPLES" \
    --max_seq_len 2048 \
    --lr 5e-5 --parser_lr 1e-3 \
    --warmup 0.01 \
    --log_steps 50 --save_steps 10000 \
    --gradient_checkpoint
