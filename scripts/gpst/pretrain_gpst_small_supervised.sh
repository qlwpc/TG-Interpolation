#!/usr/bin/env bash
# Pre-train GPST-small SUPERVISED (gold constituency trees) on the repo's BBC
# tree-stream corpus. Gold merge orders are derived from the parse trees and fed
# directly to the composition model; the parser is trained supervised (NLL on
# gold split order) but does not induce the tree.
#
# Launch (8 GPUs):
#    sbatch with -c 1 --mem-per-cpu=1M --gres=gpu:8  (local cluster)

set -e
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH}:${HOME}/TG-Interpolation}"

TREE_NPY="${1:-dataset/bbc-news/tree/train.npy}"
TOKENIZER="${2:-dataset/bbc-news/TG_GPT2_tokenizer.json}"
OUTPUT_DIR="${3:-saved_models/gpst-bbc-sup}"
NUM_SAMPLES="${4:-1000000}"
BACKBONE="${5:-olmo}"

torchrun --standalone --nnodes=1 --nproc-per-node=8 \
    scripts/gpst/run_gpst.py \
    --supervised \
    --tree_npy "$TREE_NPY" \
    --tokenizer_path "$TOKENIZER" \
    --r2d2_config_path olmo/gpst/data/en_config/r2d2_256_4_1.json \
    --gpt_config_path olmo/gpst/data/gpt2-small/config.json \
    --vocab_dir olmo/gpst/data/gpt2-small \
    --output_dir "$OUTPUT_DIR" \
    --backbone "$BACKBONE" \
    --batch_size 32 \
    --accumulation_steps 1 \
    --num_samples "$NUM_SAMPLES" \
    --max_seq_len 1024 \
    --lr 5e-5 --parser_lr 1e-3 \
    --warmup 0.01 \
    --log_steps 50 --save_steps 10000 \
    --gradient_checkpoint
