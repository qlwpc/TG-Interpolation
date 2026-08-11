#!/usr/bin/env bash
# Pre-train GPST-small UNSUPERVISED (hard-EM, parser-induced trees) on a
# tokenized lazy corpus. Adapt corpus path + GPU count to your setup.
#
# 1) Preprocess corpus first:
#    python scripts/gpst/preprocess_corpus.py --mode raw \
#      --raw_corpus_path PATH_TO_OPENWEBTEXT \
#      --tokenizer_path olmo/gpst/data/gpt2-small \
#      --output_path corpus/openwebtext.lazy
#
# 2) Launch (8 GPUs):
#    sbatch with -c 1 --mem-per-cpu=1M --gres=gpu:8  (local cluster)
#    or torchrun directly below.

set -e
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH}:${HOME}/TG-Interpolation"

CORPUS_PATH="${1:-corpus/bbc-tree.lazy}"
OUTPUT_DIR="${2:-saved_models/gpst-bbc-unsup}"
NUM_SAMPLES="${3:-2079744}"
BACKBONE="${4:-olmo}"

torchrun --standalone --nnodes=1 --nproc-per-node=8 \
    scripts/gpst/run_gpst.py \
    --unsupervised \
    --corpus_path "$CORPUS_PATH" \
    --r2d2_config_path olmo/gpst/data/en_config/r2d2_256_4_1.json \
    --gpt_config_path olmo/gpst/data/gpt2-bbc/config.json \
    --vocab_dir olmo/gpst/data/gpt2-small \
    --output_dir "$OUTPUT_DIR" \
    --backbone "$BACKBONE" \
    --batch_size 32 \
    --accumulation_steps 1 \
    --num_samples "$NUM_SAMPLES" \
    --max_seq_len 2048 \
    --lr 5e-5 --parser_lr 1e-3 \
    --warmup 0.01 \
    --log_steps 50 --save_steps 10000 \
    --gradient_checkpoint
