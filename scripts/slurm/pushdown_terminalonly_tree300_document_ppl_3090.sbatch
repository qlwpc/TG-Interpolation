#!/bin/bash
#SBATCH --job-name=pushdown-terminalonly-tree300-docppl
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem=1M
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=logs/pushdown_terminalonly_tree300_docppl_%j.log

# Exact gold-tree, document-level joint PPL over all 300 supplied trees.
set -euo pipefail

cd /home/wangpch/TG-Interpolation
export PYTHONPATH=/home/wangpch/TG-Interpolation${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

/home/wangpch/.conda/envs/LLM/bin/python scripts/evaluate_pushdown_document_ppl.py \
  --checkpoint saved_models/pushdown_terminalonly/step34354-unsharded \
  --tree-data dataset/testppl_tree/tree_300.npy \
  --sentence-index dataset/testppl_tree/tree_sent_index.npy \
  --document-index dataset/testppl_tree/tree_doc_index.npy \
  --tokenizer-path dataset/bbc-news/TG_GPT2_tokenizer.json \
  --samples-per-sentence 300 \
  --eval-batch-size 4 \
  --max-sequence-length 2048 \
  --device cuda \
  --log-every 10
