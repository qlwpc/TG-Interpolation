#!/bin/bash
#SBATCH --job-name=gpst-tree300-docppl
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem=1M
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=logs/gpst_tree300_docppl_%j.log

# Exact gold-tree, document-level PPL over all 300 supplied trees per sentence.
set -euo pipefail

cd /home/wangpch/TG-Interpolation
export PYTHONPATH=/home/wangpch/TG-Interpolation${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

/home/wangpch/.conda/envs/LLM/bin/python scripts/gpst/evaluate_document_ppl.py \
  --checkpoint saved_models/gpst-bbc-unsup/model.bin \
  --tree-data dataset/testppl_tree/tree_300.npy \
  --sentence-index dataset/testppl_tree/tree_sent_index.npy \
  --document-index dataset/testppl_tree/tree_doc_index.npy \
  --tokenizer-path dataset/bbc-news/TG_GPT2_tokenizer.json \
  --backbone olmo \
  --samples-per-sentence 300 \
  --eval-batch-size 4 \
  --max-action-nodes 2048 \
  --max-terminals 2048 \
  --device cuda \
  --log-every 10
