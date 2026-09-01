#!/usr/bin/env bash
# Controlled legacy/native n-ary cell for the first-200-sentence comparison.
#SBATCH --job-name=native-nary-ppl-200
#SBATCH --partition=normal
#SBATCH --nodelist=rtx3090b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/native-nary-ppl-200-%j.log
#SBATCH --error=logs/native-nary-ppl-200-%j.log

set -euo pipefail

workspace=/home/wangpch/TG-Interpolation
output_root=${workspace}/artifacts/analysis/pushdown_candidate_axis_200sent_20260830
cd "${workspace}"
mkdir -p "${output_root}" logs
export PYTHONPATH="${workspace}${PYTHONPATH:+:${PYTHONPATH}}"
export OLMO_FLEX_ATTENTION_NUM_STAGES=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export MPLCONFIGDIR="${output_root}/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

date -Is
hostname
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.free --format=csv,noheader
/home/wangpch/.conda/envs/LLM/bin/python scripts/evaluate_pushdown_document_ppl.py \
  --checkpoint saved_models/pushdown_terminalonly/step34354-unsharded \
  --native-data dataset/bbc-news/testppl/native_model_topk_300_v2 \
  --tokenizer-path dataset/bbc-news/TG_GPT2_tokenizer.json \
  --max-sentences 200 \
  --start-document 0 \
  --end-document 4966 \
  --max-sequence-length 2048 \
  --eval-batch-size 64 \
  --max-batch-tokens 65536 \
  --attachment-normalization stack_legal \
  --device cuda \
  --log-every 10 \
  > "${output_root}/native_nary_stack_legal.json"
date -Is
