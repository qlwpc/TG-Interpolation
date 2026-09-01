#!/usr/bin/env bash
#SBATCH --job-name=pushdown-ppl-2x2-pilot
#SBATCH --partition=normal
#SBATCH --nodelist=rtx3090b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/experiment/pushdown-ppl-2x2-pilot-20260830/slurm-%j.log

# Sentence-local auxiliary comparison:
#   parse support: beam top-300 vs supplied teacher-forced 300 trees
#   attachment normalization: v1 stack-legal vs v2 sentence-causal
# All cells sum unique joint mass without dividing by K and normalize PPL by
# the shared number of sentence-content terminal tokens.
set -euo pipefail

cd /home/wangpch/TG-Interpolation
export PYTHONPATH=/home/wangpch/TG-Interpolation${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export OLMO_FLEX_ATTENTION_NUM_STAGES=1

date -Is
hostname
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

/home/wangpch/.conda/envs/LLM/bin/python scripts/evaluate_pushdown_ppl_2x2.py \
  --checkpoint saved_models/pushdown_terminalonly/step34354-unsharded \
  --max-sentences 10 \
  --beam-size 300 \
  --eval-batch-size 32 \
  --max-batch-tokens 65536 \
  --training-attachment-normalization sentence_causal \
  --device cuda \
  --progress-every 1 \
  --include-per-sentence \
  --output artifacts/experiment/pushdown-ppl-2x2-pilot-20260830/result.json

date -Is
