#!/usr/bin/env bash
#SBATCH --job-name=gpstbin-score-eq
#SBATCH --partition=normal
#SBATCH --nodelist=rtx3090b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/gpstbin-score-eq-%j.log

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

candidate_source=${CANDIDATE_SOURCE:-gpst-strict-binary}
attachment_normalization=${ATTACHMENT_NORMALIZATION:-stack_legal}
output=${OUTPUT:-artifacts/analysis/gpst_binary_pushdown_scorer_equivalence_20260830/result.json}

/home/wangpch/.conda/envs/LLM/bin/python \
  scripts/diagnose_gpst_binary_pushdown_scorer_equivalence.py \
  --checkpoint saved_models/pushdown_terminalonly/step34354-unsharded \
  --sentence 1 \
  --candidates 16 \
  --candidate-source "${candidate_source}" \
  --attachment-normalization "${attachment_normalization}" \
  --device cuda \
  --output "${output}"
