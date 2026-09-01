#!/usr/bin/env bash
#SBATCH --job-name=pushdown-axis-metrics
#SBATCH --partition=normal
#SBATCH --nodelist=rtx3090b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/pushdown-axis-metrics-%j.log
#SBATCH --error=logs/pushdown-axis-metrics-%j.log

set -euo pipefail
cd /home/wangpch/TG-Interpolation
export PYTHONPATH=/home/wangpch/TG-Interpolation${PYTHONPATH:+:$PYTHONPATH}
export OLMO_FLEX_ATTENTION_NUM_STAGES=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export MPLCONFIGDIR=/tmp/pushdown-axis-metrics-mpl

/home/wangpch/.conda/envs/LLM/bin/python \
  scripts/diagnose_pushdown_candidate_axis_metrics.py \
  --checkpoint saved_models/pushdown_terminalonly/step34354-unsharded \
  --max-sentences 200 \
  --eval-batch-size 64 \
  --device cuda \
  --output artifacts/analysis/pushdown_candidate_axis_200sent_20260830/decomposed_metrics.json
