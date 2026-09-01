#!/usr/bin/env bash
#SBATCH --job-name=pushdown-cand0-effect
#SBATCH --partition=normal
#SBATCH --nodelist=rtx3090b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/pushdown-cand0-effect-%j.log
#SBATCH --error=logs/pushdown-cand0-effect-%j.log

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
export MPLCONFIGDIR=/tmp/pushdown-cand0-effect-mpl

/home/wangpch/.conda/envs/LLM/bin/python \
  scripts/diagnose_pushdown_candidate0_topology_effect.py \
  --checkpoint saved_models/pushdown_terminalonly/step34354-unsharded \
  --max-sentences 200 \
  --device cuda \
  --output artifacts/analysis/pushdown_candidate_axis_200sent_20260830/candidate0_topology_effect.json
