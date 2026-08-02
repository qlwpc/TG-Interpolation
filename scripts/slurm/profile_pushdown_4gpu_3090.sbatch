#!/bin/bash
#SBATCH --job-name=pushdown-4gpu
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem=1M
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --output=logs/pushdown_4gpu_%x_%j.log

set -euo pipefail

cd /home/wangpch/TG-Interpolation
export PYTHONPATH=/home/wangpch/TG-Interpolation${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export OLMO_STEP_PROFILE=1

profile_variant=${1:-baseline}
profile_overrides=("global_train_batch_size=16")
case "${profile_variant}" in
  baseline)
    ;;
  detach)
    export OLMO_PUSHDOWN_DETACH=1
    profile_overrides+=("ddp.find_unused_params=true" "ddp.grad_sync_mode=micro_batch")
    ;;
  legacy)
    export OLMO_PUSHDOWN_LEGACY_ORACLE=1
    export OLMO_PUSHDOWN_LEGACY_DEPTH_GRAD=1
    ;;
  *)
    echo "unknown profile variant: ${profile_variant}" >&2
    exit 2
    ;;
esac

run_dir="artifacts/experiment/pushdown-gpu-opt-20260802/runtime/4gpu-${profile_variant}-${SLURM_JOB_ID}"
mkdir -p "${run_dir}" logs

/home/wangpch/.conda/envs/LLM/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  scripts/train.py \
  train_configs/pushdown_profile_3090.yaml \
  "run_name=pushdown-4gpu-${profile_variant}-${SLURM_JOB_ID}" \
  "save_folder=${run_dir}" \
  "${profile_overrides[@]}"
