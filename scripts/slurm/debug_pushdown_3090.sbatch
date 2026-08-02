#!/bin/bash
#SBATCH --job-name=pushdown-debug
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem=1M
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/pushdown_debug_%x_%j.log

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
export OLMO_PUSHDOWN_PHASE_PROFILE=1
export OLMO_GRAD_NAN_PROFILE=1

debug_variant=${1:-baseline}
debug_overrides=("global_train_batch_size=4" "device_train_microbatch_size=4" "stop_at=4" "speed_monitor.window_size=3")
case "${debug_variant}" in
  baseline)
    export OLMO_PUSHDOWN_FIX2_PROFILE=1
    ;;
  detach)
    export OLMO_PUSHDOWN_DETACH=1
    debug_overrides+=("ddp.find_unused_params=true" "ddp.grad_sync_mode=micro_batch")
    ;;
  no_attachment)
    export OLMO_PUSHDOWN_SKIP_ATTACHMENT=1
    debug_overrides+=("ddp.find_unused_params=true" "ddp.grad_sync_mode=micro_batch")
    ;;
  *)
    echo "unknown debug variant: ${debug_variant}" >&2
    exit 2
    ;;
esac

run_dir="artifacts/experiment/pushdown-gpu-opt-20260802/runtime/debug-${debug_variant}-${SLURM_JOB_ID}"
mkdir -p "${run_dir}" logs

/home/wangpch/.conda/envs/LLM/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=1 \
  scripts/train.py \
  train_configs/pushdown_profile_3090.yaml \
  "run_name=pushdown-debug-${debug_variant}-${SLURM_JOB_ID}" \
  "save_folder=${run_dir}" \
  "${debug_overrides[@]}"
