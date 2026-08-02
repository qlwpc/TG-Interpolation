#!/bin/bash
#SBATCH --job-name=pushdown-profile
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem=1M
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=logs/pushdown_profile_%x_%j.log

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
profile_overrides=()
case "${profile_variant}" in
  baseline)
    ;;
  detach)
    export OLMO_PUSHDOWN_DETACH=1
    profile_overrides+=("ddp.find_unused_params=true" "ddp.grad_sync_mode=micro_batch")
    ;;
  no_bias)
    export OLMO_PUSHDOWN_NO_BIAS=1
    profile_overrides+=("ddp.find_unused_params=true" "ddp.grad_sync_mode=micro_batch")
    ;;
  no_attachment)
    export OLMO_PUSHDOWN_SKIP_ATTACHMENT=1
    export OLMO_GRAD_NAN_PROFILE=1
    profile_overrides+=("ddp.find_unused_params=true" "ddp.grad_sync_mode=micro_batch")
    ;;
  focused)
    export OLMO_PUSHDOWN_FIX2_PROFILE=1
    export OLMO_PUSHDOWN_PHASE_PROFILE=1
    export OLMO_GRAD_NAN_PROFILE=1
    profile_overrides+=("stop_at=4" "speed_monitor.window_size=3")
    ;;
  *)
    echo "unknown profile variant: ${profile_variant}" >&2
    exit 2
    ;;
esac

run_dir="artifacts/experiment/pushdown-gpu-opt-20260802/runtime/${profile_variant}-${SLURM_JOB_ID}"
mkdir -p "${run_dir}" logs

date --iso-8601=seconds
nvidia-smi
/home/wangpch/.conda/envs/LLM/bin/python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)'

/home/wangpch/.conda/envs/LLM/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  scripts/train.py \
  train_configs/pushdown_profile_3090.yaml \
  "run_name=pushdown-profile-${profile_variant}-${SLURM_JOB_ID}" \
  "save_folder=${run_dir}" \
  "${profile_overrides[@]}"

date --iso-8601=seconds
