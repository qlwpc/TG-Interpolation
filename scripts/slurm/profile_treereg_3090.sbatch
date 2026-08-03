#!/bin/bash
#SBATCH --job-name=treereg-profile
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem=1M
#SBATCH --gres=gpu:8
#SBATCH --time=01:30:00
#SBATCH --output=logs/treereg_profile_%x_%j.log

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
    treereg_every_k=10
    ;;
  lm_control)
    treereg_every_k=100000
    ;;
  focused)
    treereg_every_k=1
    profile_overrides=("stop_at=4" "torch_profiling=true" "speed_monitor.window_size=3")
    ;;
  *)
    echo "unknown profile variant: ${profile_variant}" >&2
    exit 2
    ;;
esac

run_dir="artifacts/experiment/treereg-gpu-opt-20260802/runtime/${profile_variant}-${SLURM_JOB_ID}"
mkdir -p "${run_dir}" logs

date --iso-8601=seconds
nvidia-smi
/home/wangpch/.conda/envs/LLM/bin/python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)'

/home/wangpch/.conda/envs/LLM/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  scripts/train.py \
  train_configs/treereg_profile_3090.yaml \
  "run_name=treereg-profile-${profile_variant}-${SLURM_JOB_ID}" \
  "save_folder=${run_dir}" \
  "model.treereg_every_k=${treereg_every_k}" \
  "${profile_overrides[@]}"

date --iso-8601=seconds
