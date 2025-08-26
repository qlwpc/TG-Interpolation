#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH -t 120:00:00
#SBATCH --mem-per-cpu=32768
#SBATCH --partition=critical
#SBATCH -A tukw-critical
#SBATCH --exclude=ai_gpu[26-35]

workspace=${HOME}/TG-Interpolation
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=${PYTHONPATH}:${workspace}
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# CUDA_LAUNCH_BLOCKING=1
nvidia-smi
wandb offline
cd ${workspace}
run_name=tree_test_SG
torchrun --nproc-per-node=1 --master_port 15585 scripts/train.py \
  ${workspace}/evaluation/eval_configs/SG.yaml \
  --run_name=${run_name} \
  --save_folder=${workspace}/saved_models/test_models/${run_name} \
  --workspace=${workspace} \
  --load_path=/public/home/wangpch/TG-Interpolation/saved_models/nomask_test/step55853-unsharded
#   --python_profiling=true \
#   --torch_profiling=true