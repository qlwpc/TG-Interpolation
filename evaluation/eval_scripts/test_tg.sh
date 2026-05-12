#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -t 80:00:00
#SBATCH --partition=critical
#SBATCH -A tukw-critical
#SBATCH --exclude=ai_gpu[26-35]

workspace=${HOME}/TG-Interpolation
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=${PYTHONPATH}:${workspace}
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi

cd ${workspace}
run_name=tg_test_ppl
torchrun --nproc-per-node=1 --master_port 15594 scripts/train.py \
  ${workspace}/evaluation/eval_configs/tg.yaml \
  --run_name=${run_name} \
  --save_folder=${workspace}/saved_models/test_models/${run_name} \
  --workspace=${workspace} \
  --load_path=/public/home/wangpch/TG-Interpolation/saved_models/TG_test/step55457-unsharded
  # --python_profiling=true \
  # --torch_profiling=true