#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 2
#SBATCH --gres=gpu:4
#SBATCH -t 120:00:00
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=critical
#SBATCH -A tukw-critical
# --exclude=ai_gpu[26-35]

workspace=${HOME}/TG-Interpolation
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=${PYTHONPATH}:${workspace}
nvidia-smi

cd ${workspace}
run_name=Terminal_test_xsum
torchrun --nproc-per-node=4 --master_port 15590 scripts/train.py \
  ${workspace}/evaluation/eval_configs/terminal.yaml \
  --run_name=${run_name} \
  --save_folder=${workspace}/saved_models/test_models/${run_name} \
  --workspace=${workspace} \
  --load_path=/public/home/wangpch/TG-Interpolation/saved_models/test_models/Terminal_finetune_xsum_lr3e-4_warmup100_2ep/step896-unsharded