#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH -t 120:00:00
#SBATCH --mem-per-cpu=32768
#SBATCH --partition=critical
#SBATCH -A tukw-critical
# --exclude=ai_gpu[26-35]

workspace=${HOME}/TG-Interpolation
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=${PYTHONPATH}:${workspace}
nvidia-smi
wandb offline
cd ${workspace}
run_name=TG_mix_nomask_test
torchrun --nproc-per-node=1 --master_port 15591 scripts/train.py \
  ${workspace}/train_configs/nomask_and_tg.yaml \
  --run_name=${run_name} \
  --save_folder=${workspace}/saved_models/${run_name} \
  --workspace=${workspace} 
