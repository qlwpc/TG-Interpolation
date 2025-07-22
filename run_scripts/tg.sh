#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 1
#SBATCH --gres=gpu:2
#SBATCH --mem=1M

workspace=${HOME}/TG-Interpolation
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=${PYTHONPATH}:${workspace}
nvidia-smi
wandb offline
cd ${workspace}
run_name=TG_small
torchrun --nproc-per-node=2 --master_port 15591 scripts/train.py \
  ${workspace}/train_configs/TG.yaml \
  --run_name=${run_name} \
  --save_folder=${workspace}/saved_models/${run_name} \
  --workspace=${workspace} 
