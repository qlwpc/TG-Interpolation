#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 8
#SBATCH --gres=gpu:2

workspace=${HOME}/TG-Interpolation
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=${PYTHONPATH}:${workspace}
nvidia-smi
wandb offline
cd ${workspace}
date
tar -xvf dataset.tar -C /dev/shm
date
run_name=TG_test
torchrun --nproc-per-node=2 --master_port 15592 scripts/train.py \
  ${workspace}/train_configs/TG.yaml \
  --run_name=${run_name} \
  --save_folder=${workspace}/saved_models/${run_name} \
  --workspace=${workspace} 
