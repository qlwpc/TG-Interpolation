#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -c 8
#SBATCH --gres=gpu:8
#SBATCH --time=96:00:00

workspace=${HOME}/TG-Interpolation
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=${PYTHONPATH}:${workspace}
nvidia-smi
wandb offline
cd ${workspace}
date
tar -xvf dataset.tar -C /dev/shm
date
run_name=TG_mix_nomask_bs240_lr0076
torchrun --nproc-per-node=8 --master_port 15591 scripts/train.py \
  ${workspace}/train_configs/nomask_and_tg.yaml \
  --run_name=${run_name} \
  --save_folder=${workspace}/saved_models/${run_name} \
  --workspace=${workspace} 
