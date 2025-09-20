#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 8
#SBATCH --gres=gpu:4
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
run_name=nomask_bs420_lr009
torchrun --nproc-per-node=4 --master_port 15596 scripts/train.py \
  ${workspace}/train_configs/nomask.yaml \
  --run_name=${run_name} \
  --save_folder=${workspace}/saved_models/${run_name} \
  --workspace=${workspace} 
