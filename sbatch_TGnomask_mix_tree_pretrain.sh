#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -n 8
#SBATCH -t 120:00:00
#SBATCH --gres=gpu:8

workspace=${HOME}/TG-Interpolation
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=${PYTHONPATH}:${workspace}

nvidia-smi
wandb offline
cd ${workspace}
run_name=TGnomask_mix_tree_pretrain


date
tar -xvf dataset.tar -C /dev/shm
date

torchrun  \
      --master_port=10826 \
      --nproc-per-node=8 \
scripts/train.py   \
tmp.yaml   \
      --run_name=${run_name} \
      --workspace=${workspace} \
      --save_folder=${workspace}/saved_models/${run_name} \