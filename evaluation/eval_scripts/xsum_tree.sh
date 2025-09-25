#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=1
#SBATCH -t 80:00:00

workspace=${HOME}/TG-Interpolation
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=${PYTHONPATH}:${workspace}
nvidia-smi
wandb offline
cd ${workspace}
run_name=Tree_finetune_xsum_lr1e-4_warmup50_2ep
torchrun --nproc-per-node=2 --master_port 15590 scripts/train.py \
  ${workspace}/evaluation/xsum_configs/tree.yaml \
  --run_name=${run_name} \
  --save_folder=${workspace}/saved_models/test_models/${run_name} \
  --workspace=${workspace} \
  --load_path=/home/wangpch/TG-Interpolation/saved_models/Tree_test/step49440-unsharded