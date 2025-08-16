#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -t 60:00:00

workspace=${HOME}/TG-Interpolation
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=${PYTHONPATH}:${workspace}
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi
wandb offline
cd ${workspace}
run_name=tester_tree_small
torchrun --nproc-per-node=1 --master_port 15590 scripts/train.py \
  ${workspace}/evaluation/eval_configs/tree.yaml \
  --run_name=${run_name} \
  --save_folder=${workspace}/saved_models/test_models/${run_name} \
  --workspace=${workspace} \
  --load_path=/home/wangpch/TG-Interpolation/saved_models/Tree_test/step49440-unsharded