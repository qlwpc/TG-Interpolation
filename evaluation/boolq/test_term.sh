#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 1
#SBATCH --gres=gpu:4
#SBA --mem-per-cpu=16384
#SBATCH -t 80:00:00
#SBA --partition=critical
#SBA -A tukw-critical
# --exclude=ai_gpu31

workspace=${HOME}/TG-Interpolation
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=${PYTHONPATH}:${workspace}
nvidia-smi
wandb offline
cd ${workspace}
run_name=Terminal_finetune_boolq
torchrun --nproc-per-node=4 --master_port 15590 scripts/train.py \
  ${workspace}/evaluation/boolq/terminal.yaml \
  --run_name=${run_name} \
  --save_folder=${workspace}/saved_models/test_models/${run_name} \
  --workspace=${workspace} \
  --load_path=/home/wangpch/TG-Interpolation/saved_models/Terminal-lr005-bs144/step34115-unsharded