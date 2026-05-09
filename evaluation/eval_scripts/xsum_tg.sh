#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=1
#SBATCH -t 80:00:00

workspace=${HOME}/TG-Interpolation
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=${PYTHONPATH}:${workspace}
nvidia-smi
wandb offline
cd ${workspace}
run_name=TG_finetune_xsum_fixed_6e-5_SFT
torchrun --nproc-per-node=4 --master_port 15597 scripts/train.py \
  ${workspace}/evaluation/xsum_configs/TG.yaml \
  --run_name=${run_name} \
  --save_folder=${workspace}/saved_models/test_models/${run_name} \
  --workspace=${workspace} \
  --load_path=/home/wangpch/TG-Interpolation/saved_models/TG_test/step55457-unsharded