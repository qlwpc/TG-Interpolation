#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 2
#SBATCH --gres=gpu:2
#SBATCH -t 120:00:00
#SBATCH --mem-per-cpu=32768
#SBATCH --partition=critical
#SBATCH -A tukw-critical
#SBATCH --exclude=ai_gpu[26-35]
workspace=${HOME}/TG-Interpolation
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=${PYTHONPATH}:${workspace}
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi

cd ${workspace}
run_name=mix_nomask_tg_test_ppl
torchrun --nproc-per-node=2 --master_port 15596 scripts/train.py \
  ${workspace}/evaluation/eval_configs/nomask_and_tg.yaml \
  --run_name=${run_name} \
  --save_folder=${workspace}/saved_models/test_models/${run_name} \
  --workspace=${workspace} \
  --load_path=/public/home/wangpch/TG-Interpolation/saved_models/TG_mix_nomask_bs240_lr0076/step69817-unsharded
  # --python_profiling=true \
  # --torch_profiling=true