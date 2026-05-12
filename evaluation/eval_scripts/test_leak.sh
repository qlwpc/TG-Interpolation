#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -t 120:00:00
#SBATCH --mem-per-cpu=1
#SBATC --partition=critical
#SBATC -A tukw-critical
#SBATC --exclude=ai_gpu[26-35],ai_gpu[01-13],sist-a40-0[6-9]
workspace=${HOME}/TG-Interpolation
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=${PYTHONPATH}:${workspace}
export PYTHONMALLOC=malloc
# export LD_PRELOAD=/home/wangpch/.conda/envs/LLM/lib/libasan.so
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi

cd ${workspace}
run_name=nomask_test_ppl
# valgrind --leak-check=full --show-leak-kinds=definite,possible --track-origins=yes --suppressions=valgrind-python.supp \
    python mem.py
# torchrun --nproc-per-node=1 --master_port 15596 scripts/train.py \
#   ${workspace}/evaluation/eval_configs/nomask.yaml \
#   --run_name=${run_name} \
#   --save_folder=${workspace}/saved_models/test_models/${run_name} \
#   --workspace=${workspace} \
#   --load_path=/home/wangpch/TG-Interpolation/saved_models/test_models/Nomask_finetune_xsum_fixed_6e-5_SFT/step1341-unsharded
  # --python_profiling=true \
  # --torch_profiling=true