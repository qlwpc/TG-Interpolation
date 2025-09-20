#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --partition=critical
#SBATCH -A tukw-critical
#SBATCH --exclude=ai_gpu[26-35]

workspace=${HOME}/TG-Interpolation
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=${PYTHONPATH}:${workspace}
python test_flexattn.py