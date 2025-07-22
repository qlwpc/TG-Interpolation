#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=1M
#SBATCH --gres=gpu:1
#SBATCH -t 180:00:00

export HF_ENDPOINT=https://hf-mirror.com
nvidia-smi
python parse_testppl.py