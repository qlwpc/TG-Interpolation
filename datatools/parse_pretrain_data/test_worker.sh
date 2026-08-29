#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=1M
#SBATCH --gres=gpu:1
#SBATCH -t 180:00:00
#SBATC --exclude=ai_gpu[26-33]

python worker.py