#!/usr/bin/env bash
# Submit with --dependency=afterok:<native array job>:<labeled array job>.
#SBATCH --job-name=reserved-top300-finalize
#SBATCH --partition=normal
#SBATCH --nodelist=rtx3090b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/reserved-top300-finalize-%j.log
set -euo pipefail
cd /home/wangpch/TG-Interpolation
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TOKENIZERS_PARALLELISM=false
/home/wangpch/.conda/envs/LLM/bin/python datatools/parse_test_docppl_data/finalize_reserved_top300.py
