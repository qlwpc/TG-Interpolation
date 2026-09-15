#!/bin/bash
#SBATCH --job-name=tg-defaults-long
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --account=wangpch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=validation/tg_kernel_defaults_20260915/long_%j.out
#SBATCH --error=validation/tg_kernel_defaults_20260915/long_%j.err
set -euo pipefail
cd /home/wangpch/TG-Interpolation
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/tg-defaults-triton
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
/home/wangpch/.conda/envs/LLM/bin/python -c 'import torch, triton, pybind11; print(torch.__version__, triton.__version__, pybind11.__version__); assert torch.cuda.is_available()'
/home/wangpch/.conda/envs/LLM/bin/python -m pytest tests/test_tg_defaults.py -k long_nomask -x -v --junitxml="validation/tg_kernel_defaults_20260915/long_${SLURM_JOB_ID}.xml"
