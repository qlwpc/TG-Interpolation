#!/bin/bash
#SBATCH --partition=critical
#SBATCH --account=tukw-critical
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
set -euo pipefail
cd /inspurfs/group/tukw/wangpch/projected_docppl_20261007/repo
export PATH=/opt/gridview/slurm/bin:/public/home/wangpch/anaconda3/envs/LLM/bin:$PATH
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export TORCHINDUCTOR_CACHE_DIR=/tmp/projected_inductor_${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/projected_triton_${SLURM_JOB_ID}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
/public/home/wangpch/anaconda3/envs/LLM/bin/python scripts/campaigns/projected_docppl_20261007.py "$@" --campaign /inspurfs/group/tukw/wangpch/projected_docppl_20261007
