#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
set -euo pipefail
cd /inspurfs/group/tukw/wangpch/docppl_reserved_20260907/repo
export PATH=/opt/gridview/slurm/bin:$PATH
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export TORCHINDUCTOR_CACHE_DIR=/tmp/docppl_inductor_${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/docppl_triton_${SLURM_JOB_ID}
export OLMO_FLEX_ATTENTION_NUM_STAGES=1
PYTHON=${RC_PYTHON:-/public/home/wangpch/venvs/LLM-sm120/bin/python}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PYTHON" -c 'import torch; from olmo.data import SentencepieceVocab; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_capability())'
"$PYTHON" "$@"
