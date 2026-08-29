#!/usr/bin/env bash
# Submit after rtx3090 has been restored to schedulable Slurm state.
#SBATCH --job-name=pushdown-native-docppl
#SBATCH --partition=normal
#SBATCH --nodelist=rtx3090
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=2
#SBATCH --mem=2M
#SBATCH --time=120:00:00
#SBATCH --exclusive
#SBATCH --output=docppl_runs/pushdown_native_sbatch_%j.driver.log
#SBATCH --error=docppl_runs/pushdown_native_sbatch_%j.driver.log

set -euo pipefail
cd /home/wangpch/TG-Interpolation
mkdir -p docppl_runs

# The normal QOS caps this 8-GPU job at two CPU cores. Keep BLAS single-
# threaded; the eight lightweight mmap workers then time-share only packing.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=.
export PYTHON_BIN=/home/wangpch/.conda/envs/LLM/bin/python

"$PYTHON_BIN" --version
bash scripts/run_native_document_ppl_shards.sh pushdown 8 \
  dataset/bbc-news/native_model_topk_300_v2 \
  "docppl_runs/pushdown_native_sbatch_${SLURM_JOB_ID}"
