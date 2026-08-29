#!/usr/bin/env bash
# Submit after rtx3090 has been restored to schedulable Slurm state.
#SBATCH --job-name=pushdown-native-docppl
#SBATCH --partition=normal
#SBATCH --nodelist=rtx3090
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --output=docppl_runs/pushdown_native_sbatch_%j.driver.log
#SBATCH --error=docppl_runs/pushdown_native_sbatch_%j.driver.log

set -euo pipefail
cd /home/wangpch/TG-Interpolation
mkdir -p docppl_runs

# Eight GPU workers share 32 allocated CPU cores.  Avoid unbounded BLAS/OpenMP
# thread pools competing with the Python candidate/attachment preparation.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=.
export PYTHON_BIN=/home/wangpch/.conda/envs/LLM/bin/python

"$PYTHON_BIN" --version
bash scripts/run_native_document_ppl_shards.sh pushdown 8 \
  dataset/bbc-news/native_model_topk_300_v2 \
  "docppl_runs/pushdown_native_sbatch_${SLURM_JOB_ID}"
