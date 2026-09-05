#!/usr/bin/env bash
# Four-GPU native Pushdown document-PPL run.  This deliberately co-schedules
# with the four remaining one-GPU GPST continuation shards on the 8-GPU host.
#SBATCH --job-name=pushdown-native-docppl-4g
#SBATCH --partition=normal
#SBATCH --nodelist=rtx3090
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --output=docppl_runs/pushdown_native_4gpu_%j.driver.log
#SBATCH --error=docppl_runs/pushdown_native_4gpu_%j.driver.log

set -euo pipefail
cd /home/wangpch/TG-Interpolation
mkdir -p docppl_runs

# The RTX3090 shared-memory limit needs the one-stage FlexAttention kernel.
export OLMO_FLEX_ATTENTION_NUM_STAGES=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH=.
export PYTHON_BIN=/home/wangpch/.conda/envs/LLM/bin/python

"$PYTHON_BIN" --version
bash scripts/run_native_document_ppl_shards.sh pushdown 4 \
  "${NATIVE_DATA:-dataset/bbc-news/testppl/native_model_topk_300_v2}" \
  "${RESUME_OUTPUT_DIR:-docppl_runs/pushdown_native_4gpu_${SLURM_JOB_ID}}"
