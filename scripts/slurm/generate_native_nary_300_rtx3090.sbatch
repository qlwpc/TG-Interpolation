#!/usr/bin/env bash
#SBATCH --job-name=native-nary300
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-27%7
#SBATCH --time=24:00:00
#SBATCH --output=logs/native-nary300-%A_%a.log

set -euo pipefail

workspace=/home/wangpch/TG-Interpolation
python_bin=/home/wangpch/.conda/envs/LLM/bin/python
output_dir=${workspace}/dataset/bbc-news/native_nary_300

cd "${workspace}"
mkdir -p logs "${output_dir}"
export PYTHONPATH="${workspace}:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"${python_bin}" scripts/generate_native_nary_test.py generate \
  --output "${output_dir}" \
  --shard-id "${SLURM_ARRAY_TASK_ID}" \
  --num-shards 28 \
  --device cuda \
  --cpu-workers 8 \
  --parser-batch-size 32 \
  --parser-batch-tokens 2000 \
  --max-pending 32 \
  --flush-every 100
