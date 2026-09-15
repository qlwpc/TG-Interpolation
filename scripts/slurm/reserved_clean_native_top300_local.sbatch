#!/usr/bin/env bash
#SBATCH --job-name=reserved-native300
#SBATCH --partition=normal
#SBATCH --nodelist=rtx3090b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%2
#SBATCH --time=120:00:00
#SBATCH --output=logs/reserved-native300-%A_%a.log
set -euo pipefail
cd /home/wangpch/TG-Interpolation
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TOKENIZERS_PARALLELISM=false
/home/wangpch/.conda/envs/LLM/bin/python datatools/parse_test_docppl_data/generate_native_topk.py generate \
  --ppl-dir dataset/bbc-news-reserved-clean-v1/canonical/test \
  --test-tree dataset/bbc-news-reserved-clean-v1/tree/test.npy \
  --output dataset/bbc-news-reserved-clean-v1/native_model_topk_300_v2/test \
  --num-shards 8 --shard-id "$SLURM_ARRAY_TASK_ID" --component both --device cuda \
  --cpu-workers 24 --parser-batch-size 16 --parser-batch-tokens 1500 \
  --max-pending 32 --flush-every 250
