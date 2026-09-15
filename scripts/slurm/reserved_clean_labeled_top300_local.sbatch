#!/usr/bin/env bash
#SBATCH --job-name=reserved-tree300
#SBATCH --partition=normal
#SBATCH --nodelist=rtx3090b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%2
#SBATCH --time=120:00:00
#SBATCH --output=logs/reserved-tree300-%A_%a.log
set -euo pipefail
cd /home/wangpch/TG-Interpolation
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TOKENIZERS_PARALLELISM=false
extra=()
if [[ -n "${MAX_SENTENCES:-}" ]]; then extra+=(--max-sentences "$MAX_SENTENCES"); fi
if [[ -n "${OUTPUT_ROOT:-}" ]]; then extra+=(--output "$OUTPUT_ROOT"); fi
/home/wangpch/.conda/envs/LLM/bin/python datatools/parse_test_docppl_data/generate_labeled_topk.py generate \
  --num-shards 8 --shard-id "$SLURM_ARRAY_TASK_ID" --device cuda \
  --cpu-workers 12 --parser-batch-size 16 --parser-batch-tokens 1500 "${extra[@]}"
