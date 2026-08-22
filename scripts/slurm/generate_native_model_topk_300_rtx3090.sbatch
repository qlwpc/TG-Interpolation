#!/usr/bin/env bash
#SBATCH --job-name=native-topk-v2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/native-topk-v2-%j.log

set -euo pipefail

mode=${1:?usage: sbatch ... generate_native_model_topk_300_rtx3090.sbatch MODE SHARD_ID}
shard_id=${2:?usage: sbatch ... generate_native_model_topk_300_rtx3090.sbatch MODE SHARD_ID}
workspace=/home/wangpch/TG-Interpolation
python_bin=/home/wangpch/.conda/envs/LLM/bin/python
ppl_dir=${workspace}/dataset/bbc-news/testppl_tree_updated_20260822
test_tree=${workspace}/dataset/bbc-news/tree/test.updated_20260822.npy
output_dir=${workspace}/dataset/bbc-news/native_model_topk_300_v2
old_output=${workspace}/dataset/bbc-news/native_nary_300

cd "${workspace}"
mkdir -p logs "${output_dir}"
export PYTHONPATH="${workspace}:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

common=(
  --test-tree "${test_tree}"
  --ppl-dir "${ppl_dir}"
  --output "${output_dir}"
  --shard-id "${shard_id}"
  --num-shards 28
)

case "${mode}" in
  reuse-pushdown)
    "${python_bin}" scripts/generate_native_nary_test.py reuse-pushdown \
      "${common[@]}" --reuse-pushdown-from "${old_output}"
    ;;
  gpst)
    "${python_bin}" scripts/generate_native_nary_test.py generate \
      "${common[@]}" --component gpst --device cuda --cpu-workers 8 \
      --parser-batch-size 32 --parser-batch-tokens 2000 \
      --max-pending 32 --flush-every 100
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    exit 2
    ;;
esac
