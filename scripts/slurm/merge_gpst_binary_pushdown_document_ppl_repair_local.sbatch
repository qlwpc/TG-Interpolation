#!/usr/bin/env bash
# Merge the seven finite original shards with four strict shard-4 repair slices.
# Submit after all four repair jobs with an afterany dependency so missing/failed
# slices are reported by this checker rather than leaving a pending dependency.
#SBATCH --job-name=merge-gpstbin-repair
#SBATCH --partition=normal
#SBATCH --nodelist=rtx3090b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/merge-gpstbin-repair-%j.log
#SBATCH --error=logs/merge-gpstbin-repair-%j.log

set -euo pipefail

workspace=/home/wangpch/TG-Interpolation
python_bin=/home/wangpch/.conda/envs/LLM/bin/python
original_root=${ORIGINAL_ROOT:-${workspace}/artifacts/experiment/gpst_binary_pushdown_document_ppl_20260830/full}
repair_root=${REPAIR_ROOT:-${workspace}/artifacts/experiment/gpst_binary_pushdown_document_ppl_20260830/nan_diagnosis/shard4_split_finite_guard}
output_root=${OUTPUT_ROOT:-${workspace}/artifacts/experiment/gpst_binary_pushdown_document_ppl_20260830/nan_diagnosis/repaired_full_split}
repair_shard_count=${REPAIR_SHARD_COUNT:-4}

cd "${workspace}"
mkdir -p "${output_root}"
original_shards=(
  "${original_root}"/shard_0_documents_*.json
  "${original_root}"/shard_1_documents_*.json
  "${original_root}"/shard_2_documents_*.json
  "${original_root}"/shard_3_documents_*.json
  "${original_root}"/shard_5_documents_*.json
  "${original_root}"/shard_6_documents_*.json
  "${original_root}"/shard_7_documents_*.json
)
repair_shards=("${repair_root}"/shard_*_documents_*.json)
if (( ${#original_shards[@]} != 7 )); then
  echo "expected 7 finite original shard JSON files, found ${#original_shards[@]}" >&2
  exit 1
fi
if (( ${#repair_shards[@]} != repair_shard_count )); then
  echo "expected ${repair_shard_count} shard-4 repair JSON files, found ${#repair_shards[@]}" >&2
  exit 1
fi

"${python_bin}" scripts/merge_gpst_binary_pushdown_document_ppl.py \
  "${original_shards[@]}" \
  "${repair_shards[@]}" \
  --require-full-bbc \
  --output "${output_root}/aggregate.json"
sha256sum "${output_root}/aggregate.json" >"${output_root}/aggregate.sha256"
