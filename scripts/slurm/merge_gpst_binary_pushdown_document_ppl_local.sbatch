#!/usr/bin/env bash
# Finalize the eight local full-document shards after their Slurm array exits.
# Submit with: sbatch --dependency=afterany:<array_job_id> <this-file>
# ``afterany`` deliberately runs the checker even if a worker failed; missing
# or incomplete shards then produce a visible failed finalizer instead of a
# dependency that remains pending forever.
#SBATCH --job-name=merge-gpstbin-ppl
#SBATCH --partition=normal
#SBATCH --nodelist=rtx3090b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/merge-gpstbin-ppl-%j.log
#SBATCH --error=logs/merge-gpstbin-ppl-%j.log

set -euo pipefail

workspace=/home/wangpch/TG-Interpolation
python_bin=/home/wangpch/.conda/envs/LLM/bin/python
output_root=${OUTPUT_ROOT:-${workspace}/artifacts/experiment/gpst_binary_pushdown_document_ppl_20260830/full}

cd "${workspace}"
shards=("${output_root}"/shard_[0-7]_documents_*.json)
if (( ${#shards[@]} != 8 )); then
  echo "expected 8 completed shard JSON files, found ${#shards[@]}" >&2
  exit 1
fi

"${python_bin}" scripts/merge_gpst_binary_pushdown_document_ppl.py \
  "${shards[@]}" \
  --require-full-bbc \
  --output "${output_root}/aggregate.json"
sha256sum "${output_root}/aggregate.json" >"${output_root}/aggregate.sha256"
