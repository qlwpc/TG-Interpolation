#!/usr/bin/env bash
#SBATCH --job-name=merge-nary-rbin-ppl
#SBATCH --partition=normal
#SBATCH --nodelist=rtx3090b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/merge-nary-rbin-ppl-%j.log
#SBATCH --error=logs/merge-nary-rbin-ppl-%j.log

set -euo pipefail

workspace=/home/wangpch/TG-Interpolation
output_root=${OUTPUT_ROOT:-${workspace}/artifacts/experiment/nary_right_binary_pushdown_document_ppl_20260830/full}
expected_candidates=${EXPECTED_VALID_CANDIDATE_COUNT:?EXPECTED_VALID_CANDIDATE_COUNT is required}
expected_attachment_normalization=${EXPECTED_ATTACHMENT_NORMALIZATION:-stack_legal}
expected_structure_source=${EXPECTED_STRUCTURE_SOURCE:-v2_pushdown_nary_topk_word_atom_right_binarized}
cd "${workspace}"
shopt -s nullglob
shards=("${output_root}"/shard_*_documents_*.json)
if (( ${#shards[@]} != 8 )); then
  echo "expected 8 shard JSON files, found ${#shards[@]}" >&2
  exit 2
fi

/home/wangpch/.conda/envs/LLM/bin/python \
  scripts/merge_gpst_binary_pushdown_document_ppl.py \
  "${shards[@]}" \
  --require-full-bbc \
  --expected-structure-source "${expected_structure_source}" \
  --expected-attachment-normalization "${expected_attachment_normalization}" \
  --expected-valid-candidate-count "${expected_candidates}" \
  --output "${output_root}/aggregate.json"
