#!/usr/bin/env bash
# Run one model's native document-PPL evaluation on complete document shards.
set -euo pipefail

model=${1:?missing model argument (gpst or pushdown)}
workers=${2:-8}
native_data=${3:-dataset/bbc-news/native_model_topk_300_v2}
output_dir=${4:-docppl_runs/${model}_native_full}
python_bin=${PYTHON_BIN:-python}
documents=4966
document_result_dir=${DOCUMENT_RESULT_DIR:-$output_dir/documents}
aggregate_root=${AGGREGATE_ROOT:-$output_dir}
result_prefix=${RESULT_PREFIX:-shard}
max_batch_attention_elements=${PUSHDOWN_MAX_BATCH_ATTENTION_ELEMENTS:-16777216}
mkdir -p "$output_dir"
mkdir -p "$document_result_dir"
pids=()

if [[ -n "${DOCUMENT_BOUNDS:-}" ]]; then
  IFS=',' read -r -a document_bounds <<<"$DOCUMENT_BOUNDS"
  if [[ "${#document_bounds[@]}" -ne $((workers + 1)) ]]; then
    echo "DOCUMENT_BOUNDS must contain workers+1 comma-separated boundaries" >&2
    exit 2
  fi
fi

for ((worker=0; worker<workers; worker++)); do
  if [[ -n "${DOCUMENT_BOUNDS:-}" ]]; then
    start=${document_bounds[$worker]}
    end=${document_bounds[$((worker + 1))]}
  else
    start=$((documents * worker / workers))
    end=$((documents * (worker + 1) / workers))
  fi
  gpu=$worker
  result="$output_dir/${result_prefix}_${worker}_of_${workers}.json"
  log="$output_dir/${result_prefix}_${worker}_of_${workers}.log"
  if [[ "$model" == gpst ]]; then
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. \
      "$python_bin" scripts/gpst/evaluate_document_ppl.py \
      --checkpoint saved_models/gpst-bbc-unsup/model.bin --native-data "$native_data" \
      --start-document "$start" --end-document "$end" --eval-batch-size 64 \
      --max-batch-actions 65536 --max-batch-attention-elements 16777216 --log-every 100 \
      --document-result-dir "$document_result_dir" --resume-document-results \
      >"$result" 2>"$log" &
    pids+=("$!")
  elif [[ "$model" == pushdown ]]; then
    OLMO_FLEX_ATTENTION_NUM_STAGES=1 CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. \
      "$python_bin" scripts/evaluate_pushdown_document_ppl.py \
      --checkpoint saved_models/pushdown/step34354-unsharded --native-data "$native_data" \
      --start-document "$start" --end-document "$end" --eval-batch-size 64 \
      --max-batch-tokens 65536 --max-batch-attention-elements "$max_batch_attention_elements" --log-every 100 \
      --document-result-dir "$document_result_dir" --resume-document-results \
      >"$result" 2>"$log" &
    pids+=("$!")
  else
    echo "model must be gpst or pushdown" >&2; exit 2
  fi
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "at least one document-PPL worker failed; aggregate was not written" >&2
  exit "$status"
fi
"$python_bin" scripts/merge_native_document_ppl.py "$model" "$aggregate_root" \
  --expected-documents "$documents" --expected-samples-per-sentence 300 \
  --output "$output_dir/aggregate.json"
