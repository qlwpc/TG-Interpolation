#!/usr/bin/env bash
# Run one model's native document-PPL evaluation on complete document shards.
set -euo pipefail

model=${1:?missing model argument (gpst or pushdown)}
workers=${2:-8}
native_data=${3:-dataset/bbc-news/native_model_topk_300_v2}
output_dir=${4:-docppl_runs/${model}_native_full}
python_bin=${PYTHON_BIN:-python}
documents=4966
mkdir -p "$output_dir"

for ((worker=0; worker<workers; worker++)); do
  start=$((documents * worker / workers))
  end=$((documents * (worker + 1) / workers))
  gpu=$worker
  result="$output_dir/shard_${worker}_of_${workers}.json"
  log="$output_dir/shard_${worker}_of_${workers}.log"
  if [[ "$model" == gpst ]]; then
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. \
      "$python_bin" scripts/gpst/evaluate_document_ppl.py \
      --checkpoint saved_models/gpst-bbc-unsup/model.bin --native-data "$native_data" \
      --start-document "$start" --end-document "$end" --eval-batch-size 64 \
      --max-batch-actions 65536 --log-every 100 \
      >"$result" 2>"$log" &
  elif [[ "$model" == pushdown ]]; then
    OLMO_FLEX_ATTENTION_NUM_STAGES=1 CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. \
      "$python_bin" scripts/evaluate_pushdown_document_ppl.py \
      --checkpoint saved_models/pushdown/step34354-unsharded --native-data "$native_data" \
      --start-document "$start" --end-document "$end" --eval-batch-size 64 \
      --max-batch-tokens 65536 --log-every 100 \
      >"$result" 2>"$log" &
  else
    echo "model must be gpst or pushdown" >&2; exit 2
  fi
done
wait
"$python_bin" scripts/merge_native_document_ppl.py "$model" "$output_dir" > "$output_dir/aggregate.json"
