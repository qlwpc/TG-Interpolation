#!/usr/bin/env bash
# Binary candidate-support evaluations for the terminal-only Pushdown checkpoint.
#SBATCH --job-name=nary-rbin-pushdown-ppl
#SBATCH --partition=normal
#SBATCH --nodelist=rtx3090b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --time=120:00:00
#SBATCH --output=logs/nary-rbin-pushdown-ppl-%A_%a.log
#SBATCH --error=logs/nary-rbin-pushdown-ppl-%A_%a.log

set -euo pipefail

workspace=/home/wangpch/TG-Interpolation
python_bin=/home/wangpch/.conda/envs/LLM/bin/python
checkpoint=${CHECKPOINT:-${workspace}/saved_models/pushdown_terminalonly/step34354-unsharded}
native_data=${NATIVE_DATA:-${workspace}/dataset/bbc-news/testppl/native_model_topk_300_v2}
tokenizer=${TOKENIZER_PATH:-${workspace}/dataset/bbc-news/TG_GPT2_tokenizer.json}
run_tag=${RUN_TAG:-full}
attachment_normalization=${ATTACHMENT_NORMALIZATION:-stack_legal}
candidate_source=${CANDIDATE_SOURCE:-pushdown-nary-word-atom-right-binary}
boundaries=(0 620 1241 1862 2483 3103 3724 4345 4966)
task_id=${SLURM_ARRAY_TASK_ID:-0}
start_document=${START_DOCUMENT:-${boundaries[$task_id]}}
end_document=${END_DOCUMENT:-${boundaries[$((task_id + 1))]}}
output_root=${OUTPUT_ROOT:-${workspace}/artifacts/experiment/nary_right_binary_pushdown_document_ppl_20260830/${run_tag}}
result=${output_root}/shard_${task_id}_documents_${start_document}_${end_document}.json
document_output=${output_root}/documents_${start_document}_${end_document}

if (( start_document < 0 || start_document >= end_document || end_document > 4966 )); then
  echo "invalid document interval [${start_document},${end_document})" >&2
  exit 2
fi

cd "${workspace}"
mkdir -p logs "${output_root}" "${document_output}"
export PYTHONPATH="${workspace}${PYTHONPATH:+:${PYTHONPATH}}"
export OLMO_FLEX_ATTENTION_NUM_STAGES=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export MPLCONFIGDIR="${output_root}/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

extra_args=()
if [[ -n "${MAX_SENTENCES:-}" ]]; then
  extra_args+=(--max-sentences "${MAX_SENTENCES}")
fi

date -Is
hostname
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.free --format=csv,noheader
"${python_bin}" scripts/evaluate_gpst_binary_pushdown_document_ppl.py \
  --checkpoint "${checkpoint}" \
  --native-data "${native_data}" \
  --tokenizer-path "${tokenizer}" \
  --candidate-source "${candidate_source}" \
  --attachment-normalization "${attachment_normalization}" \
  --start-document "${start_document}" \
  --end-document "${end_document}" \
  --max-sequence-length 2048 \
  --eval-batch-size "${EVAL_BATCH_SIZE:-64}" \
  --max-batch-tokens "${MAX_BATCH_TOKENS:-65536}" \
  --max-batch-attention-elements "${MAX_BATCH_ATTENTION_ELEMENTS:-16777216}" \
  --prefetch-sentences "${PREFETCH_SENTENCES:-2}" \
  --skip-merge-order-validation \
  --device cuda \
  --log-every "${LOG_EVERY:-10}" \
  --document-output-dir "${document_output}" \
  --output "${result}" \
  "${extra_args[@]}"
date -Is
