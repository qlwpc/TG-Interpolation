#!/usr/bin/env bash
# Repeat one document in one process on several local RTX 3090 GPUs.
# Usage: sbatch --array=0-3 scripts/slurm/stress_gpst_binary_pushdown_document_ppl_local.sbatch
#SBATCH --job-name=gpstbin-nan-stress
#SBATCH --partition=normal
#SBATCH --nodelist=rtx3090b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --time=02:00:00
#SBATCH --output=logs/gpstbin-nan-stress-%A_%a.log
#SBATCH --error=logs/gpstbin-nan-stress-%A_%a.log

set -euo pipefail

workspace=/home/wangpch/TG-Interpolation
python_bin=/home/wangpch/.conda/envs/LLM/bin/python
checkpoint=${CHECKPOINT:-${workspace}/saved_models/pushdown_terminalonly/step34354-unsharded}
native_data=${NATIVE_DATA:-${workspace}/dataset/bbc-news/testppl/native_model_topk_300_v2}
tokenizer=${TOKENIZER_PATH:-${workspace}/dataset/bbc-news/TG_GPT2_tokenizer.json}
document=${DOCUMENT:-2505}
repeats=${REPEATS:-50}
task_id=${SLURM_ARRAY_TASK_ID:-0}
output_root=${OUTPUT_ROOT:-${workspace}/artifacts/analysis/gpst_binary_pushdown_nan_20260830/stress}
output=${output_root}/task_${task_id}_document_${document}.json

cd "${workspace}"
mkdir -p logs "${output_root}"
export PYTHONPATH="${workspace}${PYTHONPATH:+:${PYTHONPATH}}"
export OLMO_FLEX_ATTENTION_NUM_STAGES=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export MPLCONFIGDIR="${output_root}/matplotlib_task${task_id}"
mkdir -p "${MPLCONFIGDIR}"

date -Is
hostname
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-default}"
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.free --format=csv,noheader
extra_args=()
if [[ "${DISABLE_KV_CACHE:-0}" == "1" ]]; then
  extra_args+=(--disable-kv-cache)
fi
if [[ "${DISABLE_PUSHDOWN_FLEX:-0}" == "1" ]]; then
  extra_args+=(--disable-pushdown-flex)
fi
"${python_bin}" scripts/stress_gpst_binary_pushdown_document_ppl.py \
  --checkpoint "${checkpoint}" \
  --native-data "${native_data}" \
  --tokenizer-path "${tokenizer}" \
  --document "${document}" \
  --repeats "${repeats}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-64}" \
  --output "${output}" \
  "${extra_args[@]}"
date -Is
