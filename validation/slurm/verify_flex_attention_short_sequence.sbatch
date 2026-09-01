#!/bin/bash
#SBATCH --job-name=flex-short-bwd
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1M
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=validation/slurm/flex-attention-short-sequence_%j.out
#SBATCH --error=validation/slurm/flex-attention-short-sequence_%j.err

set -uo pipefail

REPO=/home/wangpch/TG-Interpolation
FLEX_ENV_NAME=${FLEX_ENV_NAME:-LLM}
FLEX_LABEL=${FLEX_LABEL:-baseline-n127-h8}
FLEX_SEQUENCE_LENGTH=${FLEX_SEQUENCE_LENGTH:-127}
FLEX_BLOCK_MASK_HEADS=${FLEX_BLOCK_MASK_HEADS:-8}
FLEX_BATCH_SIZE=${FLEX_BATCH_SIZE:-64}
FLEX_QUERY_HEADS=${FLEX_QUERY_HEADS:-8}
FLEX_HEAD_DIM=${FLEX_HEAD_DIM:-32}
FLEX_DTYPE=${FLEX_DTYPE:-bfloat16}
FLEX_MASK_MODE=${FLEX_MASK_MODE:-causal}

cd "$REPO" || exit 2
source /data/software/anaconda3/etc/profile.d/conda.sh
conda activate "$FLEX_ENV_NAME"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/flex-short-xdg-${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/flex-short-inductor-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/flex-short-triton-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1

echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname)"
echo "environment=${FLEX_ENV_NAME}"
echo "sequence_length=${FLEX_SEQUENCE_LENGTH}"
echo "block_mask_heads=${FLEX_BLOCK_MASK_HEADS}"
echo "python=$(command -v python)"

args=(
    --label "$FLEX_LABEL"
    --sequence-length "$FLEX_SEQUENCE_LENGTH"
    --block-mask-heads "$FLEX_BLOCK_MASK_HEADS"
    --batch-size "$FLEX_BATCH_SIZE"
    --query-heads "$FLEX_QUERY_HEADS"
    --head-dim "$FLEX_HEAD_DIM"
    --dtype "$FLEX_DTYPE"
    --mask-mode "$FLEX_MASK_MODE"
)
if [[ -n "${FLEX_KEY_VALUE_HEADS:-}" ]]; then
    args+=(--key-value-heads "$FLEX_KEY_VALUE_HEADS")
fi
if [[ -n "${FLEX_KEY_VALUE_LENGTH:-}" ]]; then
    args+=(--key-value-length "$FLEX_KEY_VALUE_LENGTH")
fi
if [[ -n "${FLEX_NUM_STAGES:-}" ]]; then
    args+=(--num-stages "$FLEX_NUM_STAGES")
fi
if [[ "${FLEX_FORWARD_ONLY:-0}" == "1" ]]; then
    args+=(--forward-only)
fi
python validation/slurm/verify_flex_attention_short_sequence.py "${args[@]}"
