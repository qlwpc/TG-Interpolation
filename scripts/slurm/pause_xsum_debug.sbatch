#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:1
set -euo pipefail

: "${PAUSE_DEBUG_CHECKPOINT:?PAUSE_DEBUG_CHECKPOINT is required}"
: "${PAUSE_DEBUG_OUTPUT:?PAUSE_DEBUG_OUTPUT is required}"
PAUSE_WORKSPACE=${PAUSE_WORKSPACE:-/home/wangpch/TG-Interpolation}
PAUSE_RUNTIME_BIN=${PAUSE_RUNTIME_BIN:-/home/wangpch/.conda/envs/LLM/bin}
PAUSE_DEBUG_NUM_EXAMPLES=${PAUSE_DEBUG_NUM_EXAMPLES:-8}
PAUSE_DEBUG_MAX_REAL_TOKENS=${PAUSE_DEBUG_MAX_REAL_TOKENS:-64}
PAUSE_DEBUG_BEAM_SIZE=${PAUSE_DEBUG_BEAM_SIZE:-2}

export PATH=${PAUSE_RUNTIME_BIN}:${PATH}
export PYTHONPATH=${PAUSE_WORKSPACE}${PYTHONPATH:+:${PYTHONPATH}}
export WANDB_MODE=disabled
export MPLCONFIGDIR=${PAUSE_DEBUG_OUTPUT}/mpl
mkdir -p "${PAUSE_DEBUG_OUTPUT}" "${MPLCONFIGDIR}"
cd "${PAUSE_WORKSPACE}"

nvidia-smi
exec "${PAUSE_RUNTIME_BIN}/python" \
  "${PAUSE_WORKSPACE}/scripts/diagnose_pause_xsum_generation.py" \
  --checkpoint "${PAUSE_DEBUG_CHECKPOINT}" \
  --output-dir "${PAUSE_DEBUG_OUTPUT}" \
  --dataset-dir "${PAUSE_WORKSPACE}/dataset/Xsum" \
  --vocab "${PAUSE_WORKSPACE}/dataset/bbc-news/TG_GPT2_tokenizer.json" \
  --num-examples "${PAUSE_DEBUG_NUM_EXAMPLES}" \
  --max-real-tokens "${PAUSE_DEBUG_MAX_REAL_TOKENS}" \
  --beam-size "${PAUSE_DEBUG_BEAM_SIZE}"
