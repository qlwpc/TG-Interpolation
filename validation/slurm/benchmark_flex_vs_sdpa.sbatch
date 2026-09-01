#!/bin/bash
#SBATCH --job-name=flex-vs-sdpa
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=validation/slurm/flex-vs-sdpa_%j.out
#SBATCH --error=validation/slurm/flex-vs-sdpa_%j.err

set -uo pipefail

REPO=/home/wangpch/TG-Interpolation
FLEX_ENV_NAME=${FLEX_ENV_NAME:-LLM}
FLEX_LABEL=${FLEX_LABEL:-causal-n128}
FLEX_SEQUENCE_LENGTH=${FLEX_SEQUENCE_LENGTH:-128}
FLEX_MODE=${FLEX_MODE:-causal}
FLEX_BATCH_SIZE=${FLEX_BATCH_SIZE:-4}
FLEX_HEADS=${FLEX_HEADS:-12}
FLEX_HEAD_DIM=${FLEX_HEAD_DIM:-64}
FLEX_LAYERS=${FLEX_LAYERS:-12}
FLEX_WORKLOADS=${FLEX_WORKLOADS:-"eval train"}

cd "$REPO" || exit 2
source /data/software/anaconda3/etc/profile.d/conda.sh
conda activate "$FLEX_ENV_NAME"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/flex-vs-sdpa-xdg-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1

echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname)"
echo "environment=${FLEX_ENV_NAME}"
echo "sequence_length=${FLEX_SEQUENCE_LENGTH}"
echo "mode=${FLEX_MODE}"
echo "batch_size=${FLEX_BATCH_SIZE}"
echo "heads=${FLEX_HEADS}"
echo "head_dim=${FLEX_HEAD_DIM}"
echo "layers=${FLEX_LAYERS}"
echo "workloads=${FLEX_WORKLOADS}"
echo "python=$(command -v python)"

overall_status=0
for workload in $FLEX_WORKLOADS; do
    case "$workload" in
        eval|train) ;;
        *) echo "invalid workload: $workload" >&2; exit 2 ;;
    esac
    export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/flex-vs-sdpa-inductor-${SLURM_JOB_ID}-${workload}"
    export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/flex-vs-sdpa-triton-${SLURM_JOB_ID}-${workload}"
    python validation/slurm/benchmark_flex_vs_sdpa.py \
        --label "$FLEX_LABEL" \
        --sequence-length "$FLEX_SEQUENCE_LENGTH" \
        --mode "$FLEX_MODE" \
        --workload "$workload" \
        --batch-size "$FLEX_BATCH_SIZE" \
        --heads "$FLEX_HEADS" \
        --head-dim "$FLEX_HEAD_DIM" \
        --layers "$FLEX_LAYERS" || overall_status=1
done
exit "$overall_status"
