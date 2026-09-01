#!/bin/bash
#SBATCH --job-name=tg-llm-env
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1M
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=validation/slurm/tg-llm-env_%j.out
#SBATCH --error=validation/slurm/tg-llm-env_%j.err

set -uo pipefail

REPO=/home/wangpch/TG-Interpolation
cd "$REPO" || exit 2

source /data/software/anaconda3/etc/profile.d/conda.sh
conda activate LLM

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/tg-llm-xdg-${SLURM_JOB_ID}"
export WANDB_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/tg-llm-wandb-cache-${SLURM_JOB_ID}"
export WANDB_DATA_DIR="${SLURM_TMPDIR:-/tmp}/tg-llm-wandb-data-${SLURM_JOB_ID}"
export WANDB_DIR="${SLURM_TMPDIR:-/tmp}"
export WANDB_MODE=offline

echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "conda_prefix=${CONDA_PREFIX:-unset}"
echo "python=$(command -v python)"

status=0
nvidia-smi || status=1
python -m pip check || status=1
conda doctor -n LLM || status=1
python validation/slurm/verify_llm_gpu.py || status=1
python -m pytest -q \
    tests/test_gpst_cppbackend.py \
    tests/test_parse_pipeline_smoke.py || status=1

exit "$status"
