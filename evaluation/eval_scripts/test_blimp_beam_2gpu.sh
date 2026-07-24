#!/bin/bash
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem-per-cpu=1M
#SBATCH --gres=gpu:2
#SBATCH -t 00:40:00
#SBATCH --job-name=blimp_beam_2gpu
#SBATCH --output=analysis-output/blimp_beam/logs/2gpu_%j.log
set -euo pipefail

# 2-GPU crash regression for the unified sync_on_compute fix.
# Runs beam-search BLiMP with 2 ranks and a subset_num_batches NOT divisible by
# 2 (5 sentences -> rank counts 3/2). Before the Part A fix this deadlocked in
# compute() (the user-reported crash). After the fix it completes with a finite
# overall/overall accuracy.
#
# Usage:  sbatch evaluation/eval_scripts/test_blimp_beam_2gpu.sh

source /data/software/anaconda3/etc/profile.d/conda.sh
conda activate LLM
export PYTHONPATH=/home/wangpch/TG-Interpolation:${PYTHONPATH:-}
export HF_ENDPOINT=https://hf-mirror.com
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/wangpch/TG-Interpolation
mkdir -p analysis-output/blimp_beam/2gpu analysis-output/blimp_beam/logs

# Dedicated smoke config with subset_num_batches=5 (odd -> 3/2 split across 2 ranks).
CFG=train_configs/eval_per_metric/treereg_BLiMP_beam_2gpu.yaml
CKPT=/home/wangpch/TG-Interpolation/saved_models/treereg/step33862-unsharded

echo "=== [$(date)] BLiMP beam 2-GPU unequal-count regression (subset=5) ==="
echo "CFG=$CFG  CKPT=$CKPT"
python -c "import torch; print('devices', torch.cuda.device_count())"

torchrun --nproc-per-node=2 --master_port=17402 \
  scripts/train.py ${CFG} \
    --run_name=blimp_beam_2gpu \
    --workspace=/home/wangpch/TG-Interpolation \
    --save_folder=analysis-output/blimp_beam/2gpu \
    --save_overwrite \
    --load_path=${CKPT}

echo "=== [$(date)] 2-GPU regression done (if you see this, no deadlock) ==="
