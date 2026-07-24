#!/bin/bash
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem-per-cpu=1M
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --job-name=blimp_beam_full
#SBATCH --output=analysis-output/blimp_beam/logs/full_%j.log
set -euo pipefail

# Full beam-search BLiMP eval (all 67 tasks x 1000 pairs = 134k sentences) on
# 1 GPU. This is slow (one word_sync_beam_search per sentence, beam_size=300);
# the 24h wall-time is a generous ceiling. Reduce subset_num_batches in the YAML
# for a quicker partial run.
#
# Usage:  sbatch evaluation/eval_scripts/eval_blimp_beam_treereg.sh
# For pushdown: change CFG to pushdown_BLiMP_beam.yaml and CKPT to saved_models/pushdown/...

source /data/software/anaconda3/etc/profile.d/conda.sh
conda activate LLM
export PYTHONPATH=/home/wangpch/TG-Interpolation:${PYTHONPATH:-}
export HF_ENDPOINT=https://hf-mirror.com
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/wangpch/TG-Interpolation
mkdir -p analysis-output/treereg_eval_BLiMP_beam analysis-output/blimp_beam/logs

CFG=train_configs/eval_per_metric/treereg_BLiMP_beam.yaml
CKPT=/home/wangpch/TG-Interpolation/saved_models/treereg/step33862-unsharded

echo "=== [$(date)] Full BLiMP beam eval (treereg, 1 GPU, subset_num_batches=-1) ==="
echo "CFG=$CFG  CKPT=$CKPT"
python -c "import torch; print('devices', torch.cuda.device_count())"

torchrun --nproc-per-node=1 --master_port=17403 \
  scripts/train.py ${CFG} \
    --run_name=treereg_eval_BLiMP_beam \
    --workspace=/home/wangpch/TG-Interpolation \
    --save_folder=analysis-output/treereg_eval_BLiMP_beam \
    --save_overwrite \
    --load_path=${CKPT}

echo "=== [$(date)] full eval done ==="
