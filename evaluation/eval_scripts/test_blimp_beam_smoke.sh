#!/bin/bash
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem-per-cpu=1M
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH --job-name=blimp_beam_smoke
#SBATCH --output=analysis-output/blimp_beam/logs/smoke_%j.log
set -euo pipefail

# 1-GPU smoke test for beam-search BLiMP.
# Runs the full pipeline (Trainer.eval -> BLiMP_beam_eval_step ->
# BLiMPMetric.update_beam -> compute) on a handful of sentences to confirm
# there are no shape/device errors and the result is finite, before launching
# the full eval.
#
# Usage:  sbatch evaluation/eval_scripts/test_blimp_beam_smoke.sh
# (edit CKPT / CFG below to switch treereg<->pushdown)

source /data/software/anaconda3/etc/profile.d/conda.sh
conda activate LLM
export PYTHONPATH=/home/wangpch/TG-Interpolation:${PYTHONPATH:-}
export HF_ENDPOINT=https://hf-mirror.com
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/wangpch/TG-Interpolation
mkdir -p analysis-output/blimp_beam/smoke analysis-output/blimp_beam/logs

CFG=train_configs/eval_per_metric/treereg_BLiMP_beam_smoke.yaml
CKPT=/home/wangpch/TG-Interpolation/saved_models/treereg/step33862-unsharded

echo "=== [$(date)] BLiMP beam smoke (1 GPU, subset_num_batches=4) ==="
echo "CFG=$CFG  CKPT=$CKPT"
python -c "import torch; print('devices', torch.cuda.device_count())"

torchrun --nproc-per-node=1 --master_port=17401 \
  scripts/train.py ${CFG} \
    --run_name=blimp_beam_smoke \
    --workspace=/home/wangpch/TG-Interpolation \
    --save_folder=analysis-output/blimp_beam/smoke \
    --save_overwrite \
    --load_path=${CKPT}

echo "=== [$(date)] smoke done ==="
