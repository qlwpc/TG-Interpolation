#!/bin/bash
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem-per-cpu=1M
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH --job-name=tree_blimp_smoke
#SBATCH --output=analysis-output/blimp_beam/logs/tree_smoke_%j.log
set -euo pipefail

# 1-GPU smoke test for beam-search BLiMP on the `tree` model (Tree_test).
# Tiny subset (pair_per_task=2, subset_num_batches=4, beam_size=20) to confirm
# the full pipeline (Trainer.eval -> BLiMP_beam_eval_step -> record_beams ->
# compute -> JSON dump) runs end-to-end on GPU with no shape/device errors and
# a finite accuracy, before launching the full 50-pair run.
#
# Usage:  sbatch evaluation/eval_scripts/test_blimp_beam_tree_smoke.sh

source /data/software/anaconda3/etc/profile.d/conda.sh
conda activate LLM
export PYTHONPATH=/home/wangpch/TG-Interpolation:${PYTHONPATH:-}
export HF_ENDPOINT=https://hf-mirror.com
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OLMO_BEAM_DUMP=1

cd /home/wangpch/TG-Interpolation
mkdir -p analysis-output/tree_eval_BLiMP_beam/smoke analysis-output/blimp_beam/logs

CFG=train_configs/eval_per_metric/tree_BLiMP_beam_smoke.yaml
CKPT=/home/wangpch/TG-Interpolation/saved_models/Tree_test/step49440-unsharded

echo "=== [$(date)] BLiMP beam smoke (tree model, pair_per_task=2, subset=4, beam_size=20) ==="
echo "CFG=$CFG  CKPT=$CKPT  OLMO_BEAM_DUMP=$OLMO_BEAM_DUMP"
python -c "import torch; print('devices', torch.cuda.device_count())"

torchrun --nproc-per-node=1 --master_port=17406 \
  scripts/train.py ${CFG} \
    --run_name=tree_blimp_beam_smoke \
    --workspace=/home/wangpch/TG-Interpolation \
    --save_folder=analysis-output/tree_eval_BLiMP_beam/smoke \
    --save_overwrite \
    --load_path=${CKPT}

echo "=== [$(date)] smoke done; dump at analysis-output/tree_eval_BLiMP_beam/smoke/beam_trees_BLiMP.jsonl ==="
