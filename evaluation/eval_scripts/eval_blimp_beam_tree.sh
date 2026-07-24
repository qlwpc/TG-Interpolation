#!/bin/bash
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem-per-cpu=1M
#SBATCH --gres=gpu:4
#SBATCH -t 24:00:00
#SBATCH --job-name=tree_blimp_beam
#SBATCH --output=analysis-output/blimp_beam/logs/tree_%j.log
set -euo pipefail

# Beam-search BLiMP subset eval on the 100M `tree` model (Tree_test), 4 GPUs.
# - Scores 50 minimal pairs x 67 tasks (pair_per_task=50) via
#   OLMo.word_sync_beam_search (parse-marginalized log-likelihood), beam_size=100.
# - 4-GPU data-parallel: DistributedEvalSampler splits the 3350 sentences across
#   ranks (~838/rank). ~70 min wall-clock.
# - OLMO_BEAM_DUMP=1 writes beam_trees_BLiMP_rank{R}.jsonl (top-5 bracketed trees
#   per sentence, per rank) into save_folder for offline comparison vs
#   blimp_tree_300.npy.
#
# Usage:  sbatch evaluation/eval_scripts/eval_blimp_beam_tree.sh

source /data/software/anaconda3/etc/profile.d/conda.sh
conda activate LLM
export PYTHONPATH=/home/wangpch/TG-Interpolation:${PYTHONPATH:-}
export HF_ENDPOINT=https://hf-mirror.com
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Enable per-sentence beam-tree dump (per-rank _rank{R}.jsonl under save_folder).
export OLMO_BEAM_DUMP=1

cd /home/wangpch/TG-Interpolation
mkdir -p analysis-output/tree_eval_BLiMP_beam analysis-output/blimp_beam/logs

CFG=train_configs/eval_per_metric/tree_BLiMP_beam.yaml
CKPT=/home/wangpch/TG-Interpolation/saved_models/Tree_test/step49440-unsharded

echo "=== [$(date)] BLiMP beam subset eval (tree model, 4 GPUs, pair_per_task=50, beam_size=100) ==="
echo "CFG=$CFG  CKPT=$CKPT  OLMO_BEAM_DUMP=$OLMO_BEAM_DUMP"
python -c "import torch; print('devices', torch.cuda.device_count())"

torchrun --nproc-per-node=4 --master_port=17405 \
  scripts/train.py ${CFG} \
    --run_name=tree_eval_BLiMP_beam \
    --workspace=/home/wangpch/TG-Interpolation \
    --save_folder=analysis-output/tree_eval_BLiMP_beam \
    --save_overwrite \
    --load_path=${CKPT}

echo "=== [$(date)] eval done; beam trees at analysis-output/tree_eval_BLiMP_beam/beam_trees_BLiMP_rank*.jsonl ==="
echo "Compare: python analysis/sg_degradation/compare_beam_trees_tree300.py \\"
echo "  --dump analysis-output/tree_eval_BLiMP_beam/beam_trees_BLiMP \\"
echo "  --tree300 dataset/BLiMP/tree300/blimp_tree_300.npy \\"
echo "  --out analysis-output/blimp_beam/tree_comparison/"
