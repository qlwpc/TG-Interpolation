#!/usr/bin/env bash
# Exact, small GPST document-PPL objective audit.  It deliberately runs both
# prefix choices against one whole document, preserving all 300 candidates.
#SBATCH --job-name=gpst-objective-audit
#SBATCH --partition=normal
#SBATCH --nodelist=rtx3090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=docppl_runs/gpst_objective_audit_%j.driver.log
#SBATCH --error=docppl_runs/gpst_objective_audit_%j.driver.log

set -euo pipefail
cd /home/wangpch/TG-Interpolation
mkdir -p docppl_runs/gpst_objective_audit_${SLURM_JOB_ID}
export PYTHONPATH=.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python_bin=/home/wangpch/.conda/envs/LLM/bin/python
for prefix_mode in candidate0 joint-best; do
  output="docppl_runs/gpst_objective_audit_${SLURM_JOB_ID}/${prefix_mode}.json"
  temporary="${output}.partial"
  /usr/bin/time -v "$python_bin" scripts/gpst/diagnose_document_ppl_objective.py \
    --checkpoint saved_models/gpst-bbc-unsup/model.bin \
    --native-data dataset/bbc-news/native_model_topk_300_v2 \
    --start-document 0 --end-document 1 --prefix-mode "$prefix_mode" \
    --eval-batch-size 16 --max-batch-actions 32768 --log-every 10 \
    >"$temporary" 2>"${output}.log"
  mv "$temporary" "$output"
done
