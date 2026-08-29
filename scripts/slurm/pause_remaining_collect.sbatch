#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 00:10:00
set -euo pipefail
: "${PAUSE_CAMPAIGN_DIR:?PAUSE_CAMPAIGN_DIR is required}"

for seed in 6198 13171 31723 42 2026; do
  run_dir="${PAUSE_CAMPAIGN_DIR}/runs/pause1_sep50261_xsum_seed${seed}"
  test -f "${run_dir}/TRAIN_DONE"
  test -f "${run_dir}/EVAL_DONE"
  grep -q '^    R-AVG=' "${run_dir}/eval.log"
done

boolq_dir="${PAUSE_CAMPAIGN_DIR}/runs/pause1_sep50261_boolq_seed2026"
test -f "${boolq_dir}/TRAIN_DONE"
test -f "${boolq_dir}/EVAL_DONE"
grep -q 'eval/downstream/boolq_acc__=' "${boolq_dir}/eval.log"

date --iso-8601=seconds > "${PAUSE_CAMPAIGN_DIR}/REMOTE_REMAINING_DONE"
