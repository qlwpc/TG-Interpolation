#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 00:10:00
set -euo pipefail
: "${PAUSE_CAMPAIGN_DIR:?PAUSE_CAMPAIGN_DIR is required}"
: "${PAUSE_EVAL_ARRAY_ID:?PAUSE_EVAL_ARRAY_ID is required}"

for index in 0 1 2 3 4; do
  scontrol release "${PAUSE_EVAL_ARRAY_ID}_${index}"
done

date --iso-8601=seconds > "${PAUSE_CAMPAIGN_DIR}/XSUM_RELEASED_AFTER_BOOLQ"
