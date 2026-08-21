#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=1M
#SBATCH -t 00:05:00
#SBATCH --job-name=flexdiag-release-eval
#SBATCH --output=/home/wangpch/TG-Interpolation/artifacts/experiment/flexattention_illegal_access_diagnostic_20260820/release-eval-%j.out

set -euo pipefail
scontrol release 2981
echo "RELEASED job_array=2981 after_flex_diagnostics time=$(date --iso-8601=seconds)"
