#!/usr/bin/env bash
# Resume one complete-document interval of the native GPST document-PPL run.
# Submit with START_DOCUMENT and END_DOCUMENT in the environment, e.g.
# sbatch --export=ALL,START_DOCUMENT=3930,END_DOCUMENT=3931,EVAL_BATCH_SIZE=16 \
#   scripts/gpst_native_docppl_resume.sbatch
#
# The JSON result is atomically renamed only after successful evaluation.  A
# cancelled job therefore cannot masquerade as a valid zero-byte shard.
#SBATCH --job-name=gpst-native-docppl
#SBATCH --partition=normal
#SBATCH --nodelist=rtx3090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
# The RTX3090 login-node QOS accepts one accounting CPU per batch task.  This
# is a scheduler admission requirement, not the memory-control mechanism.
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --output=docppl_runs/gpst_native_resume_%j.driver.log
#SBATCH --error=docppl_runs/gpst_native_resume_%j.driver.log

set -euo pipefail

: "${START_DOCUMENT:?set START_DOCUMENT when submitting}"
: "${END_DOCUMENT:?set END_DOCUMENT when submitting}"
if (( START_DOCUMENT >= END_DOCUMENT )); then
  echo "invalid document interval [$START_DOCUMENT, $END_DOCUMENT)" >&2
  exit 2
fi

cd /home/wangpch/TG-Interpolation
mkdir -p docppl_runs/gpst_native_resume_20260823
export PYTHONPATH=.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
python_bin=/home/wangpch/.conda/envs/LLM/bin/python

name="documents_${START_DOCUMENT}_to_${END_DOCUMENT}"
result="docppl_runs/gpst_native_resume_20260823/${name}.json"
temporary="${result}.partial"
log="docppl_runs/gpst_native_resume_20260823/${name}.log"
monitor="docppl_runs/gpst_native_resume_20260823/${name}.memory.tsv"
eval_batch_size=${EVAL_BATCH_SIZE:-16}
max_batch_actions=${MAX_BATCH_ACTIONS:-32768}
rm -f "$temporary"

# Memory requests are not an enforcement mechanism on this login-node Slurm
# setup.  Measure actual behavior instead: the first resumed long document
# runs with batch 16, then its RSS/GPU peak determines safe later concurrency.
# A smaller candidate batch is likelihood-equivalent to the previous batch 64.
/usr/bin/time -v "$python_bin" scripts/gpst/evaluate_document_ppl.py \
  --checkpoint saved_models/gpst-bbc-unsup/model.bin \
  --native-data dataset/bbc-news/native_model_topk_300_v2 \
  --start-document "$START_DOCUMENT" --end-document "$END_DOCUMENT" \
  --eval-batch-size "$eval_batch_size" --max-batch-actions "$max_batch_actions" --log-every 25 \
  >"$temporary" 2>"$log" &
child_pid=$!
printf 'epoch_s\tpid\trss_kb\tmem_available_kb\tgpu_memory_used_mb\n' >"$monitor"
while kill -0 "$child_pid" 2>/dev/null; do
  # GNU time is a small parent process; sample its Python child when present.
  observed_pid=$(cat "/proc/${child_pid}/task/${child_pid}/children" 2>/dev/null | awk '{print $1}')
  observed_pid=${observed_pid:-$child_pid}
  rss=$(awk '/VmRSS:/ {print $2}' "/proc/${observed_pid}/status" 2>/dev/null || echo 0)
  available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd, -)
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" "$observed_pid" "$rss" "$available" "$gpu" >>"$monitor"
  sleep 30
done
wait "$child_pid"
mv "$temporary" "$result"
