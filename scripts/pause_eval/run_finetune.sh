#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN=${1:?usage: run_finetune.sh CAMPAIGN INDEX}
INDEX=${2:?usage: run_finetune.sh CAMPAIGN INDEX}
WORKSPACE=${PAUSE_WORKSPACE:-/home/wangpch/TG-Interpolation}
SCRIPT_ROOT=${PAUSE_SCRIPT_ROOT:-${WORKSPACE}/scripts}
RUNTIME_BIN=${PAUSE_RUNTIME_BIN:-/home/wangpch/.conda/envs/LLM/bin}
CAMPAIGN_DRIVER=${PAUSE_CAMPAIGN_DRIVER:-${SCRIPT_ROOT}/pause_eval_campaign.py}
XSUM_PIPELINE_VERSION=2
export PATH=${RUNTIME_BIN}:${PATH}
ROW=$(sed -n "$((INDEX + 2))p" "${CAMPAIGN}/finetune_runs.tsv")
IFS=$'\t' read -r _ RUN_NAME TASK SEED CHECKPOINT CONFIG RUN_DIR SAVE EVAL_BATCH <<< "${ROW}"
test -n "${RUN_NAME}"
mkdir -p "${RUN_DIR}" "${SAVE}" "${RUN_DIR}/mpl-train"
exec > >(tee -a "${RUN_DIR}/train.log") 2>&1
export PYTHONPATH=${WORKSPACE}${PYTHONPATH:+:${PYTHONPATH}}
export MPLCONFIGDIR=${RUN_DIR}/mpl-train
export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${WORKSPACE}"

if [ -f "${RUN_DIR}/TRAIN_DONE" ] && [ -f "$(cat "${RUN_DIR}/final_checkpoint.txt" 2>/dev/null)/model.pt" ]; then
  if [ "${TASK}" = xsum ]; then
    CONTRACT=${RUN_DIR}/training_contract.json
    if ! CONTRACT_PATH=${CONTRACT} EXPECTED_VERSION=${XSUM_PIPELINE_VERSION} \
      "${RUNTIME_BIN}/python" -c \
      'import json, os; p=json.load(open(os.environ["CONTRACT_PATH"])); assert p["xsum_pipeline_version"] == int(os.environ["EXPECTED_VERSION"])'; then
      echo "STALE_XSUM_CHECKPOINT missing/invalid ${CONTRACT}; use a new campaign directory" >&2
      exit 42
    fi
  fi
  exit 0
fi
if [ -f "${CAMPAIGN}/smoke/SMOKE_DONE" ]; then
  echo "CAMPAIGN_VERIFY inherited_from_smoke"
else
  python "${CAMPAIGN_DRIVER}" verify --campaign-dir "${CAMPAIGN}"
fi
CONFIG_PATH=${CONFIG} TASK_NAME=${TASK} python - <<'PY'
import os
from olmo.config import TrainConfig
c = TrainConfig.load(os.environ["CONFIG_PATH"], validate_paths=False)
assert c.model.transformer_grammar_type.startswith("pause")
assert c.model.pause_token_id == 50261
assert c.finetune_task == os.environ["TASK_NAME"]
assert not c.eval_on_load and c.reset_optimizer_state and c.reset_trainer_state
print("CONFIG_OK", c.run_name, c.finetune_task, c.seed, c.model.pause_token_id)
PY
nvidia-smi
PORT=$((12000 + (${SLURM_ARRAY_JOB_ID:-1000} + ${SLURM_ARRAY_TASK_ID:-INDEX}) % 40000))
TRAIN_OVERRIDES=()
if [ -n "${DEVICE_TRAIN_MICROBATCH_SIZE_OVERRIDE:-}" ]; then
  if ! [[ "${DEVICE_TRAIN_MICROBATCH_SIZE_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
    exit 22
  fi
  TRAIN_OVERRIDES+=("--device_train_microbatch_size=${DEVICE_TRAIN_MICROBATCH_SIZE_OVERRIDE}")
fi
set +e
"${RUNTIME_BIN}/torchrun" --master_port="${PORT}" --nproc-per-node=4 \
  "${SCRIPT_ROOT}/train.py" "${CONFIG}" \
  --run_name="${RUN_NAME}" --workspace="${WORKSPACE}" --save_folder="${SAVE}" \
  --save_overwrite=true --try_load_latest_save=true --load_path="${CHECKPOINT}" \
  --evaluators='[]' "${TRAIN_OVERRIDES[@]}"
TRAIN_PROCESS_RC=$?
set -e
echo "TRAIN_PROCESS_EXIT code=${TRAIN_PROCESS_RC}"
FINAL=$(readlink -f "${SAVE}/latest-unsharded" 2>/dev/null || true)
if [ -z "${FINAL}" ] || [ ! -f "${FINAL}/model.pt" ]; then
  FINAL=$(find "${SAVE}" -maxdepth 1 -type d -name 'step*-unsharded' -printf '%f\t%p\n' | sort -t $'\t' -k1,1V | tail -n 1 | cut -f2-)
fi
test -f "${FINAL}/model.pt"
test -f "${FINAL}/config.yaml"
if [ "${TRAIN_PROCESS_RC}" -ne 0 ]; then
  if ! grep -q 'Training complete' "${RUN_DIR}/train.log"; then
    exit "${TRAIN_PROCESS_RC}"
  fi
  echo "TRAIN_PROCESS_EXIT accepted_after_durable_final_checkpoint"
fi
"${RUNTIME_BIN}/python" "${SCRIPT_ROOT}/check_checkpoint_health.py" \
  "${FINAL}/model.pt" --output "${RUN_DIR}/checkpoint_health.json"
printf '%s\n' "${FINAL}" > "${RUN_DIR}/final_checkpoint.txt"
sha256sum "${FINAL}/model.pt" > "${RUN_DIR}/final_model.sha256"
rm -f -- "${FINAL}/optim.pt" "${FINAL}/train.pt"
if [ "${TASK}" = xsum ]; then
  CONTRACT_PATH=${RUN_DIR}/training_contract.json \
  PIPELINE_VERSION=${XSUM_PIPELINE_VERSION} \
  TASK_NAME=${TASK} CONFIG_PATH=${CONFIG} CHECKPOINT_PATH=${CHECKPOINT} \
  SOURCE_ROOT=${WORKSPACE} "${RUNTIME_BIN}/python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

root = Path(os.environ["SOURCE_ROOT"])
payload = {
    "xsum_pipeline_version": int(os.environ["PIPELINE_VERSION"]),
    "task": os.environ["TASK_NAME"],
    "config": os.environ["CONFIG_PATH"],
    "config_sha256": sha256(os.environ["CONFIG_PATH"]),
    "base_checkpoint": os.environ["CHECKPOINT_PATH"],
    "source_sha256": {
        str(path.relative_to(root)): sha256(path)
        for path in (
            root / "olmo/data/__init__.py",
            root / "olmo/eval/downstream.py",
            root / "olmo/model.py",
            root / "olmo/train.py",
        )
    },
}
Path(os.environ["CONTRACT_PATH"]).write_text(json.dumps(payload, indent=2) + "\n")
PY
fi
touch "${RUN_DIR}/TRAIN_DONE"
