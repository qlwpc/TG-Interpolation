#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN=${1:?usage: run_base_eval.sh CAMPAIGN INDEX}
INDEX=${2:?usage: run_base_eval.sh CAMPAIGN INDEX}
WORKSPACE=${PAUSE_WORKSPACE:-/home/wangpch/TG-Interpolation}
SCRIPT_ROOT=${PAUSE_SCRIPT_ROOT:-${WORKSPACE}/scripts}
RUNTIME_BIN=${PAUSE_RUNTIME_BIN:-/home/wangpch/.conda/envs/LLM/bin}
CAMPAIGN_DRIVER=${PAUSE_CAMPAIGN_DRIVER:-${SCRIPT_ROOT}/pause_eval_campaign.py}
export PATH=${RUNTIME_BIN}:${PATH}
ROW=$(sed -n "$((INDEX + 2))p" "${CAMPAIGN}/base_runs.tsv")
IFS=$'\t' read -r _ RUN_NAME TASK CHECKPOINT CONFIG RUN_DIR GPUS <<< "${ROW}"
test -n "${RUN_NAME}"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/output" "${RUN_DIR}/mpl"
exec > >(tee -a "${RUN_DIR}/eval.log") 2>&1
export PYTHONPATH=${WORKSPACE}${PYTHONPATH:+:${PYTHONPATH}}
export MPLCONFIGDIR=${RUN_DIR}/mpl
export WANDB_MODE=disabled
cd "${WORKSPACE}"

if [ -f "${RUN_DIR}/EVAL_DONE" ]; then
  exit 0
fi
if [ -f "${CAMPAIGN}/smoke/SMOKE_DONE" ]; then
  echo "CAMPAIGN_VERIFY inherited_from_smoke"
else
  python "${CAMPAIGN_DRIVER}" verify --campaign-dir "${CAMPAIGN}"
fi
CONFIG_PATH=${CONFIG} EXPECTED_CHECKPOINT=${CHECKPOINT} python - <<'PY'
import os
from olmo.config import TrainConfig
c = TrainConfig.load(os.environ["CONFIG_PATH"], validate_paths=False)
assert c.model.transformer_grammar_type.startswith("pause")
assert c.model.pause_token_id == 50261
assert c.load_path == os.environ["EXPECTED_CHECKPOINT"]
assert c.eval_on_load and c.eval_no_save and str(c.max_duration) in {"0", "0ep"}
print("CONFIG_OK", c.run_name, c.model.transformer_grammar_type, c.model.pause_token_id)
PY
python - <<'PY'
import json, platform, torch
print("ENVIRONMENT", json.dumps({"host": platform.node(), "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda}))
PY
nvidia-smi
PORT=$((10000 + (${SLURM_JOB_ID:-1000} + INDEX) % 50000))
"${RUNTIME_BIN}/torchrun" --master_port="${PORT}" --nproc-per-node="${GPUS}" \
  "${SCRIPT_ROOT}/train.py" "${CONFIG}" \
  --run_name="${RUN_NAME}" --workspace="${WORKSPACE}" --save_folder="${RUN_DIR}/output" \
  --load_path="${CHECKPOINT}" --eval_on_load=true --eval_return=true --eval_no_save=true \
  --max_duration=0 --stop_at=0 --reset_optimizer_state=true --reset_trainer_state=true

case "${TASK}" in
  sg) grep -q '^    avg=' "${RUN_DIR}/eval.log" ;;
  blimp) grep -q '^    overall/overall=' "${RUN_DIR}/eval.log" ;;
  docppl) grep -q '^    eval/downstream/terminal_doc_ppl_doc_ppl=' "${RUN_DIR}/eval.log" ;;
  *) exit 22 ;;
esac
touch "${RUN_DIR}/EVAL_DONE"
