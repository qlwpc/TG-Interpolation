#!/usr/bin/env bash
# Monitor SIST, this host, RTX3090, and A6000 Slurm GPU clusters.
# Usage: scripts/slurm/gpu_cluster_status.sh [--watch SECONDS]

set -euo pipefail

watch_seconds=""
if [[ ${1:-} == "--watch" ]]; then
  watch_seconds="${2:-10}"
  [[ "$watch_seconds" =~ ^[1-9][0-9]*$ ]] || {
    echo "--watch requires a positive integer." >&2
    exit 2
  }
  shift 2
fi
if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  sed -n '2,3p' "$0"
  exit 0
fi
[[ $# -eq 0 ]] || {
  echo "Usage: $0 [--watch SECONDS]" >&2
  exit 2
}

# Print scheduling allocation per GPU node, visible queue counts, and (where
# permitted) live NVIDIA telemetry. Job arrays are expanded before counting.
general_query='\
set -o pipefail
mapfile -t nodes < <(sinfo -N -h -o "%N" | sort -u)
printf "%-18s %-10s %-24s %7s %7s %7s %s\\n" NODE PARTITION GPU_MODEL TOTAL ALLOC FREE STATE
printf "%-18s %-10s %-24s %7s %7s %7s %s\\n" "------------------" "----------" "------------------------" "-------" "-------" "-------" "-----"
for node in "${nodes[@]}"; do
  info=$(scontrol show node -o "$node")
  gres=$(grep -o "Gres=[^ ]*" <<<"$info" | cut -d= -f2)
  [[ "$gres" == *"gpu:"* ]] || continue
  total=${gres##*:}
  model=${gres#gpu:}
  if [[ "$model" == *:* ]]; then model=${model%:*}; else model="unspecified"; fi
  state=$(grep -o "State=[^ ]*" <<<"$info" | cut -d= -f2)
  alloc_tres=$(grep -o "AllocTRES=[^ ]*" <<<"$info" || true)
  alloc=0
  [[ "$alloc_tres" =~ gres/gpu=([0-9]+) ]] && alloc=${BASH_REMATCH[1]}
  free=$((total - alloc))
  partitions=$(sinfo -N -h -n "$node" -o "%P" | sed "s/\*//g" | paste -sd, -)
  printf "%-18s %-10s %-24s %7d %7d %7d %s\\n" "$node" "$partitions" "$model" "$total" "$alloc" "$free" "$state"
done
echo
printf "Visible queue tasks (arrays expanded): pending="; squeue -h -t PENDING -r | wc -l
printf "Visible queue tasks (arrays expanded): running="; squeue -h -t RUNNING -r | wc -l
echo "Pending reasons"
squeue -h -t PENDING -r -o "%P|%R" | awk -F"|" "{n[\$1 \"|\" \$2]++} END {for (k in n) print n[k] \"|\" k}" | sort -nr || true
echo
echo "Live GPU telemetry: index, name, GPU %, memory %, used MiB, total MiB, temperature C, power W"
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits || echo "UNAVAILABLE: nvidia-smi/NVML cannot be read here."
'

# SIST exposes scheduler allocation but not the other users' queue, and direct
# nvidia-smi requires compute-node access. Discover all current nodes in the
# requested partitions (including new GPU types) and omit all TITAN-family GPUs.
sist_query='\
export PATH=/opt/gridview/slurm/bin:$PATH
set -o pipefail
partitions="critical,ShangHAI"
mapfile -t nodes < <(sinfo -N -h -p "$partitions" -o "%N" | sort -u)
declare -A model_total model_alloc model_nodes
printf "%-18s %-10s %-24s %7s %7s %7s %s\\n" NODE PARTITION GPU_MODEL TOTAL ALLOC FREE STATE
printf "%-18s %-10s %-24s %7s %7s %7s %s\\n" "------------------" "----------" "------------------------" "-------" "-------" "-------" "-----"
for node in "${nodes[@]}"; do
  info=$(scontrol show node -o "$node")
  gres=$(grep -o "Gres=[^ ]*" <<<"$info" | cut -d= -f2)
  [[ "$gres" == *"gpu:"* ]] || continue
  model=${gres#gpu:}; model=${model%:*}
  [[ "${model^^}" == *TITAN* ]] && continue
  total=${gres##*:}
  state=$(grep -o "State=[^ ]*" <<<"$info" | cut -d= -f2)
  alloc_tres=$(grep -o "AllocTRES=[^ ]*" <<<"$info" || true)
  alloc=0
  [[ "$alloc_tres" =~ gres/gpu=([0-9]+) ]] && alloc=${BASH_REMATCH[1]}
  free=$((total - alloc))
  node_partitions=$(sinfo -N -h -n "$node" -p "$partitions" -o "%P" | sed "s/\*//g" | paste -sd, -)
  printf "%-18s %-10s %-24s %7d %7d %7d %s\\n" "$node" "$node_partitions" "$model" "$total" "$alloc" "$free" "$state"
  ((model_total["$model"] += total))
  ((model_alloc["$model"] += alloc))
  ((model_nodes["$model"] += 1))
done
echo
echo "GPU model summary"
printf "%-24s %7s %7s %7s %7s\\n" GPU_MODEL NODES TOTAL ALLOC FREE
for model in "${!model_total[@]}"; do
  printf "%-24s %7d %7d %7d %7d\\n" "$model" "${model_nodes[$model]}" "${model_total[$model]}" "${model_alloc[$model]}" "$((model_total[$model] - model_alloc[$model]))"
done | sort
echo
echo "SIST note: queue visibility is restricted for this account; values above are Slurm GPU allocations, not live GPU utilization."
'

run_remote() {
  local host="$1" query="$2"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$host" "bash -lc $(printf '%q' "$query")"
}

run_once() {
  echo '========== SIST: critical + ShangHAI (no TITAN) =========='
  run_remote SIST "$sist_query"
  echo
  echo '========== Local RTX 3090 cluster =========='
  bash -lc "$general_query"
  echo
  echo '========== RTX3090 SSH cluster =========='
  run_remote RTX3090 "$general_query"
  echo
  echo '========== A6000 SSH cluster =========='
  run_remote A6000 "$general_query"
}

if [[ -n "$watch_seconds" ]]; then
  while true; do
    printf '\033[H\033[2JGPU cluster snapshot: %s\n\n' "$(date '+%F %T %Z')"
    run_once
    sleep "$watch_seconds"
  done
else
  run_once
fi
