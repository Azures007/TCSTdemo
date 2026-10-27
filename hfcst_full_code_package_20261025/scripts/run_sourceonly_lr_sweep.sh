#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${DATA_ROOT:?Please export DATA_ROOT to the directory containing Houston, HyRANK, and Shanghai_Hangzhou}"
DEVICE="${DEVICE:-cuda:0}"
mkdir -p logs_hfcst

run_round() {
  local tag="$1"
  shift
  local pids=()
  echo "===== Source-only LR sweep: ${tag}; device=${DEVICE} ====="
  for cfg in "$@"; do
    local log="logs_hfcst/$(basename "${cfg%.yaml}").log"
    echo "launch: ${cfg} -> ${log}"
    python -u scripts/train_hfcst.py --config "$cfg" --device "$DEVICE" > "$log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "At least one run failed in ${tag}. Check logs_hfcst/*.log before proceeding." >&2
    return 1
  fi
}

# First compare 1e-4 across all three datasets, then 3e-4 across all three.
# The previous 3e-5 runs are the reference baseline. num_cycles=0 isolates source-only
# classifier quality from the adaptation/rollback problem.
run_round lr1e-4 \
  configs_hfcst/audit_sourceonly_houston_lr1e-4.yaml \
  configs_hfcst/audit_sourceonly_hyrank_lr1e-4.yaml \
  configs_hfcst/audit_sourceonly_shanghai_hangzhou_lr1e-4.yaml

run_round lr3e-4 \
  configs_hfcst/audit_sourceonly_houston_lr3e-4.yaml \
  configs_hfcst/audit_sourceonly_hyrank_lr3e-4.yaml \
  configs_hfcst/audit_sourceonly_shanghai_hangzhou_lr3e-4.yaml

echo "All source-only runs completed. Compare baseline_source_validation OA/AA and per-class accuracy first."
echo "Do not choose hyperparameters using target GT metrics; target metrics are final-only reports."
