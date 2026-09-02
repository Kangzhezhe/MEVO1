#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PYTHON="${DATA_PYTHON:-$ROOT/.venv/bin/python}"
VALIDATION="$ROOT/result/perpcs_s1_free_trace_pipeline_full_v1/validation_report.json"
TEST="$ROOT/result/perpcs_s1_free_trace_pipeline_full_v1/test_report.json"

echo "Waiting for full S1 Per-Pcs validation/test reports..."
ticks=0
while [[ ! -f "$VALIDATION" || ! -f "$TEST" ]]; do
  if ! pgrep -f "run_s0_s1_s2_gpu_queue.sh" >/dev/null; then
    echo "Full experiment queue exited before both reports were written" >&2
    exit 2
  fi
  ticks=$((ticks + 1))
  if (( ticks % 10 == 0 )); then
    traces=0
    trace_file="$ROOT/dataset/candidate_sets/perpcs_s1_full_train/04_gold_aware_traces.jsonl"
    [[ -f "$trace_file" ]] && traces="$(wc -l < "$trace_file")"
    echo "finalizer wait: full_traces=$traces validation=$([[ -f "$VALIDATION" ]] && echo ready || echo pending) test=$([[ -f "$TEST" ]] && echo ready || echo pending)"
  fi
  sleep 60
done

"$PYTHON" "$HERE/16_finalize_full_experiment.py" \
  --selection "$ROOT/result/s0_s1_s2_comparison/best_method.json" \
  --subset "$ROOT/result/s0_s1_s2_comparison/report.json" \
  --output "$ROOT/result/s0_s1_s2_comparison/final_full_report.json"
echo "FULL_FINALIZER_EXIT=0"
