#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
QWEN_SESSION="mevo_s1_full_batch8"
QWEN_REPORT_DIR="$ROOT/result/perpcs_s1_free_trace_pipeline_full_v1"
QWEN_VALIDATION="$QWEN_REPORT_DIR/validation_report.json"
QWEN_TEST="$QWEN_REPORT_DIR/test_report.json"
LLAMA_REPORT="$ROOT/result/perpcs_s1_free_trace_llama2_7b_pipeline_full_v1/test_report.json"
CONFIG="$HERE/config_s1_llama2_7b_full.yaml"

if [[ -f "$LLAMA_REPORT" ]]; then
  echo "Llama-2-7B full report already exists: $LLAMA_REPORT"
  exit 0
fi

echo "Waiting for the Qwen S1 full pipeline to finish successfully..."
while true; do
  session_state=done
  validation_state=pending
  test_state=pending
  tmux has-session -t "$QWEN_SESSION" 2>/dev/null && session_state=running
  [[ -f "$QWEN_VALIDATION" ]] && validation_state=ready
  [[ -f "$QWEN_TEST" ]] && test_state=ready
  echo "Qwen gate: session=$session_state validation=$validation_state test=$test_state"
  if [[ "$session_state" == "done" && "$validation_state" == "ready" && "$test_state" == "ready" ]]; then
    break
  fi
  sleep 60
done

echo "Qwen S1 full pipeline completed; starting Llama-2-7B Editor pipeline."
bash "$HERE/run_pipeline.sh" "$CONFIG"
echo "LLAMA2_7B_PIPELINE_EXIT=0"
