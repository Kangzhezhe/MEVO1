#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SESSION="mevo_no_trace_ablation"
LOG_DIR="$ROOT/logs/20260824_no_trace_ablation"
LOG="$LOG_DIR/mevo_no_trace_ablation_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && bash '$HERE/run_full.sh' 2>&1 | tee '$LOG'; status=\${PIPESTATUS[0]}; echo PIPELINE_EXIT=\$status | tee -a '$LOG'; exec bash"

echo "session=$SESSION"
echo "log=$LOG"
echo "attach: tmux attach -t $SESSION"
