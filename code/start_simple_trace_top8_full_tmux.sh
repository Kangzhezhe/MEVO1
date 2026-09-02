#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SESSION="mevo_simple_trace_top8_full"
LOG_DIR="$ROOT/logs/20260814_simple_trace_top8_full"
LOG="$LOG_DIR/mevo_simple_trace_top8_full_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && bash '$HERE/run_simple_trace_top8_full_experiment.sh' 2>&1 | tee '$LOG'; status=\${PIPESTATUS[0]}; echo PIPELINE_EXIT=\$status | tee -a '$LOG'; exec bash"

echo "session=$SESSION"
echo "log=$LOG"
echo "attach: tmux attach -t $SESSION"
