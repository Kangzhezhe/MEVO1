#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SESSION="${SESSION:-mevo_trace_warmup_idpo}"
LOG_DIR="$ROOT/logs/20260811_trace_warmup_idpo"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/mevo_trace_warmup_idpo_${STAMP}.log"
mkdir -p "$LOG_DIR"

tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && bash '$HERE/run_trace_warmup_idpo_experiment.sh' 2>&1 | tee '$LOG'; status=\${PIPESTATUS[0]}; echo PIPELINE_EXIT=\$status | tee -a '$LOG'; exec bash"
echo "session=$SESSION"
echo "log=$LOG"
