#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CONFIG="${1:-$HERE/config_idpo_pilot.yaml}"
SPLIT="${2:-validation}"
SESSION="${IDPO_TMUX_SESSION:-mevo_idpo_pilot}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/idpo"
LOG="$LOG_DIR/${SESSION}_${STAMP}.log"

mkdir -p "$LOG_DIR"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "attach with: tmux attach -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && '$HERE/run_idpo_pilot.sh' '$CONFIG' '$SPLIT' 2>&1 | tee '$LOG'"
echo "started $SESSION"
echo "log: $LOG"
echo "attach: tmux attach -t $SESSION"
