#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SESSION="mevo_conditional_trace_full"
LOG_DIR="$ROOT/logs/20260809_conditional_trace_full"
LOG="$LOG_DIR/mevo_conditional_trace_full_$(date +%Y%m%d_%H%M%S).log"

if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "tmux session 已存在: $SESSION" >&2
  exit 1
fi
if [[ -z "${VIS_API_KEY:-}" ]]; then
  echo "VIS_API_KEY 未导出到当前环境" >&2
  exit 1
fi
mkdir -p "$LOG_DIR"
tmux set-environment -g VIS_API_KEY "$VIS_API_KEY"
tmux new-session -d -s "$SESSION" -c "$ROOT" \
  "bash -lc 'unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; set -o pipefail; bash code/run_conditional_trace_full_experiment.sh 2>&1 | tee \"$LOG\"; status=\${PIPESTATUS[0]}; echo PIPELINE_EXIT=\$status | tee -a \"$LOG\"; exec bash'"

echo "session=$SESSION"
echo "log=$LOG"
echo "查看: tmux attach -t $SESSION"
