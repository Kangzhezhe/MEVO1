#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SESSION="mevo_idpo_gold_test_all"
LOG_DIR="$ROOT/logs/20260802_idpo_gold_test_all"
LOG="$LOG_DIR/mevo_idpo_gold_test_all_$(date +%Y%m%d_%H%M%S).log"

if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "tmux session 已存在: $SESSION" >&2
  exit 1
fi
if [[ -z "${QWEN_API_KEY:-}" ]]; then
  echo "QWEN_API_KEY 未导出到当前环境" >&2
  exit 1
fi
mkdir -p "$LOG_DIR"
tmux set-environment -g QWEN_API_KEY "$QWEN_API_KEY"
tmux new-session -d -s "$SESSION" -c "$ROOT" \
  "bash -lc 'set -o pipefail; bash code/run_idpo_gold_test_all.sh 2>&1 | tee \"$LOG\"; status=\${PIPESTATUS[0]}; echo PIPELINE_EXIT=\$status | tee -a \"$LOG\"; exit \$status'"

echo "session=$SESSION"
echo "log=$LOG"
