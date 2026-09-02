#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CONFIG="${1:-$HERE/config.yaml}"
SESSION="${2:-mevo_gold_aware_sft}"
LOG_GROUP="${3:-20260731_gold_aware_sft}"
LOG_DIR="$ROOT/logs/$LOG_GROUP"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/${SESSION}_${STAMP}.log"
mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session 已存在: $SESSION" >&2
  exit 1
fi
# tmux server 可能早于当前 shell 启动，不会自动继承后来 export 的 Teacher Key。
# 这里只同步环境变量值，不把密钥写进配置、日志或 pane 启动命令。
for variable in VIS_API_KEY QWEN_API_KEY; do
  value="${!variable-}"
  if [[ -n "$value" ]]; then
    tmux set-environment -g "$variable" "$value"
  fi
done
unset value variable
tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && bash '$HERE/run_pipeline.sh' '$CONFIG' 2>&1 | tee '$LOG'; status=\${PIPESTATUS[0]}; echo PIPELINE_EXIT=\$status; exec bash"
echo "session=$SESSION"
echo "log=$LOG"
echo "查看: tmux attach -t $SESSION"
