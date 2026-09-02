#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
TRAIN_SESSION="mevo_idpo_gold30_m5"
SESSION="mevo_idpo_gold30_m5_eval"
LOG_DIR="$ROOT/logs/20260802_idpo_gold30_m5"
LOG="$LOG_DIR/mevo_idpo_gold30_m5_eval_$(date +%Y%m%d_%H%M%S).log"
TRAIN_PYTHON="${TRAIN_PYTHON:-/home/liux/miniconda3/envs/hydra/bin/python}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session 已存在: $SESSION" >&2
  exit 1
fi
mkdir -p "$LOG_DIR"

tmux new-session -d -s "$SESSION" -c "$ROOT" \
  "bash -lc 'while tmux has-session -t \"=$TRAIN_SESSION\" 2>/dev/null; do sleep 30; done; set -o pipefail; \"$TRAIN_PYTHON\" code/21_evaluate_user_editor_idpo.py --config code/config_idpo_gold30.yaml --split validation 2>&1 | tee \"$LOG\"; status=\${PIPESTATUS[0]}; echo EVAL_EXIT=\$status | tee -a \"$LOG\"; exit \$status'"

echo "session=$SESSION"
echo "log=$LOG"
