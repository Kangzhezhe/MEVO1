#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CONFIG="${1:-$HERE/config_idpo.yaml}"
DATA_PYTHON="${DATA_PYTHON:-$ROOT/.venv/bin/python}"
TRAIN_PYTHON="${TRAIN_PYTHON:-$DATA_PYTHON}"

# Stage 17--19 已完成；从 per-user Editor DPO 开始统一重训所有用户。
"$TRAIN_PYTHON" "$HERE/20_train_user_editor_idpo.py" --config "$CONFIG" --split test
"$TRAIN_PYTHON" "$HERE/21_evaluate_user_editor_idpo.py" --config "$CONFIG" --split test
"$TRAIN_PYTHON" "$HERE/22_train_idpo_ranker_user_heads.py" --config "$CONFIG" --split test

echo "IDPO_AFTER_PAIRS_EXIT=0"
