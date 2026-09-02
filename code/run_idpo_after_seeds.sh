#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CONFIG="${1:-$HERE/config_simple_trace_top8_idpo_first50.yaml}"
DATA_PYTHON="${DATA_PYTHON:-$ROOT/.venv/bin/python}"
TRAIN_PYTHON="${TRAIN_PYTHON:-$DATA_PYTHON}"

# Seed 已在本机构建并同步；AutoDL 只运行不依赖 Teacher API 的 GPU/本地阶段。
"$TRAIN_PYTHON" "$HERE/17_idpo_rollout.py" --config "$CONFIG" --split test
"$DATA_PYTHON" "$HERE/18_idpo_gold_score.py" --config "$CONFIG" --split test
"$DATA_PYTHON" "$HERE/19_build_idpo_pairs.py" --config "$CONFIG" --split test
"$TRAIN_PYTHON" "$HERE/20_train_user_editor_idpo.py" --config "$CONFIG" --split test
"$TRAIN_PYTHON" "$HERE/21_evaluate_user_editor_idpo.py" --config "$CONFIG" --split test
"$TRAIN_PYTHON" "$HERE/22_train_idpo_ranker_user_heads.py" --config "$CONFIG" --split test

echo "IDPO_AFTER_SEEDS_EXIT=0"
