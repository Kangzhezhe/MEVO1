#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DATA_PYTHON="${DATA_PYTHON:-$ROOT/.venv/bin/python}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/home/liux/miniconda3/envs/hydra/bin/python}"
WARMUP_CONFIG="$HERE/config_conditional_trace_warmup.yaml"
IDPO_CONFIG="$HERE/config_conditional_trace_idpo_first50_warmup.yaml"

echo "===== BALANCED TRACE WARM-UP DATA ====="
"$DATA_PYTHON" "$HERE/29_build_trace_warmup.py" --config "$WARMUP_CONFIG"

echo "===== BALANCED TRACE WARM-UP TRAIN ====="
"$TRAIN_PYTHON" "$HERE/06_train_editor_lora.py" --config "$WARMUP_CONFIG"

echo "===== NEW-USER TRACE ROLLOUT GATE ====="
"$TRAIN_PYTHON" "$HERE/30_validate_trace_rollout.py" \
  --config "$IDPO_CONFIG" --users 20 --samples 4 --minimum-trace-rate 0.10

echo "===== TRACE-AWARE PER-USER IDPO ====="
bash "$HERE/run_idpo_gold_test_all.sh" "$IDPO_CONFIG"

echo "TRACE_WARMUP_IDPO_EXIT=0"
