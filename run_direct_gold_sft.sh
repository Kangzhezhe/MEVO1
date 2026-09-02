#!/usr/bin/env bash
# 只运行“Base 输入 -> Gold 输出”的直接 SFT 对照，不进入 IDPO。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$HERE"
CONFIG="${1:-$HERE/config_direct_gold_sft_base_protocol.yaml}"
DATA_PYTHON="${DATA_PYTHON:-$ROOT/MEVO/.venv/bin/python}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/home/liux/miniconda3/envs/hydra/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
TEMP_CONFIG="$(mktemp "/tmp/mevo_direct_gold_${STAMP}_XXXXXX.yaml")"

RUN_ID="$($DATA_PYTHON "$HERE/code/create_timestamped_run_config.py" \
  --config "$CONFIG" --timestamp "$STAMP" --destination "$TEMP_CONFIG")"
RUN_DIR="$HERE/result/$RUN_ID"
mkdir -p "$RUN_DIR"
RUNTIME_CONFIG="$RUN_DIR/run_config.yaml"
mv "$TEMP_CONFIG" "$RUNTIME_CONFIG"
LOG_FILE="$HERE/logs/${RUN_ID}.log"

run_data() { echo "===== $* ====="; "$DATA_PYTHON" "$@"; }
run_train() { echo "===== $* ====="; "$TRAIN_PYTHON" "$@"; }

echo "RUN_ID=$RUN_ID"
echo "CONFIG=$RUNTIME_CONFIG"
echo "LOG=$LOG_FILE"

{
  # 准备/检索/Seed 阶段只会读取已有的标准 Per-Pcs 缓存；若缓存完整，不会调用 API。
  for split in train validation test; do
    run_data "$HERE/code/01_prepare.py" --config "$RUNTIME_CONFIG" --split "$split"
    run_data "$HERE/code/02_retrieve.py" --config "$RUNTIME_CONFIG" --split "$split"
    run_data "$HERE/code/03_generate_seeds.py" --config "$RUNTIME_CONFIG" --split "$split"
  done

  # 核心对照：每个 Query 的第一个 Base Parent -> 精确 Gold，只有 output token 计 loss。
  run_data "$HERE/code/05_build_editor_sft.py" --config "$RUNTIME_CONFIG"
  run_train "$HERE/code/06_train_editor_lora.py" --config "$RUNTIME_CONFIG"

  ADAPTER="$RUN_DIR/editor/final_adapter"
  run_train "$HERE/code/30_generate_base_predictions.py" --config "$RUNTIME_CONFIG" \
    --adapter "$ADAPTER" --output-name sft_text
  run_data "$HERE/code/29_evaluate_global.py" --config "$RUNTIME_CONFIG" \
    --prediction-subdir sft_text --report-subdir sft_text
  echo "DIRECT_GOLD_SFT_PIPELINE_EXIT=0"
} 2>&1 | tee "$LOG_FILE"
