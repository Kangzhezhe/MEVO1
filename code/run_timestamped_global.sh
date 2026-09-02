#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CONFIG="${1:-$ROOT/config_global.yaml}"
DATA_PYTHON="${DATA_PYTHON:-$ROOT/../MEVO/.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
TEMP_CONFIG="$(mktemp "/tmp/mevo_global_${STAMP}_XXXXXX.yaml")"

RUN_ID="$($DATA_PYTHON "$HERE/create_timestamped_run_config.py" \
  --config "$CONFIG" \
  --timestamp "$STAMP" \
  --destination "$TEMP_CONFIG")"
RUN_DIR="$ROOT/result/$RUN_ID"
mkdir -p "$RUN_DIR"
RUNTIME_CONFIG="$RUN_DIR/run_config.yaml"
mv "$TEMP_CONFIG" "$RUNTIME_CONFIG"
LOG_FILE="$ROOT/logs/${RUN_ID}.log"

echo "RUN_ID=$RUN_ID"
echo "CONFIG=$RUNTIME_CONFIG"
echo "LOG=$LOG_FILE"
set -o pipefail
"$HERE/run_global_pipeline.sh" "$RUNTIME_CONFIG" 2>&1 | tee "$LOG_FILE"
