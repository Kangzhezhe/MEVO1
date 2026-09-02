#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CONFIG="${1:?usage: run_editor_to_ranker.sh CONFIG}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/home/liux/miniconda3/envs/hydra/bin/python}"
DATA_PYTHON="${DATA_PYTHON:-$ROOT/.venv/bin/python}"

for split in train validation test; do
  echo "===== Editor pool split=$split ====="
  "$TRAIN_PYTHON" "$HERE/07_generate_editor_pool.py" --config "$CONFIG" --split "$split"
done
echo "===== Build Scorer data ====="
"$DATA_PYTHON" "$HERE/08_build_scorer_data.py" --config "$CONFIG"
echo "===== Train Scorer ====="
"$TRAIN_PYTHON" "$HERE/09_train_scorer.py" --config "$CONFIG"
for split in validation test; do
  echo "===== Evaluate split=$split ====="
  "$DATA_PYTHON" "$HERE/12_evaluate.py" --config "$CONFIG" --split "$split"
done
echo "EDITOR_TO_RANKER_EXIT=0"
