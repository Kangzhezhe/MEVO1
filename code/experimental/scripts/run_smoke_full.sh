#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CONFIG="${1:-$HERE/config_smoke_full.yaml}"
DATA_PYTHON="${DATA_PYTHON:-$ROOT/.venv/bin/python}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/home/liux/miniconda3/envs/hydra/bin/python}"

if [[ ! -x "$DATA_PYTHON" || ! -x "$TRAIN_PYTHON" ]]; then
  echo "缺少 DATA_PYTHON 或 TRAIN_PYTHON" >&2
  exit 1
fi

run_data() {
  echo "===== FULL SMOKE: $* ====="
  "$DATA_PYTHON" "$@"
}

run_train() {
  echo "===== FULL SMOKE: $* ====="
  "$TRAIN_PYTHON" "$@"
}

for split in train validation test; do
  run_data "$HERE/01_prepare.py" --config "$CONFIG" --split "$split"
  run_data "$HERE/02_retrieve.py" --config "$CONFIG" --split "$split"
  run_data "$HERE/03_generate_seeds.py" --config "$CONFIG" --split "$split"
done

run_data "$HERE/04_teacher_evolve.py" --config "$CONFIG" --split train
run_data "$HERE/05_build_editor_sft.py" --config "$CONFIG"
run_train "$HERE/06_train_editor_lora.py" --config "$CONFIG" --max-steps 1
for split in train validation test; do
  run_train "$HERE/07_generate_editor_pool.py" --config "$CONFIG" --split "$split"
done
run_data "$HERE/08_build_scorer_data.py" --config "$CONFIG"
run_train "$HERE/09_train_scorer.py" --config "$CONFIG"

for split in validation test; do
  run_data "$HERE/10_build_adaptation_queries.py" --config "$CONFIG" --split "$split"
  run_data "$HERE/02_retrieve.py" --config "$CONFIG" --split "adaptation_$split"
  run_data "$HERE/03_generate_seeds.py" --config "$CONFIG" --split "adaptation_$split"
  run_train "$HERE/07_generate_editor_pool.py" --config "$CONFIG" --split "adaptation_$split"
  run_train "$HERE/11_adapt_user_scorer.py" --config "$CONFIG" --split "$split"
  run_train "$HERE/12_evaluate.py" --config "$CONFIG" --split "$split"
done
run_data -m pytest -q "$HERE/test_factor_free_pipeline.py"

echo "FULL_SMOKE_EXIT=0"
