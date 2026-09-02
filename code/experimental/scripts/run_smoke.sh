#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CONFIG="${1:-$HERE/config_smoke.yaml}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

for split in train validation; do
  "$PYTHON" "$HERE/01_prepare.py" --config "$CONFIG" --split "$split"
  "$PYTHON" "$HERE/02_retrieve.py" --config "$CONFIG" --split "$split"
  "$PYTHON" "$HERE/03_generate_seeds.py" --config "$CONFIG" --split "$split"
done
"$PYTHON" "$HERE/04_teacher_evolve.py" --config "$CONFIG" --split train
"$PYTHON" "$HERE/05_build_editor_sft.py" --config "$CONFIG"

# Mock Local Editor 继续验证十候选、Ranker 数据和用户历史留一的字段衔接。
for split in train validation; do
  "$PYTHON" "$HERE/07_generate_editor_pool.py" --config "$CONFIG" --split "$split"
done
"$PYTHON" "$HERE/08_build_scorer_data.py" --config "$CONFIG"
"$PYTHON" "$HERE/10_build_adaptation_queries.py" --config "$CONFIG" --split validation
"$PYTHON" "$HERE/02_retrieve.py" --config "$CONFIG" --split adaptation_validation
"$PYTHON" "$HERE/03_generate_seeds.py" --config "$CONFIG" --split adaptation_validation
"$PYTHON" "$HERE/07_generate_editor_pool.py" --config "$CONFIG" --split adaptation_validation
"$PYTHON" -m pytest -q "$HERE/test_factor_free_pipeline.py"

echo "Smoke 数据、候选、SFT、Ranker 数据和 per-user Leave-One-Out 契约验证完成。"
echo "LoRA 和 DeBERTa 训练由 run_pipeline.sh 在 GPU 上执行。"
