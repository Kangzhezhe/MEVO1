#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CONFIG="${1:-$HERE/config_idpo_pilot.yaml}"
SPLIT="${2:-validation}"
DATA_PYTHON="${DATA_PYTHON:-$ROOT/.venv/bin/python}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/home/liux/miniconda3/envs/hydra/bin/python}"

"$DATA_PYTHON" "$HERE/10_build_adaptation_queries.py" --config "$CONFIG" --split "$SPLIT"
"$DATA_PYTHON" "$HERE/02_retrieve.py" --config "$CONFIG" --split "adaptation_$SPLIT"
"$DATA_PYTHON" "$HERE/03_generate_seeds.py" --config "$CONFIG" --split "adaptation_$SPLIT"
"$TRAIN_PYTHON" "$HERE/17_idpo_rollout.py" --config "$CONFIG" --split "$SPLIT"
PREFERENCE_SOURCE="$($DATA_PYTHON -c 'import sys; sys.path.insert(0, sys.argv[1]); from pipeline_common import load_config; print(load_config(sys.argv[2])["idpo"].get("preference_source", "teacher_judge"))' "$HERE" "$CONFIG")"
if [[ "$PREFERENCE_SOURCE" == "loo_gold" ]]; then
  "$DATA_PYTHON" "$HERE/18_idpo_gold_score.py" --config "$CONFIG" --split "$SPLIT"
else
  "$DATA_PYTHON" "$HERE/18_idpo_teacher_judge.py" --config "$CONFIG" --split "$SPLIT"
fi
"$DATA_PYTHON" "$HERE/19_build_idpo_pairs.py" --config "$CONFIG" --split "$SPLIT"
"$TRAIN_PYTHON" "$HERE/20_train_user_editor_idpo.py" --config "$CONFIG" --split "$SPLIT"

echo "IDPO_PILOT_EXIT=0"
