#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CONFIG="${1:-$ROOT/config_global.yaml}"
DATA_PYTHON="${DATA_PYTHON:-$ROOT/../MEVO/.venv/bin/python}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/home/liux/miniconda3/envs/hydra/bin/python}"

# 正式输出必须位于以时间戳开头的实验包。固定旧路径容易覆盖已有结果，也无法
# 按目录名判断新旧；新实验应通过 run_timestamped_global.sh 启动。
"$DATA_PYTHON" - "$HERE" "$CONFIG" <<'PY'
import re
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from pipeline_common import load_config, resolve_path

config = load_config(sys.argv[2])
result_root = resolve_path("result").resolve()
for key in ("editor_output_dir", "prediction_dir", "reports_dir"):
    output = resolve_path(config["paths"][key]).resolve()
    try:
        relative = output.relative_to(result_root)
    except ValueError as error:
        raise SystemExit(f"{key} 必须位于 {result_root}: {output}") from error
    if not relative.parts or not re.fullmatch(r"\d{8}_\d{6}_.+", relative.parts[0]):
        raise SystemExit(
            f"{key} 未使用时间戳实验目录: {output}\n"
            "请改用 code/run_timestamped_global.sh <config> 启动。"
        )
PY

run_data() { "$DATA_PYTHON" "$@"; }
run_train() { "$TRAIN_PYTHON" "$@"; }
run_teacher() {
  local attempt=1 max_attempts="${TEACHER_STAGE_ATTEMPTS:-30}"
  while true; do
    if run_data "$@"; then return 0; fi
    if (( attempt >= max_attempts )); then return 1; fi
    echo "Teacher stage failed ($attempt/$max_attempts); cache will resume" >&2
    attempt=$((attempt + 1)); sleep "${TEACHER_RETRY_SECONDS:-30}"
  done
}

for split in train validation test; do
  run_data "$HERE/01_prepare.py" --config "$CONFIG" --split "$split"
  run_data "$HERE/02_retrieve.py" --config "$CONFIG" --split "$split"
  run_teacher "$HERE/03_generate_seeds.py" --config "$CONFIG" --split "$split"
done
SUPERVISION_MODE="$($DATA_PYTHON -c 'import sys;sys.path.insert(0,sys.argv[1]);from pipeline_common import load_config;print(load_config(sys.argv[2]).get("sft_data",{}).get("supervision_mode","gold_aware_trace"))' "$HERE" "$CONFIG")"
if [ "$SUPERVISION_MODE" = "output_only" ] || [ "$SUPERVISION_MODE" = "plain_output_only" ]; then
  echo "$SUPERVISION_MODE ablation: skip Teacher Trace construction."
else
  SEED_SOURCE="$("$DATA_PYTHON" -c 'import sys;sys.path.insert(0,sys.argv[1]);from pipeline_common import load_config,stage_path;print(stage_path(load_config(sys.argv[2]),"train","seeds"))' "$HERE" "$CONFIG")"
  TRACE_OUTPUT="$("$DATA_PYTHON" -c 'import sys;sys.path.insert(0,sys.argv[1]);from pipeline_common import load_config,resolve_path;print(resolve_path(load_config(sys.argv[2])["paths"]["conditional_trace_dir"]))' "$HERE" "$CONFIG")"
  run_teacher "$HERE/31_build_simple_conditional_traces.py" --config "$CONFIG" \
    --count 0 --source "$SEED_SOURCE" --output-dir "$TRACE_OUTPUT" \
    --concurrency 4 --checkpoint-every 20
fi
run_data "$HERE/05_build_editor_sft.py" --config "$CONFIG"
if "$DATA_PYTHON" -c 'import sys;sys.path.insert(0,sys.argv[1]);from pipeline_common import load_config;raise SystemExit(0 if not load_config(sys.argv[2])["pipeline"].get("run_gpu",True) else 1)' "$HERE" "$CONFIG"; then
  echo "GPU stages disabled; SFT data is ready."
  exit 0
fi
EDITOR_ADAPTER="$($DATA_PYTHON -c 'import sys;sys.path.insert(0,sys.argv[1]);from pipeline_common import load_config,resolve_path; c=load_config(sys.argv[2]); print(resolve_path(c["paths"]["editor_output_dir"])/"final_adapter")' "$HERE" "$CONFIG")"
if [ -f "$EDITOR_ADAPTER/adapter_config.json" ]; then
  echo "Shared SFT adapter exists; skipping completed SFT: $EDITOR_ADAPTER"
else
  run_train "$HERE/06_train_editor_lora.py" --config "$CONFIG"
fi

# PriME-matched mode uses each original training Query with its complete
# support history.  It intentionally avoids expanding every history item into
# a separate Leave-One-Out query; the existing train prepare/retrieve/seeds are
# target-blind and can be reused directly by the global IDPO stages.
if [ "$($DATA_PYTHON -c 'import sys;sys.path.insert(0,sys.argv[1]);from pipeline_common import load_config; c=load_config(sys.argv[2]); print(c.get("global_idpo",{}).get("source_split", "loo"))' "$HERE" "$CONFIG")" = "train" ]; then
  echo "Global IDPO source=train; reusing original training support set (no LOO expansion)."
else
  run_data "$HERE/10_build_global_loo.py" --config "$CONFIG"
  run_data "$HERE/02_retrieve.py" --config "$CONFIG" --split adaptation_train
  run_teacher "$HERE/03_generate_seeds.py" --config "$CONFIG" --split adaptation_train
fi
run_train "$HERE/24_global_idpo_rollout.py" --config "$CONFIG"
run_data "$HERE/25_score_global_rollouts.py" --config "$CONFIG"
run_data "$HERE/26_build_global_pairs.py" --config "$CONFIG"
run_train "$HERE/27_train_global_idpo.py" --config "$CONFIG"
PROMPT_PROTOCOL="$($DATA_PYTHON -c 'import sys;sys.path.insert(0,sys.argv[1]);from pipeline_common import load_config;print(load_config(sys.argv[2]).get("sft_data",{}).get("prompt_protocol",""))' "$HERE" "$CONFIG")"
if [ "$PROMPT_PROTOCOL" = "base_text" ]; then
  # Base-equivalent evaluation: same plain-text prompt, deterministic decoding,
  # first non-empty title line cleaning, and the complete standard test100
  # (100 users / 608 queries).  Evaluate both checkpoints separately so the
  # output-only ablation is directly comparable with Base and Trace rows.
  SFT_ADAPTER="$EDITOR_ADAPTER"
  IDPO_ADAPTER="$(dirname "$EDITOR_ADAPTER")/global_idpo/final_adapter"
  run_train "$HERE/30_generate_base_predictions.py" --config "$CONFIG" \
    --adapter "$SFT_ADAPTER" --output-name sft_text
  run_data "$HERE/29_evaluate_global.py" --config "$CONFIG" \
    --prediction-subdir sft_text --report-subdir sft_text
  run_train "$HERE/30_generate_base_predictions.py" --config "$CONFIG" \
    --adapter "$IDPO_ADAPTER" --output-name idpo_text
  run_data "$HERE/29_evaluate_global.py" --config "$CONFIG" \
    --prediction-subdir idpo_text --report-subdir idpo_text
else
  run_train "$HERE/28_generate_global_predictions.py" --config "$CONFIG"
  run_data "$HERE/29_evaluate_global.py" --config "$CONFIG"
fi
echo "GLOBAL_MEVO_PIPELINE_EXIT=0"
