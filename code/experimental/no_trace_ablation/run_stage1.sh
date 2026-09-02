#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PIPE="$ROOT/code/26_8_24"
CONFIG="${1:-$HERE/config.yaml}"
DATA_PYTHON="${DATA_PYTHON:-$ROOT/.venv/bin/python}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/home/liux/miniconda3/envs/hydra/bin/python}"

run_data() { echo "===== $* ====="; "$DATA_PYTHON" "$@"; }
run_train() { echo "===== $* ====="; "$TRAIN_PYTHON" "$@"; }

run_teacher() {
  local maximum_attempts="${TEACHER_STAGE_ATTEMPTS:-30}"
  local retry_seconds="${TEACHER_RETRY_SECONDS:-30}"
  local attempt=1
  while true; do
    if run_data "$@"; then return 0; fi
    if (( attempt >= maximum_attempts )); then return 1; fi
    echo "Teacher stage failed ($attempt/$maximum_attempts); retry in ${retry_seconds}s" >&2
    attempt=$((attempt + 1))
    sleep "$retry_seconds"
  done
}

config_value() {
  "$DATA_PYTHON" -c 'import sys; from functools import reduce; sys.path.insert(0, sys.argv[1]); from pipeline_common import load_config; value=reduce(lambda x,k:x[k],sys.argv[3].split("."),load_config(sys.argv[2])); print(str(value).lower() if isinstance(value,bool) else value)' "$PIPE" "$CONFIG" "$1"
}

if [[ "$(config_value sft_data.supervision_mode)" != "output_only" ]]; then
  echo "No-Trace消融要求 sft_data.supervision_mode=output_only" >&2
  exit 2
fi

for split in train validation test; do
  run_data "$PIPE/01_prepare.py" --config "$CONFIG" --split "$split"
  run_data "$PIPE/02_retrieve.py" --config "$CONFIG" --split "$split"
  run_teacher "$PIPE/03_generate_seeds.py" --config "$CONFIG" --split "$split"
done

# No-Trace核心：不执行Stage 31，直接用Parent -> Gold构造output-only SFT。
run_data "$PIPE/05_build_editor_sft.py" --config "$CONFIG"

if [[ "$(config_value pipeline.run_gpu)" != "true" ]]; then
  echo "NO_TRACE_STAGE1_DATA_EXIT=0"
  exit 0
fi

if [[ "$(config_value pipeline.wait_for_gpu)" == "true" ]]; then
  threshold="$(config_value pipeline.gpu_memory_threshold_mib)"
  while true; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used < threshold )); then break; fi
    echo "GPU busy: ${used:-unknown} MiB; waiting for threshold $threshold MiB"
    sleep 60
  done
fi

run_train "$PIPE/06_train_editor_lora.py" --config "$CONFIG"
for split in train validation test; do
  run_train "$PIPE/07_generate_editor_pool.py" --config "$CONFIG" --split "$split"
done
run_data "$PIPE/08_build_scorer_data.py" --config "$CONFIG"
run_train "$PIPE/09_train_scorer.py" --config "$CONFIG"
for split in validation test; do
  run_data "$PIPE/12_evaluate.py" --config "$CONFIG" --split "$split"
done

echo "NO_TRACE_STAGE1_EXIT=0"
