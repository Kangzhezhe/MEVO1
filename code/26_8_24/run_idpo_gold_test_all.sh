#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CONFIG="${1:-$HERE/config_idpo.yaml}"
DATA_PYTHON="${DATA_PYTHON:-$ROOT/.venv/bin/python}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/home/liux/miniconda3/envs/hydra/bin/python}"

run_teacher() {
  local maximum_attempts="${TEACHER_STAGE_ATTEMPTS:-30}"
  local retry_seconds="${TEACHER_RETRY_SECONDS:-30}"
  local attempt=1
  while true; do
    if "$DATA_PYTHON" "$@"; then
      return 0
    fi
    if (( attempt >= maximum_attempts )); then
      echo "Teacher stage failed after $attempt attempts: $*" >&2
      return 1
    fi
    echo "Teacher stage failed (attempt $attempt/$maximum_attempts); cached seeds will resume in ${retry_seconds}s" >&2
    attempt=$((attempt + 1))
    sleep "$retry_seconds"
  done
}

config_value() {
  "$DATA_PYTHON" -c 'import sys; from functools import reduce; sys.path.insert(0, sys.argv[1]); from pipeline_common import load_config; print(reduce(lambda x, k: x[k], sys.argv[3].split("."), load_config(sys.argv[2])))' "$HERE" "$CONFIG" "$1"
}

run_gpu_stage() {
  local maximum_attempts="${GPU_STAGE_ATTEMPTS:-10}"
  local retry_seconds="${GPU_RETRY_SECONDS:-30}"
  local attempt=1
  while true; do
    if "$TRAIN_PYTHON" "$@"; then
      return 0
    fi
    if (( attempt >= maximum_attempts )); then
      echo "GPU stage failed after $attempt attempts: $*" >&2
      return 1
    fi
    echo "GPU stage failed (attempt $attempt/$maximum_attempts); checkpoint will resume in ${retry_seconds}s" >&2
    attempt=$((attempt + 1))
    sleep "$retry_seconds"
  done
}

if [[ "${SKIP_IDPO_PREPARE:-0}" != "1" ]]; then
  "$DATA_PYTHON" "$HERE/10_build_adaptation_queries.py" --config "$CONFIG" --split test
  "$DATA_PYTHON" "$HERE/02_retrieve.py" --config "$CONFIG" --split adaptation_test
else
  echo "Skipping completed LOO prepare/retrieve; resuming from cached Seed checkpoint."
fi
run_teacher "$HERE/03_generate_seeds.py" --config "$CONFIG" --split adaptation_test

# API Parent 构造不占 GPU，可以与其他用户任务并行。进入本地 Qwen rollout
# 前等待显存基本释放，避免抢占他人进程或直接 OOM。
GPU_THRESHOLD="$(config_value pipeline.gpu_memory_threshold_mib)"
while true; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if [[ "$used" =~ ^[0-9]+$ ]] && (( used < GPU_THRESHOLD )); then
    break
  fi
  echo "GPU busy: ${used:-unknown} MiB used; waiting for threshold ${GPU_THRESHOLD} MiB before IDPO rollout"
  sleep 60
done

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
run_gpu_stage "$HERE/17_idpo_rollout.py" --config "$CONFIG" --split test
"$DATA_PYTHON" "$HERE/18_idpo_gold_score.py" --config "$CONFIG" --split test
"$DATA_PYTHON" "$HERE/19_build_idpo_pairs.py" --config "$CONFIG" --split test
run_gpu_stage "$HERE/20_train_user_editor_idpo.py" --config "$CONFIG" --split test
run_gpu_stage "$HERE/21_evaluate_user_editor_idpo.py" --config "$CONFIG" --split test
run_gpu_stage "$HERE/22_train_idpo_ranker_user_heads.py" --config "$CONFIG" --split test

echo "IDPO_TEST_ALL_WITH_RANKER_EXIT=0"
