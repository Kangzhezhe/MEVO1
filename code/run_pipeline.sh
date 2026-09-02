#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CONFIG="${1:-$HERE/config_simple_trace_top8_full.yaml}"
# Per-Pcs 流读取依赖 ijson，GPU 阶段依赖 PyTorch；当前主机分别安装在两个环境。
DATA_PYTHON="${DATA_PYTHON:-$ROOT/.venv/bin/python}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/home/liux/miniconda3/envs/hydra/bin/python}"
if [[ ! -x "$DATA_PYTHON" || ! -x "$TRAIN_PYTHON" ]]; then
  echo "缺少 DATA_PYTHON 或 TRAIN_PYTHON，请显式指定两个环境" >&2
  exit 1
fi

run_data() {
  echo "===== $* ====="
  "$DATA_PYTHON" "$@"
}

run_train() {
  echo "===== $* ====="
  "$TRAIN_PYTHON" "$@"
}

run_teacher() {
  local maximum_attempts="${TEACHER_STAGE_ATTEMPTS:-30}"
  local retry_seconds="${TEACHER_RETRY_SECONDS:-30}"
  local attempt=1
  while true; do
    if run_data "$@"; then
      return 0
    fi
    if (( attempt >= maximum_attempts )); then
      echo "Teacher stage failed after $attempt attempts: $*" >&2
      return 1
    fi
    echo "Teacher stage failed (attempt $attempt/$maximum_attempts); cached requests will resume in ${retry_seconds}s" >&2
    attempt=$((attempt + 1))
    sleep "$retry_seconds"
  done
}

config_value() {
  "$DATA_PYTHON" -c 'import sys; from functools import reduce; sys.path.insert(0, sys.argv[1]); from pipeline_common import load_config; value = reduce(lambda current, key: current[key], sys.argv[3].split("."), load_config(sys.argv[2])); print(str(value).lower() if isinstance(value, bool) else value)' "$HERE" "$CONFIG" "$1"
}

SUPERVISION_MODE="$(config_value sft_data.supervision_mode)"
RUN_USER_ADAPTATION="$(config_value pipeline.run_user_adaptation)"
RUN_GPU="$(config_value pipeline.run_gpu)"

# Per-Pcs 三个正式划分只做格式归一化和 BM25 检索。
for split in train validation test; do
  run_data "$HERE/01_prepare.py" --config "$CONFIG" --split "$split"
  run_data "$HERE/02_retrieve.py" --config "$CONFIG" --split "$split"
  run_teacher "$HERE/03_generate_seeds.py" --config "$CONFIG" --split "$split"
done

# 根目录只保留当前最佳的 Top-8 简化 Trace 正式路线。其他历史监督定义仍可在
# 日期归档目录中复现，避免主入口继续携带大量已经淘汰的分支。
if [[ "$SUPERVISION_MODE" != "simple_conditional_trace" ]]; then
  echo "根目录正式流程仅支持 sft_data.supervision_mode=simple_conditional_trace" >&2
  exit 2
fi
TRACE_SOURCE="$(config_value paths.candidate_root)/$(config_value splits.train.processed_split)/03_seeds.jsonl"
TRACE_OUTPUT="$(config_value paths.conditional_trace_dir)"
TRACE_CONCURRENCY="$(config_value simple_conditional_trace.concurrency)"
TRACE_CHECKPOINT_EVERY="$(config_value simple_conditional_trace.checkpoint_every)"
run_teacher "$HERE/31_build_simple_conditional_traces.py" \
  --config "$CONFIG" --count 0 --source "$TRACE_SOURCE" \
  --output-dir "$TRACE_OUTPUT" --concurrency "$TRACE_CONCURRENCY" \
  --checkpoint-every "$TRACE_CHECKPOINT_EVERY"
run_data "$HERE/05_build_editor_sft.py" --config "$CONFIG"
if [[ "$RUN_GPU" != "true" ]]; then
  echo "GPU stages paused by pipeline.run_gpu=false; SFT data is ready."
  exit 0
fi
if [[ "$(config_value pipeline.wait_for_gpu)" == "true" ]]; then
  GPU_THRESHOLD="$(config_value pipeline.gpu_memory_threshold_mib)"
  while true; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used < GPU_THRESHOLD )); then
      break
    fi
    echo "GPU busy: ${used:-unknown} MiB used; waiting before shared SFT"
    sleep 60
  done
fi
run_train "$HERE/06_train_editor_lora.py" --config "$CONFIG"
# Ranker 的三个 split 都由本地 Editor target-blind 地重新生成。
for split in train validation test; do
  run_train "$HERE/07_generate_editor_pool.py" --config "$CONFIG" --split "$split"
done

run_data "$HERE/08_build_scorer_data.py" --config "$CONFIG"
run_train "$HERE/09_train_scorer.py" --config "$CONFIG"

# 先报告共享 Ranker，保证阶段一的比较不混入 per-user Head。
for split in validation test; do
  run_data "$HERE/12_evaluate.py" --config "$CONFIG" --split "$split"
done

# Per-user Editor 与 Ranker Head 统一由阶段二执行，阶段一不再保留旧适配分支。
if [[ "$RUN_USER_ADAPTATION" == "true" ]]; then
  echo "根目录正式流程要求 pipeline.run_user_adaptation=false；请运行阶段二 IDPO" >&2
  exit 2
fi
