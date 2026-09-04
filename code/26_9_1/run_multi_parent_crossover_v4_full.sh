#!/usr/bin/env bash
set -euo pipefail

# 无 Teacher 的多 Parent Crossover 正式实验：
# 1) Base 为每个 Query 生成 1 greedy + 4 sampling Parents；
# 2) 用所有 Query 构造 Parent 集合 -> Gold 的纯文本 SFT；
# 3) 训练 Llama2-7B LoRA；
# 4) 在标准 test100（100 用户、608 Query）上生成同分布 Parents 并评估。

ROOT=/home/liux/kk/MEVO_global_cot
STAMP=${1:-$(date +%Y%m%d_%H%M%S)}
NAME=${STAMP}_crossover_greedy1_sample4_teacher_free_v4_4096
DATA=/data/liux/MEVO_global_cot/dataset/editor_sets/${NAME}
RUN=/data/liux/MEVO_global_cot/result/${NAME}
LOG=/data/liux/MEVO_global_cot/logs/${NAME}.log
CONFIG=code/26_9_1/config_crossover_sft.yaml
SCRIPT=code/26_9_1/run_multi_parent_crossover_sft.py

cd "$ROOT"
mkdir -p "$DATA" "$RUN" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "===== 1/4 Generate train Parents: 1 greedy + 4 sampling ====="
/home/liux/miniconda3/envs/hydra/bin/python "$SCRIPT" \
  --config "$CONFIG" --stage pool --data-dir "$DATA" --run-dir "$RUN"

echo "===== 2/4 Compile Teacher-free multi-parent -> Gold SFT ====="
/home/liux/miniconda3/envs/hydra/bin/python "$SCRIPT" \
  --config "$CONFIG" --stage build --data-dir "$DATA" --run-dir "$RUN"

echo "===== 3/4 Train Crossover aggregation Llama2-7B LoRA ====="
/home/liux/miniconda3/envs/hydra/bin/python "$SCRIPT" \
  --config "$CONFIG" --stage train --data-dir "$DATA" --run-dir "$RUN"

echo "===== 4/4 Standard test100/608 evaluation ====="
/home/liux/miniconda3/envs/hydra/bin/python "$SCRIPT" \
  --config "$CONFIG" --stage eval --data-dir "$DATA" --run-dir "$RUN"

echo "MULTI_PARENT_CROSSOVER_V4_EXIT=0"
