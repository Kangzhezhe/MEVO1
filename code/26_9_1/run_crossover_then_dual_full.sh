#!/usr/bin/env bash
# 正式顺序实验：Crossover-only -> Dual-operator。
# 数据/API 阶段和 GPU 训练阶段使用各自已验证的 Python 环境。
set -euo pipefail

ROOT="/home/liux/kk/MEVO_global_cot"
DATA_PYTHON="/home/liux/kk/MEVO/.venv/bin/python"
TRAIN_PYTHON="/home/liux/miniconda3/envs/hydra/bin/python"
STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"

CROSS_DATA="/data/liux/MEVO_global_cot/dataset/editor_sets/${STAMP}_crossover_only_sft"
CROSS_RUN="/data/liux/MEVO_global_cot/result/${STAMP}_crossover_only_sft"
DUAL_DATA="/data/liux/MEVO_global_cot/dataset/editor_sets/${STAMP}_dual_operator_sft"
DUAL_RUN="/data/liux/MEVO_global_cot/result/${STAMP}_dual_operator_sft"
LOG_DIR="/data/liux/MEVO_global_cot/logs"
LOG_FILE="${LOG_DIR}/${STAMP}_crossover_then_dual_full.log"
PARENT_RECORDS="/data/liux/MEVO_global_cot/dataset/editor_sets/20260902_094902_direct_parent_gold_full/all_sft.jsonl"

mkdir -p "$CROSS_DATA" "$CROSS_RUN" "$DUAL_DATA" "$DUAL_RUN" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
cd "$ROOT"

echo "RUN_STAMP=$STAMP"
echo "CROSS_DATA=$CROSS_DATA"
echo "CROSS_RUN=$CROSS_RUN"
echo "DUAL_DATA=$DUAL_DATA"
echo "DUAL_RUN=$DUAL_RUN"
echo "LOG_FILE=$LOG_FILE"

echo "===== 1/7 Crossover Parent Pool: real Llama2-7B ====="
PYTHONDONTWRITEBYTECODE=1 "$TRAIN_PYTHON" code/26_9_1/run_crossover_sft.py \
  --config code/26_9_1/config_crossover_sft.yaml \
  --stage pool --data-dir "$CROSS_DATA" --run-dir "$CROSS_RUN" \
  --parent-records "$PARENT_RECORDS" \
  --pool-source base_model --teacher-mode api --extra-parents 3

echo "===== 2/7 Crossover Pair gate: VisGPT ====="
PYTHONDONTWRITEBYTECODE=1 "$DATA_PYTHON" code/26_9_1/run_crossover_sft.py \
  --config code/26_9_1/config_crossover_sft.yaml \
  --stage pairs --data-dir "$CROSS_DATA" --run-dir "$CROSS_RUN" \
  --pool-source base_model --teacher-mode api --extra-parents 3

echo "===== 3/7 Crossover-only SFT ====="
PYTHONDONTWRITEBYTECODE=1 "$TRAIN_PYTHON" code/26_9_1/run_crossover_sft.py \
  --config code/26_9_1/config_crossover_sft.yaml \
  --stage train --data-dir "$CROSS_DATA" --run-dir "$CROSS_RUN"

echo "===== 4/7 Crossover-only standard test100 evaluation ====="
PYTHONDONTWRITEBYTECODE=1 "$TRAIN_PYTHON" code/26_9_1/run_crossover_sft.py \
  --config code/26_9_1/config_crossover_sft.yaml \
  --stage eval --data-dir "$CROSS_DATA" --run-dir "$CROSS_RUN" \
  --pool-source base_model --teacher-mode api --extra-parents 3

echo "===== 5/7 Build Dual-operator SFT data ====="
PYTHONDONTWRITEBYTECODE=1 "$DATA_PYTHON" code/26_9_1/run_dual_operator_sft.py \
  --config code/26_9_1/config_dual_operator_sft.yaml \
  --stage build --crossover-data-dir "$CROSS_DATA" \
  --data-dir "$DUAL_DATA" --run-dir "$DUAL_RUN"

echo "===== 6/7 Dual-operator SFT ====="
PYTHONDONTWRITEBYTECODE=1 "$TRAIN_PYTHON" code/26_9_1/run_dual_operator_sft.py \
  --config code/26_9_1/config_dual_operator_sft.yaml \
  --stage train --crossover-data-dir "$CROSS_DATA" \
  --data-dir "$DUAL_DATA" --run-dir "$DUAL_RUN"

echo "===== 7/7 Dual-operator standard test100 evaluation ====="
PYTHONDONTWRITEBYTECODE=1 "$TRAIN_PYTHON" code/26_9_1/run_dual_operator_sft.py \
  --config code/26_9_1/config_dual_operator_sft.yaml \
  --stage eval --crossover-data-dir "$CROSS_DATA" \
  --data-dir "$DUAL_DATA" --run-dir "$DUAL_RUN"

echo "CROSSOVER_THEN_DUAL_FULL_EXIT=0"

