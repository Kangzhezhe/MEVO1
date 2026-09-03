#!/usr/bin/env bash
set -euo pipefail

# Crossover 单任务正式实验：复用已经生成的真实 Base Parent Pool，重新执行
# strict Teacher gate、4096-token LoRA SFT 和标准 test100/608 Query 评估。

ROOT=/home/liux/kk/MEVO_global_cot
STAMP=${1:-$(date +%Y%m%d_%H%M%S)}
NAME=${STAMP}_crossover_single_strict_gate_v2_4096
SOURCE_DATA=/data/liux/MEVO_global_cot/dataset/editor_sets/20260902_225316_crossover_only_sft
DATA=/data/liux/MEVO_global_cot/dataset/editor_sets/${NAME}
RUN=/data/liux/MEVO_global_cot/result/${NAME}
LOG=/data/liux/MEVO_global_cot/logs/${NAME}.log
CONFIG=code/26_9_1/config_crossover_sft.yaml

mkdir -p "$DATA" "$RUN" "$(dirname "$LOG")"
if [[ ! -e "$DATA/01_parent_pool.jsonl" ]]; then
  ln -s "$SOURCE_DATA/01_parent_pool.jsonl" "$DATA/01_parent_pool.jsonl"
fi
if [[ ! -e "$DATA/test_parent_pool.jsonl" ]]; then
  ln -s "$SOURCE_DATA/test_parent_pool.jsonl" "$DATA/test_parent_pool.jsonl"
fi

exec > >(tee -a "$LOG") 2>&1

echo "===== 1/3 Strict Teacher gate + single-task SFT data ====="
/home/liux/kk/MEVO/.venv/bin/python code/26_9_1/run_crossover_sft.py \
  --config "$CONFIG" --stage pairs \
  --data-dir "$DATA" --run-dir "$RUN" \
  --pool-source base_model --teacher-mode api

echo "===== 2/3 Crossover single-task SFT: Llama2-7B, context=4096 ====="
/home/liux/miniconda3/envs/hydra/bin/python code/26_9_1/run_crossover_sft.py \
  --config "$CONFIG" --stage train \
  --data-dir "$DATA" --run-dir "$RUN"

echo "===== 3/3 Standard test100/608 evaluation ====="
/home/liux/miniconda3/envs/hydra/bin/python code/26_9_1/run_crossover_sft.py \
  --config "$CONFIG" --stage eval \
  --data-dir "$DATA" --run-dir "$RUN" \
  --pool-source base_model --teacher-mode api

echo "CROSSOVER_GATE_V2_EXIT=0"
