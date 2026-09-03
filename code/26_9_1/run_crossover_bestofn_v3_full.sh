#!/usr/bin/env bash
set -euo pipefail

# Crossover 单任务正式实验 v3：
# 1) 为每个训练 Query 重新生成 greedy A + best-of-12 候选池；
# 2) 不看 Gold，按质量接近 A、Query 落地和增量内容选择 B；
# 3) Teacher 仅在离线阶段看 Gold，执行 merge/keep/reject 严格门控；
# 4) 训练 4096-token Llama2-7B LoRA，并在标准 test100/608 Query 上评估。

ROOT=/home/liux/kk/MEVO_global_cot
STAMP=${1:-$(date +%Y%m%d_%H%M%S)}
NAME=${STAMP}_crossover_bestof12_complement_v3_4096
DATA=/data/liux/MEVO_global_cot/dataset/editor_sets/${NAME}
RUN=/data/liux/MEVO_global_cot/result/${NAME}
LOG=/data/liux/MEVO_global_cot/logs/${NAME}.log
CONFIG=code/26_9_1/config_crossover_sft.yaml

cd "$ROOT"
mkdir -p "$DATA" "$RUN" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "===== 1/4 Build target-blind greedy + best-of-12 train Parent pools ====="
/home/liux/miniconda3/envs/hydra/bin/python code/26_9_1/run_crossover_sft.py \
  --config "$CONFIG" --stage pool \
  --data-dir "$DATA" --run-dir "$RUN" \
  --pool-source base_model

echo "===== 2/4 Select complementary B + strict Teacher Gold gate ====="
/home/liux/kk/MEVO/.venv/bin/python code/26_9_1/run_crossover_sft.py \
  --config "$CONFIG" --stage pairs \
  --data-dir "$DATA" --run-dir "$RUN" \
  --pool-source base_model --teacher-mode api

echo "===== 3/4 Crossover single-task SFT: Llama2-7B, context=4096 ====="
/home/liux/miniconda3/envs/hydra/bin/python code/26_9_1/run_crossover_sft.py \
  --config "$CONFIG" --stage train \
  --data-dir "$DATA" --run-dir "$RUN"

echo "===== 4/4 Standard test100/608 Parent pools and evaluation ====="
/home/liux/miniconda3/envs/hydra/bin/python code/26_9_1/run_crossover_sft.py \
  --config "$CONFIG" --stage eval \
  --data-dir "$DATA" --run-dir "$RUN" \
  --pool-source base_model --teacher-mode api

echo "CROSSOVER_BESTOFN_V3_EXIT=0"
