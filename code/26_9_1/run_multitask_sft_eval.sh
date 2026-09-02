#!/usr/bin/env bash
# 多任务 Title/Rationale SFT + 标准 Per-Pcs test100 评估。
#
# 用法：
#   bash code/26_9_1/run_multitask_sft_eval.sh \
#     /data/liux/MEVO_global_cot/dataset/editor_sets/20260902_094902_direct_parent_gold_full/01_base_parent_records.jsonl
#
# 该脚本不会重新生成 Base Parent；Parent 应先由
# build_direct_parent_gold_sft.py 完成。Teacher rationale 构建成功后才开始 SFT。

set -euo pipefail

ROOT="/home/liux/kk/MEVO_global_cot"
PYTHON="${PYTHON:-/home/liux/miniconda3/envs/hydra/bin/python}"
CONFIG="${CONFIG:-${ROOT}/code/26_9_1/config_multitask_title_rationale_sft.yaml}"
PARENT_RECORDS="${1:?请提供 01_base_parent_records.jsonl 路径}"
STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-/data/liux/MEVO_global_cot/dataset/editor_sets/${STAMP}_multitask_title_rationale_sft}"
RESULT_DIR="${RESULT_DIR:-/data/liux/MEVO_global_cot/result/${STAMP}_multitask_title_rationale_sft}"
LOG_DIR="${LOG_DIR:-/data/liux/MEVO_global_cot/logs}"
LOG_FILE="${LOG_DIR}/${STAMP}_multitask_title_rationale_sft.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "MULTITASK_START=$(date -Is)"
echo "config=${CONFIG}"
echo "parent_records=${PARENT_RECORDS}"
echo "data_dir=${DATA_DIR}"
echo "result_dir=${RESULT_DIR}"

cd "${ROOT}"

"${PYTHON}" code/26_9_1/build_multitask_rationale_sft.py \
  --config "${CONFIG}" \
  --split train \
  --limit 0 \
  --parent-records "${PARENT_RECORDS}" \
  --output "${DATA_DIR}"

"${PYTHON}" code/26_9_1/run_multitask_sft_eval.py \
  --config "${CONFIG}" \
  --sft-data "${DATA_DIR}" \
  --editor-output "${RESULT_DIR}/editor" \
  --prediction-dir "${RESULT_DIR}/predictions" \
  --reports-dir "${RESULT_DIR}/reports"

status=$?
echo "MULTITASK_SFT_EVAL_EXIT=${status}"
echo "MULTITASK_END=$(date -Is)"
exit "${status}"
