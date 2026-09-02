#!/usr/bin/env bash
# 等待当前多任务 SFT+评估成功，再使用同一 Parent 数据运行 Direct 基线。

set -euo pipefail
ROOT="/home/liux/kk/MEVO_global_cot"
PYTHON="${PYTHON:-/home/liux/miniconda3/envs/hydra/bin/python}"
MULTI_LOG="${MULTI_LOG:-${ROOT}/logs/20260902_multitask_after_parent.log}"
PARENT_SFT="${PARENT_SFT:-/data/liux/MEVO_global_cot/dataset/editor_sets/20260902_094902_direct_parent_gold_full}"
STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-/data/liux/MEVO_global_cot/result/${STAMP}_direct_parent_gold_sft_matched}"
LOG="${LOG:-${ROOT}/logs/${STAMP}_direct_parent_gold_sft_matched.log}"

mkdir -p "$(dirname "${LOG}")"
exec > >(tee -a "${LOG}") 2>&1
echo "WAIT_MULTITASK_START=$(date -Is)"

while ! grep -q 'MULTITASK_SFT_EVAL_EXIT=' "${MULTI_LOG}" 2>/dev/null; do
  sleep 60
done
status="$(grep 'MULTITASK_SFT_EVAL_EXIT=' "${MULTI_LOG}" | tail -1 | sed 's/.*=//')"
if [[ "${status}" != "0" ]]; then
  echo "MULTITASK_FAILED=${status}; direct baseline not started"
  exit 1
fi

echo "MULTITASK_SUCCEEDED; starting matched direct baseline"
cd "${ROOT}"
"${PYTHON}" code/26_9_1/run_direct_sft_eval.py \
  --config code/26_9_1/config_direct_gold_sft_base_protocol.yaml \
  --sft-data "${PARENT_SFT}" \
  --editor-output "${OUT}/editor" \
  --prediction-dir "${OUT}/predictions" \
  --reports-dir "${OUT}/reports"
echo "DIRECT_SFT_EVAL_EXIT=0"
