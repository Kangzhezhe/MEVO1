#!/usr/bin/env bash
# 等待全量 Base Parent 构建成功，然后自动执行多任务 rationale SFT、训练和 test100 评估。

set -euo pipefail

ROOT="/home/liux/kk/MEVO_global_cot"
PARENT_LOG="${PARENT_LOG:-${ROOT}/logs/20260902_094902_direct_parent_gold_full_build.log}"
PARENT_DIR="${PARENT_DIR:-/data/liux/MEVO_global_cot/dataset/editor_sets/20260902_094902_direct_parent_gold_full}"
NEXT_LOG="${NEXT_LOG:-${ROOT}/logs/20260902_multitask_after_parent.log}"

mkdir -p "$(dirname "${NEXT_LOG}")"
exec > >(tee -a "${NEXT_LOG}") 2>&1

echo "WAIT_PARENT_START=$(date -Is)"
echo "parent_log=${PARENT_LOG}"
echo "parent_dir=${PARENT_DIR}"

while :; do
  if grep -q 'DIRECT_PARENT_GOLD_BUILD_EXIT=' "${PARENT_LOG}" 2>/dev/null; then
    status="$(grep 'DIRECT_PARENT_GOLD_BUILD_EXIT=' "${PARENT_LOG}" | tail -1 | sed 's/.*=//')"
    if [[ "${status}" != "0" ]]; then
      echo "PARENT_BUILD_FAILED=${status}"
      exit 1
    fi
    break
  fi
  sleep 60
done

# 新版会写 01_base_parent_records；旧进程若使用旧代码，则回退到 all_sft，
# build_multitask_rationale_sft.py 会从同一 split 的 retrieve 文件补回字段。
if [[ -f "${PARENT_DIR}/01_base_parent_records.jsonl" ]]; then
  PARENT_RECORDS="${PARENT_DIR}/01_base_parent_records.jsonl"
elif [[ -f "${PARENT_DIR}/all_sft.jsonl" ]]; then
  PARENT_RECORDS="${PARENT_DIR}/all_sft.jsonl"
else
  echo "PARENT_RECORDS_MISSING=${PARENT_DIR}"
  exit 1
fi

echo "PARENT_BUILD_EXIT=0; parent_records=${PARENT_RECORDS}"
exec bash "${ROOT}/code/26_9_1/run_multitask_sft_eval.sh" "${PARENT_RECORDS}"
