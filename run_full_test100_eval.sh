#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="/home/liux/miniconda3/envs/hydra/bin/python"
CONFIG="config_global_llama2_7b_visgpt_test100_full_eval.yaml"
BASE_PID="${1:-}"

# 只让一个 Llama2-7B 进程占用 GPU；已有 Base 任务完成后再依次运行两个 Adapter。
if [[ -n "$BASE_PID" ]] && kill -0 "$BASE_PID" 2>/dev/null; then
  echo "waiting for existing Base evaluation pid=$BASE_PID"
  wait_status=0
  while kill -0 "$BASE_PID" 2>/dev/null; do sleep 10; done
fi

run_base() {
  "$PYTHON" code/30_generate_base_predictions.py --config "$CONFIG" --output-name base_text
  "$PYTHON" code/29_evaluate_global.py --config "$CONFIG" --prediction-subdir base_text --report-subdir base_text
}
run_sft() {
  "$PYTHON" code/30_generate_base_predictions.py --config "$CONFIG" \
    --adapter result/20260828_221331_mevo_global_llama2_7b_visgpt_prime_matched/editor/final_adapter \
    --output-name sft_text
  "$PYTHON" code/29_evaluate_global.py --config "$CONFIG" --prediction-subdir sft_text --report-subdir sft_text
}
run_idpo() {
  "$PYTHON" code/30_generate_base_predictions.py --config "$CONFIG" \
    --adapter result/20260828_221331_mevo_global_llama2_7b_visgpt_prime_matched/editor/global_idpo/final_adapter \
    --output-name idpo_text
  "$PYTHON" code/29_evaluate_global.py --config "$CONFIG" --prediction-subdir idpo_text --report-subdir idpo_text
}

run_base
run_sft
run_idpo
echo "FULL_TEST100_EVAL_DONE"
