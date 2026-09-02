#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
TRAIN_PYTHON="${TRAIN_PYTHON:-/home/liux/miniconda3/envs/hydra/bin/python}"
DATA_PYTHON="${DATA_PYTHON:-$ROOT/.venv/bin/python}"

S0_REPORT="$ROOT/result/perpcs_s0_output_only_pipeline_train300_v1/validation_report.md"
S1_SFT="$ROOT/dataset/editor_sets/perpcs_s1_free_trace_train300_v1/train_sft.jsonl"
S2_SFT="$ROOT/dataset/editor_sets/perpcs_s2_atomic_trace_train300_v1/train_sft.jsonl"

echo "GPU queue waiting for S0 Ranker and S1/S2 SFT data..."
while [[ ! -f "$S0_REPORT" || ! -f "$S1_SFT" || ! -f "$S2_SFT" ]]; do
  s0_state=pending; s1_state=pending; s2_state=pending
  [[ -f "$S0_REPORT" ]] && s0_state=ready
  [[ -f "$S1_SFT" ]] && s1_state=ready
  [[ -f "$S2_SFT" ]] && s2_state=ready
  echo "queue wait: s0_report=$s0_state s1_sft=$s1_state s2_sft=$s2_state"
  sleep 30
done

run_method() {
  local name="$1"
  local config="$2"
  echo "===== $name Editor SFT ====="
  "$TRAIN_PYTHON" "$HERE/06_train_editor_lora.py" --config "$config"
  echo "===== $name Editor -> Ranker ====="
  bash "$HERE/run_editor_to_ranker.sh" "$config"
}

run_method S1 "$HERE/config_s1.yaml"
run_method S2 "$HERE/config_s2.yaml"

echo "===== S0/S1/S2 comparison ====="
"$DATA_PYTHON" "$HERE/14_compare_supervision.py" \
  --output "$ROOT/result/s0_s1_s2_comparison/report.json"
BEST_SELECTION="$ROOT/result/s0_s1_s2_comparison/best_method.json"
BEST_METHOD="$($DATA_PYTHON "$HERE/15_select_best.py" \
  --report "$ROOT/result/s0_s1_s2_comparison/report.json" \
  --output "$BEST_SELECTION")"
case "$BEST_METHOD" in
  S0) BEST_CONFIG="$HERE/config_s0_full.yaml" ;;
  S1) BEST_CONFIG="$HERE/config_s1_full.yaml" ;;
  S2) BEST_CONFIG="$HERE/config_s2_full.yaml" ;;
  *) echo "未知最佳方法: $BEST_METHOD" >&2; exit 1 ;;
esac
echo "===== Full Per-Pcs best method=$BEST_METHOD config=$BEST_CONFIG ====="
bash "$HERE/run_pipeline.sh" "$BEST_CONFIG"
echo "S0_S1_S2_GPU_QUEUE_EXIT=0"
