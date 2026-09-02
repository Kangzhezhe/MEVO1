#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_CONFIG="${1:-$HERE/config_simple_trace_top8_full.yaml}"
IDPO_CONFIG="${2:-$HERE/config_simple_trace_top8_idpo_first50.yaml}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

echo "===== PHASE 1: TOP-8 SIMPLE TRACE + SHARED SFT/RANKER ====="
bash "$HERE/run_pipeline.sh" "$SFT_CONFIG"

echo "===== PHASE 2: TOP-8 SIMPLE-TRACE PER-USER IDPO ====="
bash "$HERE/run_idpo_gold_test_all.sh" "$IDPO_CONFIG"

echo "SIMPLE_TRACE_TOP8_FULL_EXPERIMENT_EXIT=0"
