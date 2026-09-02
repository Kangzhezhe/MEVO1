#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SFT_CONFIG="${1:-$HERE/config_conditional_trace_full.yaml}"
IDPO_CONFIG="${2:-$HERE/config_conditional_trace_idpo_first50.yaml}"

# VisGPT 直连在本机可用；显式移除代理，避免 tmux 继承旧代理后超时。
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

echo "===== PHASE 1: CONDITIONAL-PREFERENCE TRACE + SHARED SFT/RANKER ====="
bash "$HERE/run_pipeline.sh" "$SFT_CONFIG"

echo "===== PHASE 2: FIRST-50 USERS FULL-HISTORY LOO IDPO ====="
bash "$HERE/run_idpo_gold_test_all.sh" "$IDPO_CONFIG"

echo "CONDITIONAL_TRACE_FULL_EXPERIMENT_EXIT=0"
