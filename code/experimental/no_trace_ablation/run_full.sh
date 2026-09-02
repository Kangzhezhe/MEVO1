#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PIPE="$ROOT/code/26_8_24"
SFT_CONFIG="${1:-$HERE/config.yaml}"
IDPO_CONFIG="${2:-$HERE/config_idpo.yaml}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

echo "===== NO-TRACE PHASE 1: OUTPUT-ONLY SFT + SHARED RANKER ====="
bash "$HERE/run_stage1.sh" "$SFT_CONFIG"

echo "===== NO-TRACE PHASE 2: OUTPUT-ONLY PER-USER IDPO + USER HEAD ====="
bash "$PIPE/run_idpo_gold_test_all.sh" "$IDPO_CONFIG"

echo "NO_TRACE_FULL_EXPERIMENT_EXIT=0"
