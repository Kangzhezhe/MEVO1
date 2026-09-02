#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CONFIG="${1:-$HERE/config_rag_pag_test_first50.yaml}"
PYTHON="${DATA_PYTHON:-$ROOT/.venv/bin/python}"

"$PYTHON" "$HERE/23_evaluate_rag_pag_baselines.py" --config "$CONFIG"

