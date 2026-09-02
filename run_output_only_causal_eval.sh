#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-/home/liux/miniconda3/envs/hydra/bin/python}"
"$PYTHON" code/28_generate_global_predictions.py --config config_output_only_full_eval_sft.yaml
"$PYTHON" code/29_evaluate_global.py --config config_output_only_full_eval_sft.yaml
"$PYTHON" code/28_generate_global_predictions.py --config config_output_only_full_eval_idpo.yaml
"$PYTHON" code/29_evaluate_global.py --config config_output_only_full_eval_idpo.yaml
echo "OUTPUT_ONLY_CAUSAL_EVAL_EXIT=0"
