#!/usr/bin/env python3
"""训练 Direct Parent→Gold 纯文本 SFT，并在同一 test100 上评估。

该脚本与多任务入口使用完全相同的 Parent、History、测试集合和评估脚本，
唯一差别是 SFT 只训练 Title 输出，作为多任务 SFT 的严格消融基线。
"""

from __future__ import annotations

import copy
import importlib.util
import argparse
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))
from pipeline_common import load_config, read_jsonl, resolve_path, stage_path  # noqa: E402


def load_module(filename: str, name: str):
    path = CODE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(config: dict[str, Any], data_dir: Path) -> None:
    for filename in ("train_sft.jsonl", "validation_sft.jsonl"):
        path = data_dir / filename
        if not path.exists() or not read_jsonl(path):
            raise FileNotFoundError(f"Direct SFT 数据不存在或为空: {path}")
    seeds = stage_path(config, "test", "seeds")
    rows = read_jsonl(seeds)
    users = {str(row.get("user_id", "")).strip() for row in rows}
    users.discard("")
    expected_users = int(config.get("evaluation", {}).get("expected_users", 100))
    expected_queries = int(config.get("evaluation", {}).get("expected_queries", 608))
    if len(users) != expected_users or len(rows) != expected_queries:
        raise ValueError(
            f"test 口径错误：实际 users={len(users)}, queries={len(rows)}，"
            f"期望 users={expected_users}, queries={expected_queries}"
        )
    print(f"validated direct inputs: users={len(users)} queries={len(rows)}", flush=True)


def run(config: dict[str, Any], data_dir: str, editor_output: str, prediction_dir: str, reports_dir: str) -> None:
    config = copy.deepcopy(config)
    config["paths"]["sft_dir"] = data_dir
    config["paths"]["editor_output_dir"] = editor_output
    config["paths"]["prediction_dir"] = prediction_dir
    config["paths"]["reports_dir"] = reports_dir
    data_path = resolve_path(data_dir)
    validate(config, data_path)

    trainer = load_module("06_train_editor_lora.py", "direct_editor_trainer")
    print("starting Direct Parent→Gold output-only SFT", flush=True)
    trainer.train(config)
    adapter = resolve_path(editor_output) / "final_adapter"
    if not adapter.exists():
        raise FileNotFoundError(adapter)

    config["model"]["prediction_adapter_path"] = str(adapter)
    config["model"]["prediction_base_only"] = False
    generator = load_module("28_generate_global_predictions.py", "direct_generator")
    evaluator = load_module("29_evaluate_global.py", "direct_evaluator")
    destination = generator.run(config)
    report = evaluator.run(config)
    print(f"DIRECT_SFT_EVAL_DONE adapter={adapter} prediction={destination} report={report}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct Parent→Gold SFT + test100 evaluation")
    parser.add_argument("--config", default=str(HERE / "config_direct_gold_sft_base_protocol.yaml"))
    parser.add_argument("--sft-data", required=True)
    parser.add_argument("--editor-output", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--reports-dir", required=True)
    args = parser.parse_args()
    run(load_config(args.config), args.sft_data, args.editor_output, args.prediction_dir, args.reports_dir)


if __name__ == "__main__":
    main()
