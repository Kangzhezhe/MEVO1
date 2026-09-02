#!/usr/bin/env python3
"""运行 Title/Rationale 多任务 SFT，并在标准 Per-Pcs test100 上评估。

该脚本不重新生成 Parent，也不调用 Teacher。Parent 和 Teacher rationale
数据由 ``build_direct_parent_gold_sft.py`` 与
``build_multitask_rationale_sft.py`` 预先构造；本脚本负责：

    train_sft.jsonl + validation_sft.jsonl
        -> 共享 Editor LoRA SFT
        -> 608 Query / 100 用户单次标题推理
        -> ROUGE-1 / ROUGE-L / SacreBLEU

推理只使用 [TITLE] 子任务，RATIONALE 仅作为训练阶段的辅助任务。
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))

from pipeline_common import load_config, read_jsonl, resolve_path, stage_path  # noqa: E402


def _load_module(filename: str, name: str):
    path = CODE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_inputs(config: dict[str, Any], data_dir: Path) -> None:
    if str(config.get("sft_data", {}).get("supervision_mode")) != "multitask_title_rationale":
        raise ValueError(
            "该入口只接受 sft_data.supervision_mode=multitask_title_rationale"
        )
    for name in ("train_sft.jsonl", "validation_sft.jsonl"):
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"缺少多任务 SFT 数据: {path}")
        if not read_jsonl(path):
            raise ValueError(f"多任务 SFT 数据为空: {path}")
    seeds = stage_path(config, "test", "seeds")
    if not seeds.exists():
        raise FileNotFoundError(f"缺少标准 test100 seeds: {seeds}")
    rows = read_jsonl(seeds)
    expected_users = int(config.get("evaluation", {}).get("expected_users", 100))
    expected_queries = int(config.get("evaluation", {}).get("expected_queries", 608))
    users = {str(row.get("user_id", "")).strip() for row in rows}
    users.discard("")
    if expected_users and len(users) != expected_users:
        raise ValueError(
            f"test 用户口径错误：期望 {expected_users}，实际 {len(users)}；文件={seeds}"
        )
    if expected_queries and len(rows) != expected_queries:
        raise ValueError(
            f"test Query 口径错误：期望 {expected_queries}，实际 {len(rows)}；文件={seeds}"
        )
    print(
        f"validated inputs: sft_dir={data_dir} train/validation ready; "
        f"test users={len(users)} queries={len(rows)}",
        flush=True,
    )


def run(config: dict[str, Any], skip_train: bool = False, skip_eval: bool = False) -> dict[str, Any]:
    data_dir = resolve_path(config["paths"]["sft_dir"])
    _validate_inputs(config, data_dir)
    editor_output = resolve_path(config["paths"]["editor_output_dir"])
    editor_output.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"sft_dir": str(data_dir), "editor_output_dir": str(editor_output)}
    if not skip_train:
        trainer = _load_module("06_train_editor_lora.py", "multitask_editor_trainer")
        print("starting shared multi-task Title/Rationale SFT", flush=True)
        report["training"] = trainer.train(config)
    adapter = editor_output / "final_adapter"
    if not adapter.exists():
        raise FileNotFoundError(f"SFT adapter 不存在: {adapter}")

    if not skip_eval:
        eval_config = copy.deepcopy(config)
        prediction_dir = resolve_path(
            config["paths"].get("prediction_dir", editor_output / "predictions")
        )
        reports_dir = resolve_path(
            config["paths"].get("reports_dir", editor_output / "reports")
        )
        # 每次运行固定写入独立目录，避免覆盖 Base/Trace 实验结果。
        eval_config["paths"]["prediction_dir"] = str(prediction_dir)
        eval_config["paths"]["reports_dir"] = str(reports_dir)
        eval_config.setdefault("model", {})["prediction_adapter_path"] = str(adapter)
        eval_config.setdefault("model", {})["prediction_base_only"] = False
        eval_config.setdefault("evaluation", {})["prediction_batch_size"] = int(
            config.get("evaluation", {}).get("prediction_batch_size", 1)
        )
        generator = _load_module("28_generate_global_predictions.py", "multitask_generator")
        evaluator = _load_module("29_evaluate_global.py", "multitask_evaluator")
        print("generating [TITLE] predictions on standard 100-user test", flush=True)
        destination = generator.run(eval_config)
        print(f"evaluating predictions: {destination}", flush=True)
        report["evaluation"] = evaluator.run(eval_config)
        report["prediction_file"] = str(destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="多任务 Title/Rationale SFT + test100 评估")
    parser.add_argument(
        "--config",
        default=str(HERE / "config_multitask_title_rationale_sft.yaml"),
        help="多任务 SFT 配置；其中 sft_dir 必须指向已构建数据",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--sft-data", default="", help="覆盖配置中的多任务数据目录")
    parser.add_argument("--editor-output", default="", help="覆盖共享 Editor 输出目录")
    parser.add_argument("--prediction-dir", default="", help="覆盖测试预测目录")
    parser.add_argument("--reports-dir", default="", help="覆盖评估报告目录")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.sft_data:
        config.setdefault("paths", {})["sft_dir"] = args.sft_data
    if args.editor_output:
        config.setdefault("paths", {})["editor_output_dir"] = args.editor_output
    if args.prediction_dir:
        config.setdefault("paths", {})["prediction_dir"] = args.prediction_dir
    if args.reports_dir:
        config.setdefault("paths", {})["reports_dir"] = args.reports_dir
    result = run(config, skip_train=args.skip_train, skip_eval=args.skip_eval)
    print(f"MULTITASK_SFT_EVAL_DONE={result}", flush=True)


if __name__ == "__main__":
    main()
