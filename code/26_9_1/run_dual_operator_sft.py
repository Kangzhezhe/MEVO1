#!/usr/bin/env python3
"""训练 Mutation + Crossover 的正式 Dual-operator SFT Editor。

本脚本不重复生成 Parent，也不重复调用 Teacher。它复用
``run_crossover_sft.py`` 的 Parent Pool 和 Pair 标注：

    Mutation:  Query + History + Parent A             -> Gold
    Crossover: Query + History + Parent A + Parent B  -> Gold

当同一 Query 有合格 Crossover Pair 时，Mutation/Crossover 样本权重为
0.7/0.3；没有合格 Pair 时只保留权重为 1.0 的 Mutation。两个操作共用同一个
Llama2-7B LoRA，并用不同任务前缀消除 Prompt 歧义。

评估阶段分别输出 Mutation、Crossover 和仅用于诊断的 Gold Oracle 报告。Oracle
不能作为正式推理结果；后续必须由 Shared Ranker 代替测试不可见 Gold 完成选择。
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CODE = HERE.parent
PROJECT = CODE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(PROJECT))

from common.metrics import corpus_bleu, score  # noqa: E402
from pipeline_common import (  # noqa: E402
    build_editor_prompt,
    deterministic_split,
    load_config,
    read_jsonl,
    resolve_path,
    write_json,
    write_jsonl,
)
from run_crossover_sft import (  # noqa: E402
    AdapterGenerator,
    clean_title,
    crossover_prompt,
    evaluate_prediction_rows,
    select_target_blind_pair,
    train_adapter,
)


def mutation_prompt(row: dict[str, Any], parent: dict[str, Any], maximum_history: int) -> str:
    return "[MUTATION_TITLE]\n" + build_editor_prompt(
        row,
        "mutation",
        parent,
        None,
        maximum_history,
        supervision_mode="plain_output_only",
    )


def build_dual_examples(
    pool_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    mutation_weight: float,
    crossover_weight: float,
    smoke: bool,
) -> list[dict[str, Any]]:
    if mutation_weight <= 0 or crossover_weight <= 0:
        raise ValueError("Mutation/Crossover 权重必须大于 0")
    total = mutation_weight + crossover_weight
    mutation_weight /= total
    crossover_weight /= total
    by_id = {str(row.get("id", "")): row for row in pair_rows}
    maximum_history = int(config.get("sft_data", {}).get("maximum_history_records", 8))
    fraction = float(config.get("sft_data", {}).get("validation_fraction", 0.05))
    examples: list[dict[str, Any]] = []
    for row in pool_rows:
        sample_id = str(row.get("id", ""))
        pool = [item for item in row.get("parent_pool", []) if clean_title(item.get("text", ""))]
        gold = clean_title(row.get("target", ""))
        if not pool or not gold:
            continue
        pair_row = by_id.get(sample_id, {})
        pair = pair_row.get("selected_pair")
        accepted = bool(pair and pair_row.get("pair_annotation", {}).get("accepted"))
        split = deterministic_split(sample_id, fraction)
        current_mutation_weight = mutation_weight if accepted else 1.0
        parent = pool[0]
        common = {
            "sample_id": sample_id,
            "user_id": str(row.get("user_id", "")),
            "target": gold,
            "output": gold,
            "trace_text": "",
            "output_text": gold,
            "split": split,
            "student_prompt_sees_gold": False,
        }
        examples.append(
            {
                **common,
                "example_id": f"{sample_id}:dual:mutation:title",
                "task": "mutation_title",
                "operation_type": "mutation",
                "parent_a_id": parent["candidate_id"],
                "parent_b_id": None,
                "parent_a": parent["text"],
                "parent_b": None,
                "prompt": mutation_prompt(row, parent, maximum_history),
                "sample_weight": current_mutation_weight,
            }
        )
        if accepted:
            examples.append(
                {
                    **common,
                    "example_id": f"{sample_id}:dual:crossover:title",
                    "task": "crossover_title",
                    "operation_type": "crossover",
                    "parent_a_id": pair["parent_a"]["candidate_id"],
                    "parent_b_id": pair["parent_b"]["candidate_id"],
                    "parent_a": pair["parent_a"]["text"],
                    "parent_b": pair["parent_b"]["text"],
                    "prompt": crossover_prompt(
                        row, pair["parent_a"], pair["parent_b"], maximum_history
                    ),
                    "sample_weight": crossover_weight,
                    "teacher_pair_decision": pair_row["pair_annotation"].get("decision"),
                }
            )
    if smoke and examples and not any(x["split"] == "validation" for x in examples):
        # 同一 Query 的两个操作必须进入同一 split；移动最后一个 Query 的所有行。
        last_id = examples[-1]["sample_id"]
        for item in examples:
            if item["sample_id"] == last_id:
                item["split"] = "validation"
    return examples


def write_dual_dataset(
    output_dir: Path,
    examples: list[dict[str, Any]],
    pool_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    *,
    mutation_weight: float,
    crossover_weight: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = [x for x in examples if x["split"] == "train"]
    validation = [x for x in examples if x["split"] == "validation"]
    if not train or not validation:
        raise ValueError(
            f"Dual-operator train/validation 不能为空：train={len(train)} validation={len(validation)}"
        )
    write_jsonl(output_dir / "all_sft.jsonl", examples)
    write_jsonl(output_dir / "train_sft.jsonl", train)
    write_jsonl(output_dir / "validation_sft.jsonl", validation)
    accepted_ids = {
        str(x.get("id", ""))
        for x in pair_rows
        if x.get("pair_annotation", {}).get("accepted")
    }
    report = {
        "protocol": "dual_operator_mutation_crossover_plain_title_sft_v1",
        "source_queries": len(pool_rows),
        "queries_with_crossover": len(accepted_ids),
        "examples": len(examples),
        "mutation_examples": sum(x["task"] == "mutation_title" for x in examples),
        "crossover_examples": sum(x["task"] == "crossover_title" for x in examples),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "mutation_weight": mutation_weight,
        "crossover_weight": crossover_weight,
        "query_total_weight_normalized": True,
        "student_prompt_sees_gold": False,
        "output_is_exact_gold": True,
        "rationale_supervision": False,
    }
    write_json(output_dir / "manifest.json", report)
    print(f"Dual-operator SFT dataset -> {output_dir}; report={report}", flush=True)
    return report


def build_stage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    crossover_dir = resolve_path(args.crossover_data_dir)
    pool_path = crossover_dir / "01_parent_pool.jsonl"
    pair_path = crossover_dir / "02_crossover_pairs.jsonl"
    if not pool_path.exists() or not pair_path.exists():
        raise FileNotFoundError(
            "必须先运行 run_crossover_sft.py --stage build；缺少 Parent Pool 或 Pair 标注"
        )
    pool_rows = read_jsonl(pool_path)
    pair_rows = read_jsonl(pair_path)
    examples = build_dual_examples(
        pool_rows,
        pair_rows,
        config,
        mutation_weight=args.mutation_weight,
        crossover_weight=args.crossover_weight,
        smoke=args.smoke,
    )
    return write_dual_dataset(
        resolve_path(args.data_dir),
        examples,
        pool_rows,
        pair_rows,
        mutation_weight=args.mutation_weight,
        crossover_weight=args.crossover_weight,
    )


def train_stage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    max_steps = args.max_steps if args.max_steps > 0 else None
    return train_adapter(
        config,
        resolve_path(args.data_dir),
        resolve_path(args.run_dir) / "editor",
        max_steps,
    )


def _operation_predictions(
    config: dict[str, Any], pool_rows: list[dict[str, Any]], adapter: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    maximum_history = int(config.get("sft_data", {}).get("maximum_history_records", 8))
    crossover_settings = config.get("crossover_sft", {})
    specs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    mutation_prompts: list[str] = []
    crossover_prompts: list[str] = []
    crossover_specs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in pool_rows:
        pool = [item for item in row.get("parent_pool", []) if clean_title(item.get("text", ""))]
        if not pool:
            specs.append((row, None))
            continue
        pair = select_target_blind_pair(row, crossover_settings)
        specs.append((row, pair))
        mutation_prompts.append(mutation_prompt(row, pool[0], maximum_history))
        if pair:
            crossover_specs.append((row, pair))
            crossover_prompts.append(
                crossover_prompt(row, pair["parent_a"], pair["parent_b"], maximum_history)
            )
    generator = AdapterGenerator(config, adapter)
    try:
        batch_size = int(config.get("evaluation", {}).get("prediction_batch_size", 1))
        mutation_values = generator.generate(mutation_prompts, batch_size)
        crossover_values = generator.generate(crossover_prompts, batch_size)
    finally:
        generator.close()
    valid_mutation_specs = [item for item in specs if item[0].get("parent_pool")]
    mutation_generated = {
        str(row.get("id", "")): prediction
        for (row, _), prediction in zip(valid_mutation_specs, mutation_values)
    }
    mutation_rows: list[dict[str, Any]] = []
    for row, pair in specs:
        pool = row.get("parent_pool", [])
        parent = pool[0] if pool else {"text": ""}
        prediction = mutation_generated.get(str(row.get("id", "")), "")
        mutation_rows.append(
            {
                "id": str(row.get("id", "")),
                "user_id": str(row.get("user_id", "")),
                "source_text": str(row.get("source_text", "")),
                "target": clean_title(row.get("target", "")),
                "parent": parent["text"],
                "prediction": prediction,
                "error": None if prediction else "no_parent_or_empty_prediction",
            }
        )
    crossover_generated = {
        str(row.get("id", "")): (pair, prediction)
        for (row, pair), prediction in zip(crossover_specs, crossover_values)
    }
    crossover_rows: list[dict[str, Any]] = []
    for row, _ in specs:
        sample_id = str(row.get("id", ""))
        generated = crossover_generated.get(sample_id)
        if generated is not None:
            pair, prediction = generated
            error = None if prediction else "empty_prediction"
        else:
            pool = row.get("parent_pool", [])
            fallback = pool[0] if pool else {"text": ""}
            pair = {"parent_a": fallback, "parent_b": {"text": ""}}
            prediction = clean_title(fallback.get("text", ""))
            error = "no_pair_fallback_parent"
        crossover_rows.append(
            {
                "id": sample_id,
                "user_id": str(row.get("user_id", "")),
                "source_text": str(row.get("source_text", "")),
                "target": clean_title(row.get("target", "")),
                "parent_a": pair["parent_a"]["text"],
                "parent_b": pair["parent_b"]["text"],
                "prediction": prediction,
                "error": error,
            }
        )
    return mutation_rows, crossover_rows


def _oracle_rows(
    mutation_rows: list[dict[str, Any]], crossover_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_cross = {str(row["id"]): row for row in crossover_rows}
    output: list[dict[str, Any]] = []
    for mutation in mutation_rows:
        crossover = by_cross.get(str(mutation["id"]))
        candidates = [("mutation", str(mutation.get("prediction", "")))]
        if crossover:
            candidates.append(("crossover", str(crossover.get("prediction", ""))))
        selected_type, selected_text = max(
            candidates,
            key=lambda item: score(item[1], str(mutation.get("target", "")))["rouge_l"],
        )
        output.append(
            {
                **mutation,
                "prediction": selected_text,
                "selected_operation": selected_type,
                "diagnostic_only_gold_oracle": True,
            }
        )
    return output


def _write_report_dir(path: Path, rows: list[dict[str, Any]], protocol: str) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    write_jsonl(path / "test_predictions.jsonl", rows)
    report = evaluate_prediction_rows(rows)
    report["protocol"] = protocol
    write_json(path / "global_test_report.json", report)
    return report


def eval_stage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    test_pool_path = (
        resolve_path(args.test_parent_pool)
        if args.test_parent_pool
        else resolve_path(args.crossover_data_dir) / "test_parent_pool.jsonl"
    )
    if not test_pool_path.exists():
        raise FileNotFoundError(
            f"缺少 test Parent Pool: {test_pool_path}；先运行 Crossover 脚本 eval/all 阶段"
        )
    pool_rows = read_jsonl(test_pool_path)
    if args.test_limit > 0:
        pool_rows = pool_rows[: args.test_limit]
    adapter = resolve_path(args.run_dir) / "editor" / "final_adapter"
    if not adapter.exists():
        raise FileNotFoundError(f"缺少 Dual-operator Adapter: {adapter}")
    mutation_rows, crossover_rows = _operation_predictions(config, pool_rows, adapter)
    oracle_rows = _oracle_rows(mutation_rows, crossover_rows)
    root = resolve_path(args.run_dir) / "reports"
    mutation_report = _write_report_dir(root / "mutation", mutation_rows, "dual_operator_mutation")
    crossover_report = _write_report_dir(
        root / "crossover", crossover_rows, "dual_operator_crossover"
    )
    oracle_report = _write_report_dir(
        root / "oracle", oracle_rows, "dual_operator_gold_oracle_diagnostic"
    )
    comparison = {
        "mutation": mutation_report,
        "crossover": crossover_report,
        "oracle": oracle_report,
        "test_pool_queries": len(pool_rows),
        "crossover_pair_queries": sum(
            row.get("error") != "no_pair_fallback_parent" for row in crossover_rows
        ),
        "crossover_pair_coverage": sum(
            row.get("error") != "no_pair_fallback_parent" for row in crossover_rows
        )
        / max(len(pool_rows), 1),
        "oracle_is_deployable": False,
    }
    write_json(root / "comparison.json", comparison)
    print(f"Dual-operator evaluation -> {root}; report={comparison}", flush=True)
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-operator Mutation+Crossover SFT")
    parser.add_argument(
        "--config", default=str(HERE / "config_dual_operator_sft.yaml")
    )
    parser.add_argument("--stage", choices=("build", "train", "eval", "all"), default="all")
    parser.add_argument("--crossover-data-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--test-parent-pool", default="")
    parser.add_argument("--mutation-weight", type=float, default=0.7)
    parser.add_argument("--crossover-weight", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.smoke:
        print("SMOKE MODE: 仅验证数据、训练和推理契约，不产生正式指标。", flush=True)
    result: dict[str, Any] = {}
    if args.stage in {"build", "all"}:
        result["build"] = build_stage(args, config)
    if args.stage in {"train", "all"}:
        result["train"] = train_stage(args, config)
    if args.stage in {"eval", "all"}:
        result["eval"] = eval_stage(args, config)
    print(f"DUAL_OPERATOR_SFT_DONE={result}", flush=True)


if __name__ == "__main__":
    main()
