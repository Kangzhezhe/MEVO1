"""阶段 05：构造无泄漏的 Editor SFT 数据。

S0 直接从四个 target-blind Parent 构造 output-only 监督；plain_output_only
进一步只保留一行 Gold 标题，不包含 JSON 包装。Gold-aware 模式保留为后续
对照。所有模式的 Student Prompt 都看不到 Gold。
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline_common import (  # noqa: E402
    build_configured_editor_prompt,
    deterministic_split,
    load_config,
    read_jsonl,
    resolve_path,
    response_parts,
    stage_path,
    write_json,
    write_jsonl,
)


def _conditional_examples(config: dict) -> list[dict[str, Any]]:
    """读取新的条件偏好 Trace。

    Stage 28 已将每个 Parent 编译成可直接训练的 prompt / trace /
    output 三段；这里只做 Query 级划分和等权重标。
    """

    trace_dir = resolve_path(config["paths"]["conditional_trace_dir"])
    source = trace_dir / "04_compact_student_sft.jsonl"
    rows = read_jsonl(source)
    if not rows:
        raise ValueError(f"条件偏好Trace监督为空: {source}")
    by_sample: dict[str, int] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        by_sample[sample_id] = by_sample.get(sample_id, 0) + 1
    maximum = int(config["sft_data"]["maximum_examples_per_query"])
    if any(count < 1 or count > maximum for count in by_sample.values()):
        raise ValueError("每个Query的条件偏好Trace数越界")
    fraction = float(config["sft_data"]["validation_fraction"])
    for row in rows:
        sample_id = str(row["sample_id"])
        row["sample_weight"] = 1.0 / by_sample[sample_id]
        row["split"] = deterministic_split(sample_id, fraction)
        if str(row["output"]) == "":
            raise ValueError(f"sample={sample_id} 的Gold为空")
    return rows


def _example(row: dict[str, Any], trace: dict[str, Any], config: dict) -> dict[str, Any]:
    parents = {str(item["candidate_id"]): item for item in row["candidates"]}
    parent = parents[str(trace["parent_id"])]
    prompt = build_editor_prompt(
        row,
        "mutation",
        parent,
        None,
        int(config["sft_data"]["maximum_history_records"]),
    )
    example = {
        "example_id": f"{row['id']}:{parent['candidate_id']}:gold-aware",
        "sample_id": str(row["id"]),
        "user_id": str(row.get("user_id", row["id"])),
        "operation_type": "mutation",
        "parent_a_id": str(parent["candidate_id"]),
        "parent_b_id": None,
        "decision": str(trace["decision"]),
        "task_correction": str(trace["task_correction"]),
        "profile_signal": trace["profile_signal"],
        "edit_action": str(trace["edit_action"]),
        "output": str(row["target"]),
        "prompt": prompt,
    }
    trace_text, output_text = response_parts(example)
    example["trace_text"] = trace_text
    example["output_text"] = output_text
    return example


def _output_only_example(
    row: dict[str, Any], parent: dict[str, Any], config: dict, plain: bool = False
) -> dict[str, Any]:
    """S0：没有 Trace span，只对紧凑 JSON 中的最终 Gold 计算 Loss。"""

    prompt = build_configured_editor_prompt(
        config,
        row,
        "mutation",
        parent,
        None,
        int(config["sft_data"]["maximum_history_records"]),
        supervision_mode="plain_output_only" if plain else "output_only",
    )
    return {
        "example_id": f"{row['id']}:{parent['candidate_id']}:s0-output-only",
        "sample_id": str(row["id"]),
        "user_id": str(row.get("user_id", row["id"])),
        "operation_type": "mutation",
        "parent_a_id": str(parent["candidate_id"]),
        "parent_b_id": None,
        "output": str(row["target"]),
        "prompt": prompt,
        "trace_text": "",
        "output_text": (
            str(row["target"]).replace("\n", " ").strip()
            if plain
            else json.dumps({"output": str(row["target"])}, ensure_ascii=False, separators=(",", ":"))
        ),
    }


def _atomic_example(
    row: dict[str, Any], trace: dict[str, Any], config: dict
) -> dict[str, Any]:
    """S2：原子操作作为辅助监督，最终 output 仍由程序设为 Gold。"""

    parents = {str(item["candidate_id"]): item for item in row["candidates"]}
    parent = parents[str(trace["parent_id"])]
    prompt = build_editor_prompt(
        row,
        "mutation",
        parent,
        None,
        int(config["sft_data"]["maximum_history_records"]),
        supervision_mode="atomic_trace",
    )
    prefix = {
        "task_operations": trace.get("task_operations", []),
        "personalized_operations": trace.get("personalized_operations", []),
    }
    encoded = json.dumps(prefix, ensure_ascii=False, separators=(",", ":"))
    return {
        "example_id": f"{row['id']}:{parent['candidate_id']}:s2-atomic",
        "sample_id": str(row["id"]),
        "user_id": str(row.get("user_id", row["id"])),
        "operation_type": "mutation",
        "parent_a_id": str(parent["candidate_id"]),
        "parent_b_id": None,
        "output": str(row["target"]),
        "prompt": prompt,
        "trace_text": encoded[:-1] + ',"output":',
        "output_text": json.dumps(str(row["target"]), ensure_ascii=False) + "}",
    }


def build(config: dict) -> dict[str, Any]:
    mode = str(config["sft_data"].get("supervision_mode", "gold_aware_trace"))
    conditional_modes = {
        "conditional_preference_trace",
        "simple_conditional_trace",
    }
    if mode not in {
        "output_only",
        "plain_output_only",
        "gold_aware_trace",
        "atomic_trace",
        "conditional_preference_trace",
        "simple_conditional_trace",
    }:
        raise ValueError(f"未知 sft_data.supervision_mode={mode}")
    if mode in conditional_modes:
        examples = _conditional_examples(config)
        rows = [{"id": sample_id} for sample_id in {item["sample_id"] for item in examples}]
    else:
        examples = []
        source_stage = {
            "output_only": "seeds",
            "plain_output_only": "seeds",
            "gold_aware_trace": "gold_traces",
            "atomic_trace": "atomic_traces",
        }[mode]
        rows = read_jsonl(stage_path(config, "train", source_stage))
    maximum = int(config["sft_data"]["maximum_examples_per_query"])
    for row in rows if mode not in conditional_modes else []:
        if mode in {"output_only", "plain_output_only"}:
            values = list(row.get("candidates", []))
            # 纯 Gold 直接 SFT 对照：每个 Query 只使用 Base 推理时采用的
            # 第一个 Parent，避免把多个 Teacher seed/候选混入监督信号。
            # 默认保持旧行为（每个 Query 使用全部候选），只有显式开启时生效。
            if bool(config["sft_data"].get("one_parent_per_query", False)):
                values = values[:1]
        elif mode == "gold_aware_trace":
            values = list(row.get("gold_aware_traces", []))
        else:
            values = list(row.get("atomic_traces", []))
            # S2 的严格门控可能拒绝一个 Query。退回同一 Parent 的 S0 标签，
            # 保证比较时 Query 覆盖率一致，同时在 manifest 中记录 fallback。
            if not values:
                values = list(row.get("candidates", []))
        if not values or len(values) > maximum:
            raise ValueError(
                f"sample={row['id']} 的监督数必须在 1..{maximum}，实际={len(values)}"
            )
        sample_weight = 1.0 / len(values)
        split = deterministic_split(
            str(row["id"]), float(config["sft_data"]["validation_fraction"])
        )
        for value in values:
            if mode in {"output_only", "plain_output_only"} or (mode == "atomic_trace" and "parent_id" not in value):
                example = _output_only_example(row, value, config, plain=mode == "plain_output_only")
                if mode == "atomic_trace":
                    example["supervision_fallback"] = "s0_output_only"
            elif mode == "atomic_trace":
                example = _atomic_example(row, value, config)
            else:
                example = _example(row, value, config)
            if example["output"] != str(row["target"]):
                raise AssertionError("SFT output 必须精确等于 Gold")
            example["sample_weight"] = sample_weight
            example["split"] = split
            examples.append(example)

    output_dir = resolve_path(config["paths"]["sft_dir"])
    train = [item for item in examples if item["split"] == "train"]
    validation = [item for item in examples if item["split"] == "validation"]
    if not train or not validation:
        raise ValueError("SFT train/validation 必须都非空；smoke 数据过小时请增加样本")
    write_jsonl(output_dir / "all_sft.jsonl", examples)
    write_jsonl(output_dir / "train_sft.jsonl", train)
    write_jsonl(output_dir / "validation_sft.jsonl", validation)
    report = {
        "protocol": (
            {"output_only": "s0_output_only_global_editor_sft_v1",
             "plain_output_only": "s0_plain_output_only_global_editor_sft_v1",
             "gold_aware_trace": "gold_aware_global_editor_sft_v1",
             "atomic_trace": "s2_atomic_trace_global_editor_sft_v1",
             "conditional_preference_trace": "conditional_preference_global_editor_sft_v1",
             "simple_conditional_trace": "top8_simple_conditional_global_editor_sft_v1"}[mode]
        ),
        "supervision_mode": mode,
        "source_queries": len(rows),
        "examples": len(examples),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "examples_per_query_max": maximum,
        "one_parent_per_query": bool(
            config["sft_data"].get("one_parent_per_query", False)
        ),
        "query_normalized_sample_weight": True,
        # S1 与 S2 都由看见 Gold 的 Teacher 构造训练监督；只有 Student Prompt
        # 以及 validation/test 候选生成始终 target-blind。
        "teacher_sees_gold": mode in {
            "gold_aware_trace",
            "atomic_trace",
            "conditional_preference_trace",
            "simple_conditional_trace",
        },
        "student_prompt_sees_gold": False,
        "sft_output_is_exact_gold": True,
        "trace_loss_weight": float(config["training"]["trace_loss_weight"]),
        "output_loss_weight": float(config["training"]["output_loss_weight"]),
        "explicit_user_factors": False,
    }
    write_json(output_dir / "manifest.json", report)
    print(f"Editor SFT ({mode}) -> {output_dir}; report={report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="05 - 构建 Gold-aware Editor SFT")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    args = parser.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
