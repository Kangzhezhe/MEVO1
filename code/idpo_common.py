"""阶段二 IDPO 的数据契约与小型公共工具。

IDPO 的 response 必须与阶段一 Editor 的输出格式一致，但 Prompt/response
中不能出现 Leave-One-Out 隐藏的标题。所有 Gold 只保留在 adaptation prepare
数据中供离线评估，绝不写入 rollout 或 Judge 输入。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pipeline_common import resolve_path, score, write_json, write_jsonl


def idpo_root(config: dict[str, Any], round_index: int) -> Path:
    root = resolve_path(config["paths"]["idpo_dir"])
    return root / f"round_{int(round_index)}"


def idpo_path(config: dict[str, Any], round_index: int, name: str) -> Path:
    return idpo_root(config, round_index) / name


def seed_with_score(candidate: dict[str, Any], target: str) -> dict[str, Any]:
    """为 LOO Seed Parent 补齐与 rollout response 相同的 Gold 指标契约。"""

    text = str(candidate.get("text", "")).strip()
    if not text:
        raise ValueError(f"Seed candidate={candidate.get('candidate_id')} 缺少文本")
    metrics = score(text, target)
    return {
        **candidate,
        "candidate_id": str(candidate["candidate_id"]),
        "type": str(candidate.get("type", "task_seed")),
        "text": text,
        "scores": {
            "rouge_1": float(metrics["rouge_1"]),
            "rouge_l": float(metrics["rouge_l"]),
        },
    }


def write_idpo_jsonl(
    config: dict[str, Any], round_index: int, name: str, rows: list[dict[str, Any]]
) -> Path:
    path = idpo_path(config, round_index, name)
    write_jsonl(path, rows)
    return path


def write_idpo_json(
    config: dict[str, Any], round_index: int, name: str, value: Any
) -> Path:
    path = idpo_path(config, round_index, name)
    write_json(path, value)
    return path


def canonical_response_parts(
    operation_type: str, value: dict[str, Any], plain_output_only: bool = False
) -> tuple[str, str, str]:
    """返回 ``(trace_text, output_text, full_response)``。

    Trace-aware IDPO 需要像阶段一 SFT 一样区分辅助 Trace span 与最终输出
    span，才能在 DPO 序列分数中分别归一化、加权。旧的 output-only 和
    Gold-aware 数据契约仍然兼容。
    """

    # 新的条件偏好 Trace 是第一阶段的辅助监督。IDPO 只对学生
    # 真正 on-policy 生成的响应做偏好优化；配置可选择紧凑输出或完整 Trace。
    if plain_output_only and set(value) == {"output"}:
        output = str(value["output"]).replace("\n", " ").strip()
        return "", output, output
    if set(value) == {"output"}:
        result = {"output": str(value["output"])}
        output_text = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        return "", output_text, output_text
    if {"evidence_ids", "edit_reason", "edit_action", "output"}.issubset(value):
        prefix = {
            "evidence_ids": value["evidence_ids"],
            "edit_reason": str(value["edit_reason"]),
            "edit_action": str(value["edit_action"]),
        }
    elif {
        "history_analysis", "preference", "applicability", "edit_plan", "output"
    }.issubset(value):
        prefix = {
            "history_analysis": value["history_analysis"],
            "preference": value["preference"],
            "applicability": value["applicability"],
            "edit_plan": value["edit_plan"],
        }
    elif operation_type == "mutation":
        prefix = {
            "decision": str(value["decision"]),
            "task_correction": str(value["task_correction"]),
            "profile_signal": value["profile_signal"],
            "edit_action": str(value["edit_action"]),
        }
    elif operation_type == "crossover":
        prefix = {
            "decision": str(value["decision"]),
            "profile_signal": value["profile_signal"],
            "merge_action": str(value["merge_action"]),
        }
    else:
        raise ValueError(f"未知 IDPO operation_type={operation_type}")
    encoded = json.dumps(prefix, ensure_ascii=False, separators=(",", ":"))
    trace_text = encoded[:-1] + ',"output":'
    output_text = json.dumps(str(value["output"]), ensure_ascii=False) + "}"
    return trace_text, output_text, trace_text + output_text


def canonical_response(operation_type: str, value: dict[str, Any]) -> str:
    """序列化模型真正学习的 response，不把 parent ID 混入输出标签。"""

    return canonical_response_parts(operation_type, value)[2]


def candidate_output(value: dict[str, Any]) -> str:
    output = str(value.get("output", "")).strip()
    if not output or "\n" in output or len(output) > 300:
        raise ValueError("IDPO response output 必须是非空单行短文本")
    return output


def group_pairs_by_user(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        user_id = str(row.get("user_id", "")).strip()
        if not user_id:
            raise ValueError(f"IDPO pair={row.get('pair_id')} 缺少 user_id")
        grouped[user_id].append(row)
    return dict(grouped)
