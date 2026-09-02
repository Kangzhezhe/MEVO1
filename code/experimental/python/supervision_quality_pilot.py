"""在相同 50 Query 上比较 Output-only、自由 Trace 和原子 Trace 监督质量。"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from common.concurrency import BoundedJobError, run_bounded  # noqa: E402
from common.teacher import TeacherClient  # noqa: E402
from pipeline_common import (  # noqa: E402
    load_config,
    normalized_text,
    read_jsonl,
    render_local_prompt,
    resolve_path,
    stage_path,
    visible_history,
    write_json,
    write_jsonl,
)


OP_TYPES = {"add", "remove", "replace", "reorder", "compress", "expand", "format", "preserve"}
# 只拦截“答案泄漏”的元话语，不能误伤论文内容里的 locality of reference、
# ground truth 或 scored trajectories 等合法术语。
FORBIDDEN = re.compile(
    r"\b(?:gold(?:en)? (?:answer|title|output)|reference(?:'s)? "
    r"(?:answer|title|output|wording|phrasing|focus)|desired[_ -]?"
    r"(?:output|title|answer)|"
    r"target (?:answer|title|output)|rouge(?:-[a-z])?|evaluation metric)\b",
    re.IGNORECASE,
)
TOKENS = re.compile(r"[a-z0-9]+")


def _tokens(value: str) -> set[str]:
    return set(TOKENS.findall(str(value).casefold()))


def _select(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """固定哈希抽样，输入文件顺序变化也不会改变选中的 Query。"""

    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest(),
    )
    if len(ranked) < count:
        raise ValueError(f"可用 Gold-aware Query 只有 {len(ranked)}，不足 {count}")
    return ranked[:count]


def _contrast_rows(selected: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for index, row in enumerate(selected):
        for offset in range(1, len(selected)):
            other = selected[(index + offset) % len(selected)]
            if str(other.get("user_id")) != str(row.get("user_id")):
                result[str(row["id"])] = other
                break
        if str(row["id"]) not in result:
            raise ValueError(f"sample={row['id']} 找不到随机用户历史")
    return result


def _client(config: dict[str, Any], output_dir: Path) -> TeacherClient:
    settings = copy.deepcopy(config["teacher"])
    settings["max_tokens"] = max(4500, int(settings.get("max_tokens", 0)))
    cache_dir = output_dir / "teacher_cache"
    settings["cache_dir"] = str(cache_dir)
    return TeacherClient(settings, cache_dir)


def _text(value: Any, field: str, maximum: int = 240, allow_empty: bool = True) -> str:
    text = str(value or "").strip()
    if not allow_empty and not text:
        raise ValueError(f"{field} 不能为空")
    if len(text) > maximum:
        raise ValueError(f"{field} 超过 {maximum} 字符")
    if FORBIDDEN.search(text):
        raise ValueError(f"{field} 包含禁止的答案/指标措辞")
    return text


def _operation(
    value: Any,
    field: str,
    gold: str,
    parent: str,
    valid_evidence: set[str],
    personalized: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} 必须是 object")
    kind = str(value.get("type", "")).lower()
    if kind not in OP_TYPES:
        raise ValueError(f"{field}.type 无效: {kind}")
    source = _text(value.get("source_span"), f"{field}.source_span", 180)
    target = _text(value.get("target_span"), f"{field}.target_span", 180)
    # 原子 span 不能退化为“整条输入替换为整条答案”。
    if source and normalized_text(source) == normalized_text(parent):
        raise ValueError(f"{field}.source_span 不得等于完整 Parent")
    if target and normalized_text(target) == normalized_text(gold):
        raise ValueError(f"{field}.target_span 不得等于完整 Desired Output")
    source_tokens = _tokens(source)
    parent_tokens = _tokens(parent)
    target_tokens = _tokens(target)
    gold_tokens = _tokens(gold)
    if len(source_tokens) >= 8 and len(source_tokens) / max(len(parent_tokens), 1) >= 0.75:
        raise ValueError(f"{field}.source_span 覆盖大部分 Parent，不是原子操作")
    if len(target_tokens) >= 8 and len(target_tokens) / max(len(gold_tokens), 1) >= 0.75:
        raise ValueError(f"{field}.target_span 覆盖大部分 Desired Output，不是原子操作")
    result = {"type": kind, "source_span": source, "target_span": target}
    if personalized:
        evidence = list(dict.fromkeys(str(item) for item in value.get("evidence_ids", [])))
        if len(evidence) < 2:
            raise ValueError(f"{field} 至少需要两个真实历史证据")
        unknown = set(evidence) - valid_evidence
        if unknown:
            raise ValueError(f"{field} 引用了非真实历史 ID: {sorted(unknown)}")
        result.update(
            {
                "evidence_ids": evidence,
                "history_pattern": _text(
                    value.get("history_pattern"), f"{field}.history_pattern", allow_empty=False
                ),
                "application": _text(
                    value.get("application"), f"{field}.application", allow_empty=False
                ),
            }
        )
    else:
        # 对 task operation，type/source/target 已足以定义监督；不要求自由文本
        # reason，避免 Teacher 产生“与给定答案一致”一类事后元解释。
        pass
    return result


def _validate_atomic(payload: Any, row: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("traces"), list):
        raise ValueError("响应必须包含 traces list")
    parents = {str(item["candidate_id"]): item for item in row["candidates"]}
    values = [item for item in payload["traces"] if isinstance(item, dict)]
    by_parent = {str(item.get("parent_id")): item for item in values}
    if len(values) != len(parents) or set(by_parent) != set(parents):
        raise ValueError("必须为四个 Parent 各返回一次 Trace")
    valid_evidence = {
        str(item["id"])
        for item in visible_history(row, 4)
    }
    output = []
    for parent_id, parent in parents.items():
        value = by_parent[parent_id]
        task_values = value.get("task_operations", [])
        personal_values = value.get("personalized_operations", [])
        if not isinstance(task_values, list) or not isinstance(personal_values, list):
            raise ValueError("operations 必须是 list")
        if len(task_values) > 8 or len(personal_values) > 4:
            raise ValueError("原子操作数量超过预算")
        if normalized_text(parent["text"]) != normalized_text(row["target"]) and not task_values:
            raise ValueError("Parent 与 Desired Output 不同时 task_operations 不能为空")
        output.append(
            {
                "parent_id": parent_id,
                "task_operations": [
                    _operation(item, f"task_operations[{index}]", row["target"], parent["text"], valid_evidence, False)
                    for index, item in enumerate(task_values)
                ],
                "personalized_operations": [
                    _operation(item, f"personalized_operations[{index}]", row["target"], parent["text"], valid_evidence, True)
                    for index, item in enumerate(personal_values)
                ],
                "output": str(row["target"]),
            }
        )
    return output


def _atomic_one(
    row: dict[str, Any],
    contrast: dict[str, Any],
    client: TeacherClient,
    retries: int,
) -> dict[str, Any]:
    payload = {
        "current_input": str(row["source_text"]),
        "desired_output": str(row["target"]),
        "true_history": visible_history(row, 4),
        "contrast_history": visible_history(contrast, 4),
        "parents": [
            {"parent_id": item["candidate_id"], "text": item["text"]}
            for item in row["candidates"]
        ],
    }
    base_prompt = render_local_prompt(
        "05_atomic_supervision_pilot.txt",
        payload=json.dumps(payload, ensure_ascii=False),
    )
    error: Exception | None = None
    for attempt in range(retries + 1):
        prompt = base_prompt
        if attempt:
            prompt += (
                "\n\nSCHEMA RETRY: Generate a fresh independent response. Do not quote any "
                f"previous response. Fix this validation issue: {error}"
            )
        task = f"atomic_supervision_pilot_retry_{attempt}"
        response, _ = client.json(task, prompt, payload)
        try:
            traces = _validate_atomic(response, row)
            return {
                "id": str(row["id"]),
                "user_id": str(row.get("user_id", row["id"])),
                "source_text": str(row["source_text"]),
                "target": str(row["target"]),
                "retrieved_profile": row.get("retrieved_profile", []),
                "contrast_user_id": str(contrast.get("user_id", contrast["id"])),
                "candidates": row["candidates"],
                "atomic_traces": traces,
            }
        except (TypeError, ValueError) as current:
            client.invalidate(task, prompt)
            error = current
            print(
                f"atomic pilot retry sample={row['id']} attempt={attempt + 1}/{retries + 1} "
                f"reason={current}",
                flush=True,
            )
    # 严格质量实验中，拒绝本身也是结果；不能因一个 Query 让其余样本消失，
    # 更不能为了凑齐 50 条而静默放宽原子性标准。
    return {
        "id": str(row["id"]),
        "user_id": str(row.get("user_id", row["id"])),
        "source_text": str(row["source_text"]),
        "target": str(row["target"]),
        "retrieved_profile": row.get("retrieved_profile", []),
        "contrast_user_id": str(contrast.get("user_id", contrast["id"])),
        "candidates": row["candidates"],
        "atomic_traces": [],
        "quality_rejected": True,
        "rejection_reason": str(error),
    }


def _s0(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(row["id"]),
            "user_id": str(row.get("user_id", row["id"])),
            "source_text": str(row["source_text"]),
            "target": str(row["target"]),
            "candidates": row["candidates"],
            "supervision": [
                {"parent_id": item["candidate_id"], "output": str(row["target"])}
                for item in row["candidates"]
            ],
        }
        for row in rows
    ]


def _s1(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(row["id"]),
            "user_id": str(row.get("user_id", row["id"])),
            "source_text": str(row["source_text"]),
            "target": str(row["target"]),
            "retrieved_profile": row.get("retrieved_profile", []),
            "candidates": row["candidates"],
            "free_traces": row["gold_aware_traces"],
        }
        for row in rows
    ]


def _coverage(parent: str, gold: str, text: str) -> float:
    changed = _tokens(parent) ^ _tokens(gold)
    if not changed:
        return 1.0
    return len(changed & _tokens(text)) / len(changed)


def _report(s0: list[dict], s1: list[dict], s2: list[dict], output_dir: Path) -> dict:
    free, atomic = [], []
    for row in s1:
        parents = {x["candidate_id"]: x["text"] for x in row["candidates"]}
        for trace in row["free_traces"]:
            free.append((row, parents[trace["parent_id"]], trace))
    for row in s2:
        parents = {x["candidate_id"]: x["text"] for x in row["candidates"]}
        for trace in row.get("atomic_traces", []):
            atomic.append((row, parents[trace["parent_id"]], trace))

    free_forbidden = sum(
        bool(FORBIDDEN.search(" ".join([
            x["task_correction"], x["profile_signal"]["observation"], x["edit_action"]
        ])))
        for _, _, x in free
    )
    free_copy = sum(
        normalized_text(r["target"]) in normalized_text(x["edit_action"])
        for r, _, x in free
    )
    free_evidence = sum(bool(x["profile_signal"]["evidence_ids"]) for _, _, x in free)
    free_coverage = [
        _coverage(p, r["target"], " ".join([x["task_correction"], x["edit_action"]]))
        for r, p, x in free
    ]

    atomic_operations = [
        operation
        for _, _, trace in atomic
        for operation in trace["task_operations"] + trace["personalized_operations"]
    ]
    atomic_personal = [
        operation for _, _, trace in atomic for operation in trace["personalized_operations"]
    ]
    atomic_coverage = []
    for row, parent, trace in atomic:
        text = " ".join(
            " ".join(str(v) for v in operation.values() if not isinstance(v, list))
            for operation in trace["task_operations"] + trace["personalized_operations"]
        )
        atomic_coverage.append(_coverage(parent, row["target"], text))

    duplicate_queries = 0
    for row in s0:
        values = [normalized_text(item["text"]) for item in row["candidates"]]
        duplicate_queries += len(set(values)) < len(values)

    report = {
        "queries": len(s0),
        "examples_per_method": sum(len(row["supervision"]) for row in s0),
        "same_query_ids": [row["id"] for row in s0] == [row["id"] for row in s1] == [row["id"] for row in s2],
        "parent_duplicate_queries": duplicate_queries,
        "s0_output_only": {
            "exact_gold_outputs": sum(item["output"] == row["target"] for row in s0 for item in row["supervision"]),
            "trace_available": False,
        },
        "s1_free_trace": {
            "traces": len(free),
            "exact_gold_outputs": sum(x["output"] == r["target"] for r, _, x in free),
            "forbidden_or_reference_leakage": free_forbidden,
            "full_gold_copied_in_edit_action": free_copy,
            "nonempty_profile_evidence": free_evidence,
            "mean_changed_token_coverage": statistics.mean(free_coverage),
            "median_changed_token_coverage": statistics.median(free_coverage),
        },
        "s2_atomic_trace": {
            "accepted_queries": sum(not row.get("quality_rejected", False) for row in s2),
            "quality_rejected_queries": sum(bool(row.get("quality_rejected", False)) for row in s2),
            "traces": len(atomic),
            "exact_gold_outputs": sum(x["output"] == r["target"] for r, _, x in atomic),
            "atomic_operations": len(atomic_operations),
            "mean_operations_per_trace": len(atomic_operations) / max(len(atomic), 1),
            "personalized_operations": len(atomic_personal),
            "traces_with_personalized_operation": sum(bool(x["personalized_operations"]) for _, _, x in atomic),
            "minimum_evidence_per_personalized_operation": min((len(x["evidence_ids"]) for x in atomic_personal), default=0),
            "forbidden_or_reference_leakage": sum(bool(FORBIDDEN.search(json.dumps(x, ensure_ascii=False))) for x in atomic_operations),
            "full_gold_span_leakage": sum(
                any(normalized_text(r["target"]) == normalized_text(str(v)) for v in op.values() if isinstance(v, str))
                for r, _, x in atomic
                for op in x["task_operations"] + x["personalized_operations"]
            ),
            "mean_changed_token_coverage": statistics.mean(atomic_coverage),
            "median_changed_token_coverage": statistics.median(atomic_coverage),
        },
    }
    write_json(output_dir / "quality_report.json", report)
    lines = [
        "# 50 Query 三种监督信号质量对比",
        "",
        f"- Query：{report['queries']}；每种监督样本：{report['examples_per_method']}；三组 ID 完全一致：{report['same_query_ids']}。",
        f"- 含重复 Parent 的 Query：{duplicate_queries}/{report['queries']}（三组共同数据问题）。",
        "",
        "| 指标 | S0 Output-only | S1 Free Trace | S2 Atomic Trace |",
        "|---|---:|---:|---:|",
        f"| 精确 Gold Output | {report['s0_output_only']['exact_gold_outputs']} | {report['s1_free_trace']['exact_gold_outputs']} | {report['s2_atomic_trace']['exact_gold_outputs']} |",
        f"| Trace 数 | 0 | {report['s1_free_trace']['traces']} | {report['s2_atomic_trace']['traces']} |",
        f"| 通过严格质量门控的 Query | N/A | N/A | {report['s2_atomic_trace']['accepted_queries']}/{report['queries']} |",
        f"| 禁止/Reference 泄漏 | N/A | {report['s1_free_trace']['forbidden_or_reference_leakage']} | {report['s2_atomic_trace']['forbidden_or_reference_leakage']} |",
        f"| 完整 Gold 写入 Trace | N/A | {report['s1_free_trace']['full_gold_copied_in_edit_action']} | {report['s2_atomic_trace']['full_gold_span_leakage']} |",
        f"| 含个性化证据的 Trace | N/A | {report['s1_free_trace']['nonempty_profile_evidence']} | {report['s2_atomic_trace']['traces_with_personalized_operation']} |",
        f"| 变化 token 解释覆盖率（均值） | N/A | {report['s1_free_trace']['mean_changed_token_coverage']:.3f} | {report['s2_atomic_trace']['mean_changed_token_coverage']:.3f} |",
        "",
        "## 解释",
        "",
        "- S0 是最干净的输出监督，但不能教模型显式利用 Parent 或历史证据。",
        "- S1 若泄漏和完整答案复述较高，只能视为弱的事后解释监督。",
        "- S2 通过原子 span、至少两条真实历史证据和随机历史对照约束个性化归因；其主要风险是过滤后个性化覆盖率过低。",
    ]
    (output_dir / "quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run(config: dict[str, Any], count: int, seed: int, concurrency: int) -> dict:
    source = read_jsonl(stage_path(config, "train", "gold_traces"))
    selected = _select(source, count, seed)
    contrasts = _contrast_rows(selected)
    output_dir = resolve_path("dataset/editor_sets/supervision_quality_pilot_50")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "selected_queries.jsonl", selected)
    s0, s1 = _s0(selected), _s1(selected)
    write_jsonl(output_dir / "s0_output_only.jsonl", s0)
    write_jsonl(output_dir / "s1_free_trace.jsonl", s1)

    destination = output_dir / "s2_atomic_trace.jsonl"
    existing = read_jsonl(destination) if destination.exists() else []
    selected_ids = {str(row["id"]) for row in selected}
    by_id = {str(row["id"]): row for row in existing if str(row["id"]) in selected_ids}
    jobs = [row for row in selected if str(row["id"]) not in by_id]
    client = _client(config, output_dir)
    print(f"atomic pilot queries={len(jobs)}, resume={len(by_id)}/{len(selected)}, concurrency={concurrency}", flush=True)

    def checkpoint() -> None:
        write_jsonl(destination, [by_id[str(row["id"])] for row in selected if str(row["id"]) in by_id])

    def done(row: dict, result: dict, completed: int) -> None:
        by_id[str(row["id"])] = result
        if completed % 5 == 0:
            checkpoint()
        print(f"atomic pilot progress {completed}/{len(jobs)} sample={row['id']}", flush=True)

    try:
        run_bounded(
            jobs,
            lambda row: _atomic_one(row, contrasts[str(row["id"])], client, 3),
            done,
            max_workers=concurrency,
            thread_name_prefix="atomic-supervision-pilot",
        )
    except BoundedJobError as failure:
        checkpoint()
        raise RuntimeError(f"Atomic pilot 失败 sample={failure.job['id']}: {failure.error}") from failure.error
    checkpoint()
    s2 = [by_id[str(row["id"])] for row in selected]
    report = _report(s0, s1, s2, output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="50 Query 三种监督信号质量对比")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    run(load_config(args.config), args.count, args.seed, args.concurrency)


if __name__ == "__main__":
    main()
