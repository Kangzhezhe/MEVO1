"""不训练模型，审计小集合 Atomic Trace 并导出可直接训练的 SFT 样本。

本脚本复用已经固定生成的 S2 Atomic Trace，避免再次付费生成相同监督。它用独立
Teacher 请求检查任务编辑、真实历史支持、随机用户反事实和当前任务适用性，然后：

* 可信个性化轨迹：保留 task + personalized operations；
* 个性化归因不可信：删除 personalized operations，保留 task operations；
* 任务轨迹也不可信：回退为 output-only。

这里的 Judge 是快速代理审计，不能替代最终人工抽查和下游训练对照实验。
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
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
    build_editor_prompt,
    load_config,
    normalized_text,
    read_jsonl,
    render_local_prompt,
    resolve_path,
    visible_history,
    write_json,
    write_jsonl,
)
from supervision_quality_pilot import FORBIDDEN, _coverage, _select  # noqa: E402


def _load_stage05():
    path = HERE / "05_build_editor_sft.py"
    spec = importlib.util.spec_from_file_location("validated_trace_stage05", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE05 = _load_stage05()


def _client(config: dict[str, Any], output_dir: Path) -> TeacherClient:
    settings = copy.deepcopy(config["teacher"])
    settings["temperature"] = 0.0
    settings["max_tokens"] = max(3500, int(settings.get("max_tokens", 0)))
    settings["cache_dir"] = str(output_dir / "judge_cache")
    return TeacherClient(settings, output_dir / "judge_cache")


def _contrast_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """为每条样本确定一个不同用户的反事实历史，固定且可复现。"""

    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        for offset in range(1, len(rows)):
            other = rows[(index + offset) % len(rows)]
            if str(other.get("user_id")) != str(row.get("user_id")):
                result[str(row["id"])] = other
                break
        if str(row["id"]) not in result:
            raise ValueError(f"sample={row['id']} 没有不同用户的反事实历史")
    return result


def _bounded_text(value: Any, field: str, maximum: int = 360) -> str:
    text = str(value or "").strip()
    if len(text) > maximum:
        raise ValueError(f"{field} 超过 {maximum} 字符")
    return text


def _validate_audit(payload: Any, row: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("audits"), list):
        raise ValueError("Judge 响应必须包含 audits list")
    traces = {str(item["parent_id"]): item for item in row["atomic_traces"]}
    values = [item for item in payload["audits"] if isinstance(item, dict)]
    by_parent = {str(item.get("parent_id")): item for item in values}
    if len(values) != len(traces) or set(by_parent) != set(traces):
        raise ValueError("Judge 必须为每个 Parent 返回且只返回一次审计")

    result = []
    history_titles = {
        str(item["id"]): str(item.get("output", item.get("title", "")))
        for item in visible_history(row, 4)
    }
    for parent_id, trace in traces.items():
        value = by_parent[parent_id]
        personal = value.get("personalized", [])
        if not isinstance(personal, list):
            raise ValueError("personalized audit 必须是 list")
        expected = len(trace.get("personalized_operations", []))
        by_index = {int(item.get("index", -1)): item for item in personal if isinstance(item, dict)}
        if expected and set(by_index) != set(range(expected)):
            raise ValueError(f"parent={parent_id} 个性化操作审计数量不匹配")
        if not expected and personal:
            raise ValueError(f"parent={parent_id} 没有个性化操作但 Judge 返回了审计")
        personal_result = []
        for index in range(expected):
            item = by_index[index]
            confidence = float(item.get("confidence", 0.0))
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence 必须位于 [0,1]")
            spans = item.get("supporting_spans", [])
            if not isinstance(spans, list):
                raise ValueError("supporting_spans 必须是 list")
            span_result = []
            for span_index, evidence in enumerate(spans[:4]):
                if not isinstance(evidence, dict):
                    raise ValueError("supporting_spans item 必须是 object")
                evidence_id = str(evidence.get("evidence_id", ""))
                span = _bounded_text(
                    evidence.get("span"), f"supporting_spans[{span_index}].span", 180
                )
                title = history_titles.get(evidence_id)
                exact = bool(title is not None and span and span.casefold() in title.casefold())
                span_result.append(
                    {"evidence_id": evidence_id, "span": span, "exact_in_title": exact}
                )
            cited = set(trace["personalized_operations"][index].get("evidence_ids", []))
            grounded_ids = {
                value["evidence_id"]
                for value in span_result
                if value["exact_in_title"] and value["evidence_id"] in cited
            }
            personal_result.append(
                {
                    "index": index,
                    "inferred_pattern": _bounded_text(
                        item.get("inferred_pattern"), "inferred_pattern", 300
                    ),
                    "supporting_spans": span_result,
                    "two_exact_cited_evidence": len(grounded_ids) >= 2,
                    "evidence_supported": bool(item.get("evidence_supported", False)),
                    "contrast_specific": bool(item.get("contrast_specific", False)),
                    "applicable": bool(item.get("applicable", False)),
                    "genuinely_personal": bool(item.get("genuinely_personal", False)),
                    "confidence": confidence,
                    "reason": _bounded_text(item.get("reason"), "reason"),
                }
            )
        coverage = float(value.get("task_coverage", 0.0))
        if not 0.0 <= coverage <= 1.0:
            raise ValueError("task_coverage 必须位于 [0,1]")
        overall = str(value.get("overall", "output_only")).strip().lower()
        if overall not in {"personalized", "task_only", "output_only"}:
            raise ValueError(f"overall 无效: {overall}")
        issues = value.get("issues", [])
        if not isinstance(issues, list):
            raise ValueError("issues 必须是 list")
        result.append(
            {
                "parent_id": parent_id,
                "task_atomic": bool(value.get("task_atomic", False)),
                "task_query_supported": bool(value.get("task_query_supported", False)),
                "task_coverage": coverage,
                "personalized": personal_result,
                "overall": overall,
                "issues": [_bounded_text(item, "issue", 240) for item in issues[:6]],
            }
        )
    return result


def _audit_one(
    row: dict[str, Any], contrast: dict[str, Any], client: TeacherClient, retries: int
) -> dict[str, Any]:
    # 盲审不提供生成器声称的 history_pattern/application，避免 Judge 直接复述
    # 一个听起来合理、但并不被历史标题支持的事后解释。
    blind_traces = []
    for trace in row["atomic_traces"]:
        blind_traces.append(
            {
                "parent_id": trace["parent_id"],
                "task_operations": trace.get("task_operations", []),
                "personalized_operations": [
                    {
                        "type": item.get("type"),
                        "source_span": item.get("source_span"),
                        "target_span": item.get("target_span"),
                        "evidence_ids": item.get("evidence_ids", []),
                    }
                    for item in trace.get("personalized_operations", [])
                ],
            }
        )
    payload = {
        "current_input": str(row["source_text"]),
        "final_output": str(row["target"]),
        "true_history": visible_history(row, 4),
        "contrast_history": visible_history(contrast, 4),
        "parents": row["candidates"],
        "traces": blind_traces,
    }
    base = render_local_prompt(
        "06_validated_trace_judge.txt",
        payload=json.dumps(payload, ensure_ascii=False),
    )
    error: Exception | None = None
    for attempt in range(retries + 1):
        prompt = base
        if attempt:
            prompt += (
                "\n\nSCHEMA RETRY: Return a fresh complete JSON response and fix: "
                f"{error}"
            )
        task = f"validated_trace_judge_retry_{attempt}"
        response, _ = client.json(task, prompt, payload)
        try:
            return {
                "id": str(row["id"]),
                "user_id": str(row.get("user_id", row["id"])),
                "contrast_user_id": str(contrast.get("user_id", contrast["id"])),
                "audits": _validate_audit(response, row),
            }
        except (TypeError, ValueError) as current:
            client.invalidate(task, prompt)
            error = current
    raise RuntimeError(f"sample={row['id']} Judge schema 连续失败: {error}")


def _personal_pass(item: dict[str, Any], threshold: float) -> bool:
    return (
        item["two_exact_cited_evidence"]
        and bool(item["inferred_pattern"])
        and item["evidence_supported"]
        and item["contrast_specific"]
        and item["applicable"]
        and item["genuinely_personal"]
        and float(item["confidence"]) >= threshold
    )


def _sanitize_trace(
    row: dict[str, Any], trace: dict[str, Any], audit: dict[str, Any], threshold: float
) -> tuple[str, dict[str, Any] | None]:
    task_ok = (
        audit["task_atomic"]
        and audit["task_query_supported"]
        and float(audit["task_coverage"]) >= 0.6
    )
    if not task_ok:
        return "output_only", None
    kept = [
        operation
        for index, operation in enumerate(trace.get("personalized_operations", []))
        if _personal_pass(audit["personalized"][index], threshold)
    ]
    cleaned = {
        "parent_id": str(trace["parent_id"]),
        "task_operations": trace.get("task_operations", []),
        "personalized_operations": kept,
        "output": str(row["target"]),
    }
    return ("personalized" if kept else "task_only"), cleaned


def _sft_example(
    row: dict[str, Any], parent: dict[str, Any], mode: str, trace: dict[str, Any] | None, config: dict
) -> dict[str, Any]:
    if mode == "output_only":
        example = STAGE05._output_only_example(row, parent, config)
        example["supervision_tier"] = "output_only"
        return example
    if trace is None:
        raise ValueError("Atomic SFT 样本缺少 trace")
    example = STAGE05._atomic_example(row, trace, config)
    example["supervision_tier"] = mode
    return example


def _build_report(
    selected: list[dict[str, Any]], audits: dict[str, dict[str, Any]], config: dict,
    output_dir: Path, threshold: float,
) -> dict[str, Any]:
    # 旧 checkpoint 也要按当前代码重新核验 exact span，避免字段解析修复后仍
    # 沿用历史布尔值。该步骤纯本地执行，不重新请求 Teacher。
    for row in selected:
        history_titles = {
            str(item["id"]): str(item.get("output", item.get("title", "")))
            for item in visible_history(row, 4)
        }
        trace_map = {str(item["parent_id"]): item for item in row["atomic_traces"]}
        for audit in audits[str(row["id"])]["audits"]:
            trace = trace_map[str(audit["parent_id"])]
            for personal in audit["personalized"]:
                operation = trace["personalized_operations"][int(personal["index"])]
                cited = {str(item) for item in operation.get("evidence_ids", [])}
                grounded = set()
                for evidence in personal.get("supporting_spans", []):
                    evidence_id = str(evidence.get("evidence_id", ""))
                    span = str(evidence.get("span", "")).strip()
                    title = history_titles.get(evidence_id, "")
                    exact = bool(span and span.casefold() in title.casefold())
                    evidence["exact_in_title"] = exact
                    if exact and evidence_id in cited:
                        grounded.add(evidence_id)
                personal["two_exact_cited_evidence"] = len(grounded) >= 2

    flat = []
    sft_examples = []
    positive_cases = []
    rejected_personal_cases = []
    output_only_cases = []
    for row in selected:
        parent_map = {str(item["candidate_id"]): item for item in row["candidates"]}
        trace_map = {str(item["parent_id"]): item for item in row["atomic_traces"]}
        audit_map = {str(item["parent_id"]): item for item in audits[str(row["id"])]["audits"]}
        for parent_id, trace in trace_map.items():
            audit = audit_map[parent_id]
            mode, cleaned = _sanitize_trace(row, trace, audit, threshold)
            parent = parent_map[parent_id]
            example = _sft_example(row, parent, mode, cleaned, config)
            example["quality_audit"] = audit
            sft_examples.append(example)
            original_personal = len(trace.get("personalized_operations", []))
            kept_personal = len(cleaned.get("personalized_operations", [])) if cleaned else 0
            flat.append(
                {
                    "sample_id": str(row["id"]),
                    "parent_id": parent_id,
                    "mode": mode,
                    "coverage": float(audit["task_coverage"]),
                    "original_personal": original_personal,
                    "kept_personal": kept_personal,
                    "automatic_coverage": _coverage(
                        str(parent["text"]), str(row["target"]), json.dumps(trace, ensure_ascii=False)
                    ),
                    "leakage": bool(FORBIDDEN.search(json.dumps(trace, ensure_ascii=False))),
                }
            )
            case = {
                "sample_id": str(row["id"]),
                "query": str(row["source_text"]),
                "parent": str(parent["text"]),
                "gold": str(row["target"]),
                "trace": trace,
                "audit": audit,
                "sanitized_mode": mode,
                "sanitized_trace": cleaned,
            }
            if mode == "personalized" and len(positive_cases) < 3:
                positive_cases.append(case)
            if original_personal > kept_personal and len(rejected_personal_cases) < 3:
                rejected_personal_cases.append(case)
            elif mode == "output_only" and len(output_only_cases) < 2:
                output_only_cases.append(case)

    rejected_cases = rejected_personal_cases + output_only_cases
    write_jsonl(output_dir / "trainable_sft_samples.jsonl", sft_examples)
    write_json(output_dir / "positive_cases.json", positive_cases)
    write_json(output_dir / "rejected_cases.json", rejected_cases)
    personal_audits = [
        item
        for row in selected
        for audit in audits[str(row["id"])]["audits"]
        for item in audit["personalized"]
    ]
    report = {
        "queries": len(selected),
        "raw_traces": len(flat),
        "trainable_sft_samples": len(sft_examples),
        "teacher_model": config["teacher"]["model"],
        "same_model_judge_limitation": True,
        "tiers": {
            mode: sum(item["mode"] == mode for item in flat)
            for mode in ("personalized", "task_only", "output_only")
        },
        "task": {
            "atomic_rate": sum(
                audit["task_atomic"]
                for row in selected for audit in audits[str(row["id"])]["audits"]
            ) / max(len(flat), 1),
            "query_supported_rate": sum(
                audit["task_query_supported"]
                for row in selected for audit in audits[str(row["id"])]["audits"]
            ) / max(len(flat), 1),
            "mean_judge_coverage": statistics.mean(item["coverage"] for item in flat),
            "mean_automatic_changed_token_coverage": statistics.mean(
                item["automatic_coverage"] for item in flat
            ),
        },
        "personalized": {
            "raw_operations": len(personal_audits),
            "two_exact_cited_evidence_rate": sum(x["two_exact_cited_evidence"] for x in personal_audits) / max(len(personal_audits), 1),
            "evidence_supported_rate": sum(x["evidence_supported"] for x in personal_audits) / max(len(personal_audits), 1),
            "contrast_specific_rate": sum(x["contrast_specific"] for x in personal_audits) / max(len(personal_audits), 1),
            "applicable_rate": sum(x["applicable"] for x in personal_audits) / max(len(personal_audits), 1),
            "genuinely_personal_rate": sum(x["genuinely_personal"] for x in personal_audits) / max(len(personal_audits), 1),
            "fully_valid_operations": sum(_personal_pass(x, threshold) for x in personal_audits),
            "kept_rate": sum(_personal_pass(x, threshold) for x in personal_audits) / max(len(personal_audits), 1),
        },
        "leakage_traces": sum(item["leakage"] for item in flat),
        "confidence_threshold": threshold,
        "paths": {
            "sft_samples": str(output_dir / "trainable_sft_samples.jsonl"),
            "positive_cases": str(output_dir / "positive_cases.json"),
            "rejected_cases": str(output_dir / "rejected_cases.json"),
        },
    }
    write_json(output_dir / "quality_report.json", report)
    p = report["personalized"]
    t = report["task"]
    lines = [
        "# Validated Atomic Trace 小样本质量报告",
        "",
        f"- Query：{report['queries']}；Parent/Trace：{report['raw_traces']}；Teacher/Judge：`{report['teacher_model']}`。",
        f"- 清洗后分层：个性化 {report['tiers']['personalized']}，仅任务 {report['tiers']['task_only']}，Output-only {report['tiers']['output_only']}。",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| Task 原子性通过率 | {t['atomic_rate']:.1%} |",
        f"| Task Query 支持率 | {t['query_supported_rate']:.1%} |",
        f"| Judge 变化解释覆盖率 | {t['mean_judge_coverage']:.3f} |",
        f"| 自动变化 token 覆盖率 | {t['mean_automatic_changed_token_coverage']:.3f} |",
        f"| 原始个性化操作 | {p['raw_operations']} |",
        f"| 两条引用证据可逐字核验 | {p['two_exact_cited_evidence_rate']:.1%} |",
        f"| 历史证据支持率 | {p['evidence_supported_rate']:.1%} |",
        f"| 相对随机用户具有特异性 | {p['contrast_specific_rate']:.1%} |",
        f"| 当前 Query 适用率 | {p['applicable_rate']:.1%} |",
        f"| 真正个性化操作率 | {p['genuinely_personal_rate']:.1%} |",
        f"| 全部门控通过 | {p['fully_valid_operations']}/{p['raw_operations']} ({p['kept_rate']:.1%}) |",
        f"| Reference/指标泄漏 | {report['leakage_traces']} |",
        "",
        "## 使用边界",
        "",
        "- `personalized` 可用于个性化 Trace Loss；`task_only` 只计算任务编辑 Trace Loss；`output_only` 只监督最终 Gold。",
        "- 生成与 Judge 当前使用同一模型但不同请求，因此可能存在自洽偏差；正式扩充前仍需人工复核典型正负 Case。",
    ]
    (output_dir / "quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run(config: dict[str, Any], count: int, seed: int, concurrency: int, threshold: float) -> dict[str, Any]:
    source_dir = resolve_path("dataset/editor_sets/supervision_quality_pilot_50")
    raw = read_jsonl(source_dir / "s2_atomic_trace.jsonl")
    available = [row for row in raw if row.get("atomic_traces") and not row.get("quality_rejected")]
    selected = _select(available, count, seed)
    contrasts = _contrast_map(selected)
    output_dir = resolve_path(f"dataset/editor_sets/validated_trace_blind_quality_pilot_{count}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "selected_raw_traces.jsonl", selected)

    destination = output_dir / "judge_audits.jsonl"
    existing = read_jsonl(destination) if destination.exists() else []
    selected_ids = {str(row["id"]) for row in selected}
    by_id = {str(row["id"]): row for row in existing if str(row["id"]) in selected_ids}
    jobs = [row for row in selected if str(row["id"]) not in by_id]
    client = _client(config, output_dir)
    print(
        f"validated trace pilot queries={len(jobs)} resume={len(by_id)}/{len(selected)} concurrency={concurrency}",
        flush=True,
    )

    def checkpoint() -> None:
        write_jsonl(destination, [by_id[str(row["id"])] for row in selected if str(row["id"]) in by_id])

    def done(row: dict[str, Any], result: dict[str, Any], completed: int) -> None:
        by_id[str(row["id"])] = result
        checkpoint()
        print(f"validated trace audit {completed}/{len(jobs)} sample={row['id']}", flush=True)

    try:
        run_bounded(
            jobs,
            lambda row: _audit_one(row, contrasts[str(row["id"])], client, 2),
            done,
            max_workers=concurrency,
            thread_name_prefix="validated-trace-judge",
        )
    except BoundedJobError as failure:
        checkpoint()
        raise RuntimeError(f"Trace Judge 失败 sample={failure.job['id']}: {failure.error}") from failure.error
    checkpoint()
    report = _build_report(selected, by_id, config, output_dir, threshold)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validated Atomic Trace 小样本质量实验")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    args = parser.parse_args()
    run(load_config(args.config), args.count, args.seed, args.concurrency, args.confidence_threshold)


if __name__ == "__main__":
    main()
