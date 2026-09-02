"""构造 Top-8 + 简化个性化 Trace 的 Gold-aware SFT 监督。

设计目标：

* BM25 Top-8 只是证据候选池，最终每条 Trace 最多引用两条历史；
* 一条可靠历史即可支持局部个性化编辑，不再强制“两条共同规律”；
* 不要求精确 span、全 Parent applicability 或额外 Blind Judge；
* Student 只学习扁平的 evidence/reason/action/output 四字段；
* 没有可靠证据时安全回退为 output-only，而不是丢弃 Query。

Teacher 在离线标签构造时可以看到 Gold；写给 Student 的 Prompt 始终不含 Gold。
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable


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


META_WORDS = re.compile(
    r"\b(?:gold|target|reference answer|rouge|bleu|metric|score)\b",
    re.IGNORECASE,
)


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("simple_conditional_trace", {})


def _history(row: dict[str, Any], config: dict[str, Any]) -> list[dict[str, str]]:
    """返回压缩后的 Top-k 历史副本，保留原始 ID 供证据校验。"""

    settings = _settings(config)
    values = visible_history(row, int(settings.get("history_top_k", 8)))
    input_limit = int(settings.get("history_input_max_chars", 500))
    output_limit = int(settings.get("history_output_max_chars", 300))
    compact = []
    for item in values:
        compact.append(
            {
                "id": str(item["id"]),
                "input": str(item["input"])[:input_limit].rstrip()
                if input_limit > 0
                else str(item["input"]),
                "output": str(item["output"])[:output_limit].rstrip()
                if output_limit > 0
                else str(item["output"]),
            }
        )
    return compact


def _parents(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"parent_id": str(item["candidate_id"]), "text": str(item["text"])}
        for item in row.get("candidates", [])
    ]


def _clean_text(value: Any, maximum: int = 600) -> str:
    # 这里只做结构安全清理，不用复杂规则重新评价 Teacher 的语义。
    text = " ".join(str(value or "").strip().split())
    return text[:maximum].rstrip()


def _fallback(parent_id: str, reason: str) -> dict[str, Any]:
    return {
        "parent_id": parent_id,
        "evidence_ids": [],
        "edit_reason": "",
        "edit_action": "",
        "personalized": False,
        "fallback_reason": reason,
    }


def _validate_response(
    value: Any,
    row: dict[str, Any],
    history: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """宽容解析单次响应；局部坏 Parent 回退，不拖垮整个 Query。"""

    if isinstance(value, list):
        raw_traces = value
    elif isinstance(value, dict):
        raw_traces = value.get("traces")
    else:
        raw_traces = None
    if not isinstance(raw_traces, list):
        raise ValueError("响应必须包含 traces list")

    parent_ids = [str(item["candidate_id"]) for item in row.get("candidates", [])]
    visible_ids = {str(item["id"]) for item in history}
    by_parent: dict[str, dict[str, Any]] = {}
    for item in raw_traces:
        if not isinstance(item, dict):
            continue
        parent_id = str(item.get("parent_id", "")).strip()
        if parent_id in parent_ids and parent_id not in by_parent:
            by_parent[parent_id] = item
    if not by_parent:
        raise ValueError("没有识别出任何合法 parent_id")

    clean = []
    for parent_id in parent_ids:
        item = by_parent.get(parent_id)
        if item is None:
            clean.append(_fallback(parent_id, "Teacher omitted this Parent."))
            continue
        raw_ids = item.get("evidence_ids", [])
        if not isinstance(raw_ids, list):
            clean.append(_fallback(parent_id, "evidence_ids is not a list."))
            continue
        evidence_ids = []
        for raw_id in raw_ids:
            history_id = str(raw_id).strip()
            if history_id in visible_ids and history_id not in evidence_ids:
                evidence_ids.append(history_id)
            if len(evidence_ids) == 2:
                break
        reason = _clean_text(item.get("edit_reason"))
        action = _clean_text(item.get("edit_action"))
        if not evidence_ids:
            clean.append(_fallback(parent_id, "No valid visible evidence was selected."))
            continue
        if not reason or not action:
            clean.append(_fallback(parent_id, "Reason or action is empty."))
            continue
        if META_WORDS.search(reason) or META_WORDS.search(action):
            clean.append(_fallback(parent_id, "Trace contains evaluation meta-language."))
            continue
        clean.append(
            {
                "parent_id": parent_id,
                "evidence_ids": evidence_ids,
                "edit_reason": reason,
                "edit_action": action,
                "personalized": True,
                "fallback_reason": "",
            }
        )
    return clean


def _client(config: dict[str, Any], cache: Path) -> TeacherClient:
    settings = copy.deepcopy(config["teacher"])
    settings["temperature"] = 0.0
    settings["cache_dir"] = str(cache)
    return TeacherClient(settings, cache)


def _request_one(
    row: dict[str, Any], config: dict[str, Any], client: TeacherClient
) -> dict[str, Any]:
    history = _history(row, config)
    payload = {
        "current_input": str(row["source_text"]),
        "retrieved_history": history,
        "parents": _parents(row),
        "desired_output": str(row["target"]),
    }
    base_prompt = render_local_prompt(
        "13_simple_conditional_trace.txt",
        payload=json.dumps(payload, ensure_ascii=False),
    )
    retries = int(_settings(config).get("schema_retries", 2))
    error: Exception | None = None
    for attempt in range(retries + 1):
        prompt = base_prompt
        if attempt:
            prompt += (
                "\n\nSCHEMA RETRY: Return a fresh complete compact JSON object. "
                f"Fix this error: {error}"
            )
        task = f"simple_conditional_trace_retry_{attempt}"
        value, _ = client.json(task, prompt, payload)
        try:
            traces = _validate_response(value, row, history)
            return {
                "id": str(row["id"]),
                "user_id": str(row.get("user_id", row["id"])),
                "traces": traces,
                "quality_rejected": False,
                "rejection_reason": "",
            }
        except (TypeError, ValueError) as current:
            client.invalidate(task, prompt)
            error = current
    # Schema 连续失败时保留 Query 覆盖率，以 output-only 进入 SFT。
    return {
        "id": str(row["id"]),
        "user_id": str(row.get("user_id", row["id"])),
        "traces": [
            _fallback(str(parent["candidate_id"]), f"Schema retries exhausted: {error}")
            for parent in row.get("candidates", [])
        ],
        "quality_rejected": True,
        "rejection_reason": str(error),
    }


def _run_stage(
    rows: list[dict[str, Any]],
    destination: Path,
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    concurrency: int,
    checkpoint_every: int,
) -> dict[str, dict[str, Any]]:
    existing = read_jsonl(destination) if destination.exists() else []
    selected = {str(row["id"]) for row in rows}
    by_id = {
        str(row["id"]): row for row in existing if str(row["id"]) in selected
    }
    jobs = [row for row in rows if str(row["id"]) not in by_id]
    print(
        f"simple trace jobs={len(jobs)} resume={len(by_id)}/{len(rows)} "
        f"concurrency={concurrency}",
        flush=True,
    )

    def checkpoint() -> None:
        write_jsonl(
            destination,
            [by_id[str(row["id"])] for row in rows if str(row["id"]) in by_id],
        )

    def done(row: dict[str, Any], result: dict[str, Any], completed: int) -> None:
        by_id[str(row["id"])] = result
        interval = max(1, checkpoint_every)
        if completed % interval == 0 or completed == len(jobs):
            checkpoint()
        if completed == 1 or completed % interval == 0 or completed == len(jobs):
            personalized = sum(item["personalized"] for item in result["traces"])
            print(
                f"simple trace {completed}/{len(jobs)} sample={row['id']} "
                f"personalized={personalized}/{len(result['traces'])}",
                flush=True,
            )

    try:
        run_bounded(
            jobs,
            worker,
            done,
            max_workers=max(1, concurrency),
            thread_name_prefix="simple-trace",
        )
    except BoundedJobError as failure:
        checkpoint()
        raise RuntimeError(
            f"Simple Trace失败 sample={failure.job['id']}: {failure.error}"
        ) from failure.error
    checkpoint()
    return by_id


def _student_prompt(
    row: dict[str, Any],
    parent: dict[str, Any],
    config: dict[str, Any],
    personalized: bool,
) -> str:
    settings = _settings(config)
    mode = "simple_conditional_trace_idpo" if personalized else "output_only"
    return build_editor_prompt(
        row,
        "mutation",
        parent,
        None,
        int(settings.get("history_top_k", 8)),
        supervision_mode=mode,
        history_input_max_chars=int(settings.get("history_input_max_chars", 500)),
        history_output_max_chars=int(settings.get("history_output_max_chars", 300)),
    )


def _compile(
    rows: list[dict[str, Any]],
    generated: dict[str, dict[str, Any]],
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    examples = []
    evidence_counts = []
    fallback_reasons: dict[str, int] = {}
    for row in rows:
        result = generated[str(row["id"])]
        parents = {
            str(item["candidate_id"]): item for item in row.get("candidates", [])
        }
        for trace in result["traces"]:
            parent_id = str(trace["parent_id"])
            parent = parents[parent_id]
            personalized = bool(trace["personalized"])
            if personalized:
                prefix = {
                    "evidence_ids": trace["evidence_ids"],
                    "edit_reason": trace["edit_reason"],
                    "edit_action": trace["edit_action"],
                }
                encoded = json.dumps(prefix, ensure_ascii=False, separators=(",", ":"))
                trace_text = encoded[:-1] + ',"output":'
                output_text = json.dumps(str(row["target"]), ensure_ascii=False) + "}"
                evidence_counts.append(len(trace["evidence_ids"]))
                tier = "simple_personalized_trace"
            else:
                trace_text = ""
                output_text = json.dumps(
                    {"output": str(row["target"])},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                reason = str(trace.get("fallback_reason", "Unknown fallback."))
                fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
                tier = "output_only"
            examples.append(
                {
                    "example_id": f"{row['id']}:{parent_id}:simple-conditional-trace",
                    "sample_id": str(row["id"]),
                    "user_id": str(row.get("user_id", row["id"])),
                    "operation_type": "mutation",
                    "parent_a_id": parent_id,
                    "parent_b_id": None,
                    "output": str(row["target"]),
                    "prompt": _student_prompt(row, parent, config, personalized),
                    "trace_text": trace_text,
                    "output_text": output_text,
                    "supervision_tier": tier,
                    "quality_audit": trace,
                }
            )

    write_jsonl(output_dir / "04_compact_student_sft.jsonl", examples)
    personalized = sum(
        item["supervision_tier"] == "simple_personalized_trace" for item in examples
    )
    report = {
        "protocol": "top8_gold_aware_simple_conditional_trace_v1",
        "queries": len(rows),
        "parents": len(examples),
        "history_top_k": int(_settings(config).get("history_top_k", 8)),
        "history_input_max_chars": int(
            _settings(config).get("history_input_max_chars", 500)
        ),
        "teacher_model": str(config["teacher"]["model"]),
        "teacher_sees_gold": True,
        "student_prompt_sees_gold": False,
        "minimum_evidence": 1,
        "maximum_evidence": 2,
        "personalized_traces": personalized,
        "output_only": len(examples) - personalized,
        "personalized_trace_rate": personalized / max(len(examples), 1),
        "one_evidence_traces": sum(count == 1 for count in evidence_counts),
        "two_evidence_traces": sum(count == 2 for count in evidence_counts),
        "schema_rejected_queries": sum(
            bool(item.get("quality_rejected")) for item in generated.values()
        ),
        "fallback_reasons": fallback_reasons,
    }
    write_json(output_dir / "quality_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def run(
    config: dict[str, Any],
    count: int,
    concurrency: int,
    source: str | Path,
    output_dir: str | Path,
    checkpoint_every: int,
) -> dict[str, Any]:
    source_path = resolve_path(source)
    all_rows = read_jsonl(source_path)
    rows = all_rows[:count] if count > 0 else all_rows
    if not rows:
        raise ValueError(f"Simple Trace输入为空: {source_path}")
    for row in rows:
        if not row.get("candidates"):
            raise ValueError(f"sample={row['id']} 没有Parent")
        # Top-8 是上限；短历史用户仍允许进入并在必要时回退 output-only。
        if not visible_history(row, 1):
            print(f"warning: sample={row['id']} 没有可见历史，将回退output-only")

    destination = resolve_path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "00_selected_queries.jsonl", rows)
    client = _client(config, destination / "teacher_cache")
    generated = _run_stage(
        rows,
        destination / "01_simple_traces.jsonl",
        lambda row: _request_one(row, config, client),
        concurrency,
        checkpoint_every,
    )
    return _compile(rows, generated, destination, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-8 简化条件偏好 Trace 构造")
    parser.add_argument("--config", default=str(HERE / "config_simple_trace_top8_full.yaml"))
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    args = parser.parse_args()
    run(
        load_config(args.config),
        args.count,
        args.concurrency,
        args.source,
        args.output_dir,
        args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
