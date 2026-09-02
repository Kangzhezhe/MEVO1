"""Top-4 条件偏好 Trace 的两阶段构造与小样本质量验证。

该 Pilot 不训练模型，只回答一个更前置的问题：Teacher 构造出的个性化 SFT
监督是否值得训练。协议刻意保持简单：

1. Target-blind Teacher 只从 Top-4 历史提取一个条件偏好并判断 Parent 适用性；
2. Gold-aware Teacher 固定该偏好，只决定它是否能解释 Parent -> Gold 的编辑；
3. 独立请求隐藏生成器的偏好描述，重新从证据推断并审计计划；
4. 任一门控失败就回退为 output-only，不保留 task-only 编辑解释。

这里验证的是“历史证据与个性化计划是否一致”，不做跨用户反事实，也不声称
提取到全局稳定或用户特有的 Factor。
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import re
import statistics
import sys
import unicodedata
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
from supervision_quality_pilot import FORBIDDEN  # noqa: E402


TOP_K = 4


class SchemaError(RuntimeError):
    """区分业务 Schema 失败和 Teacher 服务失败。"""


def _boolean(value: Any, field: str) -> bool:
    """不允许 bool("false") == True 这种隐式转换污染质量门控。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{field}必须为boolean")


def _text(
    value: Any,
    field: str,
    maximum: int = 300,
    allow_empty: bool = False,
    check_forbidden: bool = True,
) -> str:
    result = str(value or "").strip()
    if not allow_empty and not result:
        raise ValueError(f"{field}不能为空")
    if len(result) > maximum:
        raise ValueError(f"{field}超过{maximum}字符")
    if check_forbidden and FORBIDDEN.search(result):
        raise ValueError(f"{field}包含答案/指标元话语")
    return result


def _tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    normalized = normalized.translate(
        str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-"})
    )
    return re.findall(r"[a-z0-9]+", normalized)


def _grounded(span: str, text: str, threshold: float = 0.88) -> bool:
    """容忍Unicode标点/空白差异，但不接受语义改写作为原文证据。"""

    query = _tokens(span)
    source = _tokens(text)
    if not query or not source:
        return False
    width = len(query)
    if width > len(source):
        return False
    # token 子序列完全一致是最常见路径；滑窗相似度仅修复少量格式或词形噪声。
    if any(source[start : start + width] == query for start in range(len(source) - width + 1)):
        return True
    best = 0.0
    for candidate_width in range(max(1, width - 2), min(len(source), width + 2) + 1):
        for start in range(len(source) - candidate_width + 1):
            ratio = difflib.SequenceMatcher(
                None, query, source[start : start + candidate_width], autojunk=False
            ).ratio()
            if ratio > best:
                best = ratio
            if best >= threshold:
                return True
    return False


def _history(row: dict[str, Any]) -> list[dict[str, str]]:
    return visible_history(row, TOP_K)


def _parents(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"parent_id": str(item["candidate_id"]), "text": str(item["text"])}
        for item in row["candidates"]
    ]


def _client(config: dict[str, Any], cache: Path, max_tokens: int) -> TeacherClient:
    settings = copy.deepcopy(config["teacher"])
    settings["temperature"] = 0.0
    settings["max_tokens"] = max(max_tokens, int(settings.get("max_tokens", 0)))
    settings["cache_dir"] = str(cache)
    return TeacherClient(settings, cache)


def _request(
    client: TeacherClient,
    task: str,
    prompt_name: str,
    payload: dict[str, Any],
    validator: Callable[[Any], Any],
    retries: int = 2,
) -> Any:
    base = render_local_prompt(
        prompt_name, payload=json.dumps(payload, ensure_ascii=False)
    )
    error: Exception | None = None
    for attempt in range(retries + 1):
        prompt = base
        if attempt:
            prompt += f"\n\nSCHEMA RETRY: Return fresh complete JSON and fix: {error}"
        name = f"{task}_retry_{attempt}"
        value, _ = client.json(name, prompt, payload)
        try:
            return validator(value)
        except (TypeError, ValueError) as current:
            client.invalidate(name, prompt)
            error = current
    raise SchemaError(f"{task} schema连续失败: {error}")


def _validate_discovery(value: Any, row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("发现结果必须为object")
    histories = {str(item["id"]): item for item in _history(row)}
    parents = {str(item["candidate_id"]): item for item in row["candidates"]}

    raw_preference = value.get("preference")
    preference = None
    if raw_preference is not None:
        if not isinstance(raw_preference, dict):
            raise ValueError("preference必须为object或null")
        if str(raw_preference.get("preference_id", "")) != "p1":
            raise ValueError("preference_id必须为p1")
        evidence = raw_preference.get("evidence")
        if not isinstance(evidence, list) or not 2 <= len(evidence) <= TOP_K:
            raise ValueError("preference需要2到4条历史证据")
        clean_evidence = []
        evidence_ids = set()
        for index, item in enumerate(evidence):
            # 单条引文错误不应丢掉同一偏好中其他两条真实证据。
            # 但经过原文校验后仍必须剩下至少两个不同历史。
            if not isinstance(item, dict):
                continue
            history_id = str(item.get("history_id", ""))
            if history_id not in histories or history_id in evidence_ids:
                continue
            try:
                input_span = _text(
                    item.get("input_span"), f"evidence[{index}].input_span", 900
                )
                output_span = _text(
                    item.get("output_span"), f"evidence[{index}].output_span", 400
                )
            except ValueError:
                continue
            if not _grounded(input_span, histories[history_id]["input"]):
                continue
            if not _grounded(output_span, histories[history_id]["output"], 0.92):
                continue
            evidence_ids.add(history_id)
            clean_evidence.append(
                {
                    "history_id": history_id,
                    "input_span": input_span,
                    "output_span": output_span,
                }
            )
        if len(clean_evidence) >= 2:
            confidence = float(raw_preference.get("confidence", 0.0))
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("preference confidence越界")
            preference = {
                "preference_id": "p1",
                "condition": _text(raw_preference.get("condition"), "condition", 400),
                "preferred_behavior": _text(
                    raw_preference.get("preferred_behavior"),
                    "preferred_behavior",
                    400,
                ),
                "evidence": clean_evidence,
                "confidence": confidence,
            }

    raw_applicability = value.get("applicability")
    if not isinstance(raw_applicability, list):
        raise ValueError("applicability必须为list")
    values = [item for item in raw_applicability if isinstance(item, dict)]
    by_parent = {str(item.get("parent_id", "")): item for item in values}
    if len(values) != len(parents) or set(by_parent) != set(parents):
        raise ValueError("必须为每个Parent返回且只返回一次适用性")
    applicability = []
    for parent_id in parents:
        item = by_parent[parent_id]
        decision = str(item.get("decision", "")).strip().lower()
        if decision not in {"apply", "skip"}:
            raise ValueError("decision必须为apply或skip")
        matched = _text(
            item.get("matched_input_span"),
            "matched_input_span",
            700,
            allow_empty=True,
        )
        if decision == "apply":
            if preference is None:
                # 原偏好的某条证据未通过原文校验时，偏好会被
                # 安全降级为null；对应的apply也应降级，而非导致整条
                # Query的Schema重试。
                decision = "skip"
                matched = ""
            elif not _grounded(matched, str(row["source_text"])):
                # 适用性引用失败只影响这个 Parent，不应抹掉已经由两条历史
                # 支持的条件偏好；保守降级为 skip，后续不会生成个性化计划。
                decision = "skip"
                matched = ""
        elif matched:
            # skip 已经明确表示不适用；忽略模型多返回的解释片段，而不是因此
            # 拒绝整个 Query。
            matched = ""
        applicability.append(
            {
                "parent_id": parent_id,
                "decision": decision,
                "matched_input_span": matched,
                "reason": _text(item.get("reason"), "applicability.reason", 900),
            }
        )
    return {"preference": preference, "applicability": applicability}


def _empty_discovery(row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "user_id": str(row.get("user_id", row["id"])),
        "preference": None,
        "applicability": [
            {
                "parent_id": str(parent["candidate_id"]),
                "decision": "skip",
                "matched_input_span": "",
                "reason": "No reliable target-blind conditional preference was validated.",
            }
            for parent in row["candidates"]
        ],
        "quality_rejected": True,
        "rejection_reason": reason,
    }


def _discover(row: dict[str, Any], client: TeacherClient) -> dict[str, Any]:
    payload = {
        "current_input": str(row["source_text"]),
        "top4_history": _history(row),
        "parents": _parents(row),
    }
    try:
        result = _request(
            client,
            "conditional_preference_discovery",
            "10_conditional_preference_discovery.txt",
            payload,
            lambda value: _validate_discovery(value, row),
            2,
        )
    except SchemaError as error:
        return _empty_discovery(row, str(error))
    return {
        "id": str(row["id"]),
        "user_id": str(row.get("user_id", row["id"])),
        **result,
        "quality_rejected": False,
        "rejection_reason": "",
    }


def _selected_id(value: Any) -> str | None:
    if value is None or str(value).strip().lower() in {"", "null", "none"}:
        return None
    return str(value).strip()


def _validate_alignment(
    value: Any, row: dict[str, Any], discovery: dict[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("traces"), list):
        raise ValueError("对齐结果必须包含traces list")
    parents = {str(item["candidate_id"]): str(item["text"]) for item in row["candidates"]}
    values = [item for item in value["traces"] if isinstance(item, dict)]
    by_parent = {str(item.get("parent_id", "")): item for item in values}
    if len(values) != len(parents) or set(by_parent) != set(parents):
        raise ValueError("必须为每个Parent返回且只返回一次Trace")
    applicable = {
        item["parent_id"]: item["decision"] == "apply"
        for item in discovery["applicability"]
    }
    preference_exists = discovery["preference"] is not None
    traces = []
    for parent_id, parent_text in parents.items():
        item = by_parent[parent_id]
        selected = _selected_id(item.get("selected_preference_id"))
        if selected is not None:
            if selected != "p1" or not preference_exists or not applicable[parent_id]:
                raise ValueError(f"Parent={parent_id}选择了未供应或不适用偏好")
        raw_plan = item.get("plan")
        if not isinstance(raw_plan, list):
            raise ValueError("plan必须为list")
        plan = [_text(step, "plan step", 320) for step in raw_plan]
        if selected is None and plan:
            raise ValueError("未选择偏好时plan必须为空")
        if selected is not None and not 1 <= len(plan) <= 2:
            raise ValueError("选择偏好时plan必须含1到2步")
        for step in plan:
            if normalized_text(step) in {
                normalized_text(parent_text),
                normalized_text(row["target"]),
            }:
                raise ValueError("plan不得复制完整Parent或Gold")
        traces.append(
            {
                "parent_id": parent_id,
                "selected_preference_id": selected,
                "plan": plan,
                "output": str(row["target"]),
            }
        )
    return traces


def _empty_alignment(
    row: dict[str, Any],
    discovery: dict[str, Any],
    reason: str,
    rejected: bool = True,
) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "user_id": str(row.get("user_id", row["id"])),
        "preference": discovery["preference"],
        "applicability": discovery["applicability"],
        "traces": [
            {
                "parent_id": str(parent["candidate_id"]),
                "selected_preference_id": None,
                "plan": [],
                "output": str(row["target"]),
            }
            for parent in row["candidates"]
        ],
        "quality_rejected": rejected,
        "rejection_reason": reason,
    }


def _align(
    row: dict[str, Any], discovery: dict[str, Any], client: TeacherClient
) -> dict[str, Any]:
    # 没有可靠偏好，或没有任何 Parent 被 target-blind 阶段判为适用时，
    # Gold-aware 阶段没有可做的个性化判断，直接产生合法空计划而不浪费 API。
    if discovery["preference"] is None or not any(
        item["decision"] == "apply" for item in discovery["applicability"]
    ):
        return _empty_alignment(
            row,
            discovery,
            "No target-blind applicable preference; alignment skipped.",
            rejected=False,
        )
    payload = {
        "current_input": str(row["source_text"]),
        "parents": _parents(row),
        "desired_output": str(row["target"]),
        "fixed_target_blind_result": {
            "preference": discovery["preference"],
            "applicability": discovery["applicability"],
        },
    }
    try:
        traces = _request(
            client,
            "conditional_preference_alignment",
            "11_conditional_preference_align.txt",
            payload,
            lambda value: _validate_alignment(value, row, discovery),
            2,
        )
    except SchemaError as error:
        return _empty_alignment(row, discovery, str(error))
    return {
        "id": str(row["id"]),
        "user_id": str(row.get("user_id", row["id"])),
        "preference": discovery["preference"],
        "applicability": discovery["applicability"],
        "traces": traces,
        "quality_rejected": False,
        "rejection_reason": "",
    }


def _validate_audit(
    value: Any, generated: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("盲审结果必须为object")
    preference = generated["preference"]
    raw_preference_audit = value.get("preference_audit")
    preference_audit = None
    if preference is None:
        if raw_preference_audit is not None:
            raise ValueError("无偏好时preference_audit必须为null")
    else:
        # Judge 返回 null 表示它无法从隐藏解释后的证据独立重构
        # 该偏好；这是有效的负审计，不是 Schema 错误。
        raw_preference_audit = (
            raw_preference_audit
            if isinstance(raw_preference_audit, dict)
            else {}
        )
        confidence = float(raw_preference_audit.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("preference_audit confidence越界")
        preference_audit = {
            "inferred_condition": _text(
                raw_preference_audit.get("inferred_condition"),
                "inferred_condition",
                280,
                allow_empty=True,
            ),
            "inferred_behavior": _text(
                raw_preference_audit.get("inferred_behavior"),
                "inferred_behavior",
                280,
                allow_empty=True,
            ),
            "evidence_supported": _boolean(
                raw_preference_audit.get("evidence_supported", False),
                "preference_audit.evidence_supported",
            ),
            "conditional_behavior": _boolean(
                raw_preference_audit.get("conditional_behavior", False),
                "preference_audit.conditional_behavior",
            ),
            "confidence": confidence,
            "reason": _text(
                raw_preference_audit.get("reason"),
                "preference_audit.reason",
                6000,
                allow_empty=True,
            ),
        }

    selected = {
        trace["parent_id"]: trace
        for trace in generated["traces"]
        if trace["selected_preference_id"] is not None
    }
    raw_plans = value.get("plan_audits")
    if not isinstance(raw_plans, list):
        raise ValueError("plan_audits必须为list")
    values = [item for item in raw_plans if isinstance(item, dict)]
    by_parent = {str(item.get("parent_id", "")): item for item in values}
    if set(by_parent) - set(selected):
        raise ValueError("plan_audits包含未提供的Parent")
    plan_audits = []
    for parent_id in selected:
        item = by_parent.get(parent_id, {})
        confidence = float(item.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("plan_audit confidence越界")
        plan_audits.append(
            {
                "parent_id": parent_id,
                "query_applicable": _boolean(
                    item.get("query_applicable", False), "plan.query_applicable"
                ),
                "preference_consistent": _boolean(
                    item.get("preference_consistent", False),
                    "plan.preference_consistent",
                ),
                "realized_in_output": _boolean(
                    item.get("realized_in_output", False), "plan.realized_in_output"
                ),
                "content_supported": _boolean(
                    item.get("content_supported", False), "plan.content_supported"
                ),
                "actionable": _boolean(
                    item.get("actionable", False), "plan.actionable"
                ),
                "confidence": confidence,
                "reason": _text(
                    item.get("reason"),
                    "plan_audit.reason",
                    6000,
                    allow_empty=True,
                ),
            }
        )
    return {"preference_audit": preference_audit, "plan_audits": plan_audits}


def _failed_audit(generated: dict[str, Any], reason: str) -> dict[str, Any]:
    preference_audit = None
    if generated["preference"] is not None:
        preference_audit = {
            "inferred_condition": "",
            "inferred_behavior": "",
            "evidence_supported": False,
            "conditional_behavior": False,
            "confidence": 0.0,
            "reason": "Blind Judge schema rejected.",
        }
    return {
        "preference_audit": preference_audit,
        "plan_audits": [
            {
                "parent_id": trace["parent_id"],
                "query_applicable": False,
                "preference_consistent": False,
                "realized_in_output": False,
                "content_supported": False,
                "actionable": False,
                "confidence": 0.0,
                "reason": "Blind Judge schema rejected.",
            }
            for trace in generated["traces"]
            if trace["selected_preference_id"] is not None
        ],
        "quality_rejected": True,
        "rejection_reason": reason,
    }


def _judge(
    row: dict[str, Any], generated: dict[str, Any], client: TeacherClient
) -> dict[str, Any]:
    preference = generated["preference"]
    if preference is None:
        return {
            "id": str(row["id"]),
            "user_id": str(row.get("user_id", row["id"])),
            "preference_audit": None,
            "plan_audits": [],
            "quality_rejected": False,
            "rejection_reason": "No preference; blind audit skipped.",
        }
    blind_preference = None
    if preference is not None:
        blind_preference = {
            "preference_id": "p1",
            "evidence": preference["evidence"],
        }
    proposed = [
        {
            "parent_id": trace["parent_id"],
            "preference_id": trace["selected_preference_id"],
            "plan": trace["plan"],
        }
        for trace in generated["traces"]
        if trace["selected_preference_id"] is not None
    ]
    payload = {
        "current_input": str(row["source_text"]),
        "top4_history": _history(row),
        "parents": _parents(row),
        "final_output": str(row["target"]),
        "blind_preference_candidate": blind_preference,
        "proposed_personalized_plans": proposed,
    }
    try:
        audit = _request(
            client,
            "conditional_preference_blind_judge",
            "12_conditional_preference_blind_judge.txt",
            payload,
            lambda value: _validate_audit(value, generated),
            2,
        )
    except SchemaError as error:
        audit = _failed_audit(generated, str(error))
    return {
        "id": str(row["id"]),
        "user_id": str(row.get("user_id", row["id"])),
        **audit,
        "quality_rejected": bool(audit.get("quality_rejected", False)),
        "rejection_reason": str(audit.get("rejection_reason", "")),
    }


def _run_stage(
    rows: list[dict[str, Any]],
    destination: Path,
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    concurrency: int,
    label: str,
    force_ids: set[str] | None = None,
    checkpoint_every: int = 1,
) -> dict[str, dict[str, Any]]:
    existing = read_jsonl(destination) if destination.exists() else []
    selected_ids = {str(row["id"]) for row in rows}
    forced = force_ids or set()
    by_id = {
        str(row["id"]): row
        for row in existing
        if str(row["id"]) in selected_ids and str(row["id"]) not in forced
    }
    jobs = [row for row in rows if str(row["id"]) not in by_id]
    print(
        f"{label} jobs={len(jobs)} resume={len(by_id)}/{len(rows)} "
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
        interval = max(1, int(checkpoint_every))
        if completed % interval == 0 or completed == len(jobs):
            checkpoint()
        if completed == 1 or completed % interval == 0 or completed == len(jobs):
            print(
                f"{label} {completed}/{len(jobs)} sample={row['id']} "
                f"rejected={bool(result.get('quality_rejected'))}",
                flush=True,
            )

    try:
        run_bounded(
            jobs,
            worker,
            done,
            max_workers=concurrency,
            thread_name_prefix=label,
        )
    except BoundedJobError as failure:
        checkpoint()
        raise RuntimeError(
            f"{label}失败 sample={failure.job['id']}: {failure.error}"
        ) from failure.error
    checkpoint()
    return by_id


def _preference_valid(
    preference: dict[str, Any] | None,
    audit: dict[str, Any] | None,
    threshold: float,
) -> bool:
    return bool(
        preference
        and audit
        and audit["inferred_condition"]
        and audit["inferred_behavior"]
        and audit["evidence_supported"]
        and audit["conditional_behavior"]
        and float(audit["confidence"]) >= threshold
    )


def _plan_valid(
    plan: dict[str, Any], preference_ok: bool, threshold: float
) -> bool:
    return bool(
        preference_ok
        and plan["query_applicable"]
        and plan["preference_consistent"]
        and plan["realized_in_output"]
        and plan["content_supported"]
        and float(plan["confidence"]) >= threshold
    )


def _student_prompt(
    row: dict[str, Any], parent: dict[str, Any], personalized: bool
) -> str:
    # 使用正式 Editor 的 Prompt 布局，确保 1024 长度诊断与后续训练一致。
    base = build_editor_prompt(
        row,
        "mutation",
        parent,
        None,
        TOP_K,
        supervision_mode="output_only",
    )
    if not personalized:
        return base
    old = 'REQUIRED_SCHEMA:\n{"output":"..."}'
    new = (
        "REQUIRED_SCHEMA:\n"
        '{"history_analysis":{"evidence_ids":["..."]},'
        '"preference":{"condition":"...","preferred_behavior":"..."},'
        '"applicability":{"decision":"apply","matched_input_span":"..."},'
        '"edit_plan":["..."],"output":"..."}'
    )
    return base.replace(old, new)


def _compile_and_report(
    rows: list[dict[str, Any]],
    discoveries: dict[str, dict[str, Any]],
    generated: dict[str, dict[str, Any]],
    audits: dict[str, dict[str, Any]],
    output_dir: Path,
    config: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    examples = []
    preference_records = []
    plan_records = []
    positive_cases = []
    rejected_cases = []
    positive_query_ids: set[str] = set()
    rejected_query_ids: set[str] = set()

    for row in rows:
        sample_id = str(row["id"])
        discovery = discoveries[sample_id]
        aligned = generated[sample_id]
        audit = audits[sample_id]
        preference_audit = audit["preference_audit"]
        preference_ok = _preference_valid(
            discovery["preference"], preference_audit, threshold
        )
        if discovery["preference"] is not None:
            preference_records.append(
                {
                    "sample_id": sample_id,
                    "generated": discovery["preference"],
                    "audit": preference_audit,
                    "valid": preference_ok,
                }
            )
        parent_map = {
            str(parent["candidate_id"]): parent for parent in row["candidates"]
        }
        applicability = {
            item["parent_id"]: item for item in discovery["applicability"]
        }
        plan_audit_map = {
            item["parent_id"]: item for item in audit["plan_audits"]
        }
        for trace in aligned["traces"]:
            parent_id = trace["parent_id"]
            parent = parent_map[parent_id]
            plan_audit = plan_audit_map.get(parent_id)
            valid = bool(
                plan_audit and _plan_valid(plan_audit, preference_ok, threshold)
            )
            if plan_audit is not None:
                plan_records.append(
                    {
                        "sample_id": sample_id,
                        "parent_id": parent_id,
                        "trace": trace,
                        "audit": plan_audit,
                        "valid": valid,
                    }
                )

            if valid:
                assert preference_audit is not None
                evidence_ids = [
                    item["history_id"]
                    for item in discovery["preference"]["evidence"]
                ]
                prefix = {
                    "history_analysis": {"evidence_ids": evidence_ids},
                    "preference": {
                        # 使用盲审独立重构的语义，避免把生成器未经验证的解释
                        # 直接作为 Student 标签。
                        "condition": preference_audit["inferred_condition"],
                        "preferred_behavior": preference_audit["inferred_behavior"],
                    },
                    "applicability": {
                        "decision": "apply",
                        "matched_input_span": applicability[parent_id][
                            "matched_input_span"
                        ],
                    },
                    "edit_plan": trace["plan"],
                }
                encoded = json.dumps(
                    prefix, ensure_ascii=False, separators=(",", ":")
                )
                trace_text = encoded[:-1] + ',"output":'
                output_text = json.dumps(str(row["target"]), ensure_ascii=False) + "}"
                tier = "personalized_trace"
            else:
                trace_text = ""
                output_text = json.dumps(
                    {"output": str(row["target"])},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                tier = "output_only"
            example = {
                "example_id": f"{sample_id}:{parent_id}:conditional-preference",
                "sample_id": sample_id,
                "user_id": str(row.get("user_id", row["id"])),
                "operation_type": "mutation",
                "parent_a_id": parent_id,
                "parent_b_id": None,
                "output": str(row["target"]),
                "prompt": _student_prompt(row, parent, valid),
                "trace_text": trace_text,
                "output_text": output_text,
                "supervision_tier": tier,
                "quality_audit": plan_audit,
            }
            examples.append(example)
            case = {
                "sample_id": sample_id,
                "query": str(row["source_text"]),
                "top4_history": _history(row),
                "parent": str(parent["text"]),
                "gold": str(row["target"]),
                "target_blind_discovery": discovery,
                "gold_aware_trace": trace,
                "blind_audit": {
                    "preference": preference_audit,
                    "plan": plan_audit,
                },
                "supervision_tier": tier,
            }
            if (
                valid
                and sample_id not in positive_query_ids
                and len(positive_cases) < 4
            ):
                positive_cases.append(case)
                positive_query_ids.add(sample_id)
            elif (
                trace["selected_preference_id"] is not None
                and sample_id not in rejected_query_ids
                and len(rejected_cases) < 4
            ):
                rejected_cases.append(case)
                rejected_query_ids.add(sample_id)

    write_jsonl(output_dir / "04_compact_student_sft.jsonl", examples)
    write_json(output_dir / "positive_cases.json", positive_cases)
    write_json(output_dir / "rejected_cases.json", rejected_cases)

    generated_preferences = len(preference_records)
    schema_rejected = sum(
        bool(item.get("quality_rejected")) for item in discoveries.values()
    )
    selected_plans = len(plan_records)
    preference_audits = [item["audit"] for item in preference_records]
    plan_audits = [item["audit"] for item in plan_records]
    apply_decisions = sum(
        item["decision"] == "apply"
        for discovery in discoveries.values()
        for item in discovery["applicability"]
    )
    report: dict[str, Any] = {
        "protocol": "top4_target_blind_preference_then_gold_aware_plan_v1",
        "queries": len(rows),
        "parents": len(examples),
        "history_top_k": TOP_K,
        "training_max_length": int(config["training"]["max_length"]),
        "teacher_model": config["teacher"]["model"],
        "cross_user_specificity_test": False,
        "discovery": {
            "schema_rejected_queries": schema_rejected,
            "preference_null_queries": len(rows) - generated_preferences,
            "model_declined_queries": len(rows)
            - generated_preferences
            - schema_rejected,
            "generated_preferences": generated_preferences,
            "apply_decisions": apply_decisions,
            "evidence_supported_rate": sum(
                bool(item and item["evidence_supported"])
                for item in preference_audits
            )
            / max(generated_preferences, 1),
            "conditional_behavior_rate": sum(
                bool(item and item["conditional_behavior"])
                for item in preference_audits
            )
            / max(generated_preferences, 1),
            "valid_preferences": sum(item["valid"] for item in preference_records),
        },
        "alignment": {
            "schema_rejected_queries": sum(
                bool(item.get("quality_rejected")) for item in generated.values()
            ),
            "selected_plans": selected_plans,
        },
        "blind_plan_audit": {
            "judge_rejected_queries": sum(
                bool(item.get("quality_rejected")) for item in audits.values()
            ),
            "query_applicable_rate": sum(
                item["query_applicable"] for item in plan_audits
            )
            / max(selected_plans, 1),
            "preference_consistent_rate": sum(
                item["preference_consistent"] for item in plan_audits
            )
            / max(selected_plans, 1),
            "realized_in_output_rate": sum(
                item["realized_in_output"] for item in plan_audits
            )
            / max(selected_plans, 1),
            "content_supported_rate": sum(
                item["content_supported"] for item in plan_audits
            )
            / max(selected_plans, 1),
            "actionable_rate": sum(item["actionable"] for item in plan_audits)
            / max(selected_plans, 1),
            "fully_valid_plans": sum(item["valid"] for item in plan_records),
        },
        "student_supervision": {
            "personalized_trace": sum(
                item["supervision_tier"] == "personalized_trace"
                for item in examples
            ),
            "output_only": sum(
                item["supervision_tier"] == "output_only" for item in examples
            ),
            "task_only_trace": 0,
            "trace_meta_leakage": sum(
                bool(FORBIDDEN.search(item["trace_text"])) for item in examples
            ),
        },
        "limitations": [
            "不做own-vs-other用户反事实，因此只验证历史与计划一致性，不证明用户特异性",
            "生成器与盲审使用同一Teacher模型的不同请求，可能存在模型级共同偏差",
            "小样本质量Pilot不等价于下游SFT效果",
        ],
    }
    _add_length_diagnostics(report, examples, config)
    length = report["length_diagnostics"]
    blocking_reasons = []
    if not generated_preferences:
        blocking_reasons.append("未生成任何可审计的条件偏好，监督已退化为Output-only")
    if length.get("evaluated") and length.get(
        "personalized_missing_cited_history_after_truncation", 0
    ):
        blocking_reasons.append(
            "1024-token截断后，个性化Trace引用的历史证据未完整保留"
        )
    if generated_preferences and schema_rejected / len(rows) >= 0.3:
        blocking_reasons.append("条件偏好发现阶段的Schema拒绝率不低于30%")
    if selected_plans and all(
        report["blind_plan_audit"][key] == 1.0
        for key in (
            "query_applicable_rate",
            "preference_consistent_rate",
            "realized_in_output_rate",
            "content_supported_rate",
            "actionable_rate",
        )
    ):
        blocking_reasons.append("同模型盲审所有计划维度均为100%，存在明显宽松/自洽偏差风险")
    report["decision"] = {
        "ready_for_full_sft": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
    }
    write_json(output_dir / "quality_report.json", report)
    _write_markdown_report(report, output_dir)
    return report


def _add_length_diagnostics(
    report: dict[str, Any], examples: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    """按正式 Trainer 的截断规则检查 Top-4 在 1024 token 下是否仍可见。"""

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            resolve_path(config["model"]["path"]),
            use_fast=True,
            local_files_only=True,
        )
        maximum = int(config["training"]["max_length"])
        truncated = 0
        personalized = 0
        missing_evidence = 0
        response_lengths = []
        for example in examples:
            prompt_ids = tokenizer.encode(
                str(example["prompt"]), add_special_tokens=True
            )
            trace_ids = tokenizer.encode(
                str(example["trace_text"]), add_special_tokens=False
            )
            output_ids = tokenizer.encode(
                str(example["output_text"]), add_special_tokens=False
            ) + [tokenizer.eos_token_id]
            response_length = len(trace_ids) + len(output_ids)
            response_lengths.append(response_length)
            maximum_prompt = maximum - response_length
            retained = prompt_ids
            if len(prompt_ids) > maximum_prompt:
                truncated += 1
                head = max(1, maximum_prompt // 2)
                retained = prompt_ids[:head] + prompt_ids[-(maximum_prompt - head) :]
            if example["supervision_tier"] == "personalized_trace":
                personalized += 1
                decoded = tokenizer.decode(retained)
                prefix = json.loads(example["trace_text"][:-10] + "}")
                evidence_ids = prefix["history_analysis"]["evidence_ids"]
                if any(str(evidence_id) not in decoded for evidence_id in evidence_ids):
                    missing_evidence += 1
        report["length_diagnostics"] = {
            "evaluated": True,
            "prompt_truncated_examples": truncated,
            "prompt_truncated_rate": truncated / max(len(examples), 1),
            "mean_response_tokens": statistics.mean(response_lengths),
            "max_response_tokens": max(response_lengths, default=0),
            "personalized_examples": personalized,
            "personalized_missing_cited_history_after_truncation": missing_evidence,
        }
    except Exception as error:  # 长度诊断不能破坏已经完成的 API Pilot。
        report["length_diagnostics"] = {
            "evaluated": False,
            "error": f"{type(error).__name__}: {error}",
        }


def _write_markdown_report(report: dict[str, Any], output_dir: Path) -> None:
    discovery = report["discovery"]
    alignment = report["alignment"]
    plan = report["blind_plan_audit"]
    student = report["student_supervision"]
    length = report["length_diagnostics"]
    lines = [
        "# Top-4 Conditional Preference Trace 质量报告",
        "",
        f"- Query：{report['queries']}；Parent：{report['parents']}；Teacher/Judge：`{report['teacher_model']}`。",
        f"- 协议：target-blind 偏好与适用性 → Gold-aware 编辑计划 → 隐藏生成解释的盲审。",
        f"- 历史：BM25 Top-{report['history_top_k']}；训练长度上限：{report['training_max_length']}。",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 生成条件偏好 | {discovery['generated_preferences']}/{report['queries']} |",
        f"| Teacher主动返回无可靠偏好 | {discovery['model_declined_queries']}/{report['queries']} |",
        f"| 空偏好合计（主动拒绝+Schema拒绝） | {discovery['preference_null_queries']}/{report['queries']} |",
        f"| 发现阶段 Schema 拒绝 | {discovery['schema_rejected_queries']} |",
        f"| 盲审历史证据支持率 | {discovery['evidence_supported_rate']:.1%} |",
        f"| 盲审确属条件行为率 | {discovery['conditional_behavior_rate']:.1%} |",
        f"| 完全有效偏好 | {discovery['valid_preferences']}/{discovery['generated_preferences']} |",
        f"| Target-blind apply 判断 | {discovery['apply_decisions']}/{report['parents']} |",
        f"| Gold-aware 实际选择计划 | {alignment['selected_plans']}/{report['parents']} |",
        f"| 对齐阶段 Schema 拒绝 | {alignment['schema_rejected_queries']} |",
        f"| 计划当前 Query 适用率 | {plan['query_applicable_rate']:.1%} |",
        f"| 计划与偏好一致率 | {plan['preference_consistent_rate']:.1%} |",
        f"| 偏好在 Gold 中实现率 | {plan['realized_in_output_rate']:.1%} |",
        f"| 计划内容有 Query 支持率 | {plan['content_supported_rate']:.1%} |",
        f"| 计划具体可执行率 | {plan['actionable_rate']:.1%} |",
        f"| 完全有效个性化计划 | {plan['fully_valid_plans']}/{alignment['selected_plans']} |",
        f"| 最终个性化 Trace / Output-only | {student['personalized_trace']} / {student['output_only']} |",
        f"| Task-only Trace | {student['task_only_trace']} |",
        f"| Student Trace 元答案泄漏 | {student['trace_meta_leakage']} |",
    ]
    if length.get("evaluated"):
        lines.extend(
            [
                f"| {report['training_max_length']}下 Prompt 被截断 | {length['prompt_truncated_examples']}/{report['parents']} ({length['prompt_truncated_rate']:.1%}) |",
                f"| 截断后缺少所引证历史的个性化样本 | {length['personalized_missing_cited_history_after_truncation']}/{length['personalized_examples']} |",
                f"| 平均/最大 Response token | {length['mean_response_tokens']:.1f} / {length['max_response_tokens']} |",
            ]
        )
    decision = report["decision"]
    lines.extend(
        [
            "",
            "## 当前结论",
            "",
            f"- 是否可直接扩展到全量 SFT：`{str(decision['ready_for_full_sft']).lower()}`。",
            *[f"- 阻塞项：{reason}。" for reason in decision["blocking_reasons"]],
            "",
            "## 判定边界",
            "",
            "- 本实验不做跨用户对照，只能证明历史证据、当前适用性、编辑计划和 Gold 之间是否自洽。",
            "- 任一偏好或计划门控失败时，样本直接回退为 Output-only；不会留下与个性化无关的 task Trace。",
            "- 生成和盲审仍由同一个模型承担，正式扩充前应人工复核导出的正反 Case。",
        ]
    )
    (output_dir / "quality_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run(
    config: dict[str, Any],
    count: int,
    concurrency: int,
    threshold: float,
    retry_rejected: bool = False,
    source: str | Path | None = None,
    output_dir: str | Path | None = None,
    checkpoint_every: int = 1,
) -> dict[str, Any]:
    source_path = resolve_path(
        source
        or (
            "dataset/editor_sets/validated_trace_blind_quality_pilot_20/"
            "selected_raw_traces.jsonl"
        )
    )
    all_rows = read_jsonl(source_path)
    rows = all_rows[:count] if count > 0 else all_rows
    if not rows:
        raise ValueError(f"Trace输入为空:{source_path}")
    for row in rows:
        if len(row.get("candidates", [])) != 4:
            raise ValueError(f"sample={row['id']} 必须恰有4个Parent")
        if len(_history(row)) < 2:
            raise ValueError(f"sample={row['id']} 可见历史不足2条")

    destination = resolve_path(
        output_dir
        or f"dataset/editor_sets/conditional_preference_trace_pilot_{count}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "00_selected_queries.jsonl", rows)

    discovery_client = _client(config, destination / "discovery_cache", 3000)
    discovery_path = destination / "01_target_blind_preferences.jsonl"
    retry_ids = set()
    if retry_rejected and discovery_path.exists():
        retry_ids = {
            str(item["id"])
            for item in read_jsonl(discovery_path)
            if item.get("quality_rejected")
        }
    discoveries = _run_stage(
        rows,
        discovery_path,
        lambda row: _discover(row, discovery_client),
        concurrency,
        "preference-discovery",
        force_ids=retry_ids,
        checkpoint_every=checkpoint_every,
    )
    alignment_client = _client(config, destination / "alignment_cache", 3000)
    generated = _run_stage(
        rows,
        destination / "02_gold_aware_plans.jsonl",
        lambda row: _align(row, discoveries[str(row["id"])], alignment_client),
        concurrency,
        "gold-aware-alignment",
        force_ids=retry_ids,
        checkpoint_every=checkpoint_every,
    )
    judge_client = _client(config, destination / "judge_cache", 3000)
    audits = _run_stage(
        rows,
        destination / "03_blind_audits.jsonl",
        lambda row: _judge(row, generated[str(row["id"])], judge_client),
        concurrency,
        "blind-judge",
        force_ids=retry_ids,
        checkpoint_every=checkpoint_every,
    )
    report = _compile_and_report(
        rows,
        discoveries,
        generated,
        audits,
        destination,
        config,
        threshold,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Top-4 条件偏好 Trace 构造与质量验证 Pilot"
    )
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument(
        "--source",
        help="输入 03_seeds.jsonl；不指定时使用原20条Pilot。",
    )
    parser.add_argument(
        "--output-dir",
        help="独立的Trace中间产物和SFT监督目录。",
    )
    parser.add_argument(
        "--retry-rejected",
        action="store_true",
        help="只重新请求此前Schema拒绝的Query，并刷新其下游阶段",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="每完成N条原子更新断点文件；全量建议20。",
    )
    args = parser.parse_args()
    run(
        load_config(args.config),
        args.count,
        args.concurrency,
        args.confidence_threshold,
        args.retry_rejected,
        args.source,
        args.output_dir,
        args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
