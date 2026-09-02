"""Top-8局部用户信号 -> 任务适用性 -> Parent编辑轨迹的小样本实验。

不训练模型。使用与此前严格Trace实验相同的20 Query和80 Parent，分别缓存：
1. 不看Gold的局部用户信号；2. 看Gold但不能新增信号的原子编辑；3. 隐藏生成
解释后的独立盲审。最后导出紧凑Student JSON SFT样本和质量报告。
"""

from __future__ import annotations

import argparse
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
    visible_history,
    write_json,
    write_jsonl,
)
from supervision_quality_pilot import FORBIDDEN, _coverage  # noqa: E402

DIMENSIONS = {"content_focus", "specificity", "structure", "compression", "terminology", "surface_style"}
OPS = {"add", "remove", "replace", "reorder", "compress", "expand", "format", "preserve"}
CAUSES = {"task", "profile", "task_and_profile"}


def _text(value: Any, field: str, maximum: int = 360, empty: bool = False, check_forbidden: bool = True) -> str:
    result = str(value or "").strip()
    if not empty and not result:
        raise ValueError(f"{field}不能为空")
    if len(result) > maximum:
        raise ValueError(f"{field}超过{maximum}字符")
    if check_forbidden and FORBIDDEN.search(result):
        raise ValueError(f"{field}包含答案/指标元话语")
    return result


def _client(config: dict[str, Any], cache: Path, max_tokens: int) -> TeacherClient:
    settings = copy.deepcopy(config["teacher"])
    settings["temperature"] = 0.0
    settings["max_tokens"] = max(max_tokens, int(settings.get("max_tokens", 0)))
    settings["cache_dir"] = str(cache)
    return TeacherClient(settings, cache)


def _history(row: dict[str, Any]) -> list[dict[str, Any]]:
    return visible_history(row, 8)


def _parents(row: dict[str, Any]) -> list[dict[str, str]]:
    return [{"parent_id": str(x["candidate_id"]), "text": str(x["text"])} for x in row["candidates"]]


def _request(client: TeacherClient, task: str, prompt_name: str, payload: dict, validator, retries: int = 2):
    base = render_local_prompt(prompt_name, payload=json.dumps(payload, ensure_ascii=False))
    error: Exception | None = None
    for attempt in range(retries + 1):
        prompt = base if not attempt else base + f"\n\nSCHEMA RETRY: fresh JSON; fix: {error}"
        name = f"{task}_retry_{attempt}"
        value, _ = client.json(name, prompt, payload)
        try:
            return validator(value)
        except (TypeError, ValueError) as current:
            client.invalidate(name, prompt)
            error = current
    raise RuntimeError(f"{task} schema失败: {error}")


def _validate_signals(value: Any, row: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("signals"), list):
        raise ValueError("必须包含signals list")
    raw = [x for x in value["signals"] if isinstance(x, dict)]
    if len(raw) > 2:
        raise ValueError("最多两个局部信号")
    history = {str(x["id"]): x for x in _history(row)}
    parent_ids = {str(x["candidate_id"]) for x in row["candidates"]}
    output = []
    ids = set()
    for index, item in enumerate(raw):
        signal_id = str(item.get("signal_id") or f"s{index + 1}")
        if signal_id in ids:
            raise ValueError("signal_id重复")
        ids.add(signal_id)
        dimension = str(item.get("dimension", "")).strip().lower()
        if dimension not in DIMENSIONS:
            raise ValueError(f"dimension无效:{dimension}")
        evidence = item.get("evidence", [])
        if not isinstance(evidence, list):
            raise ValueError("evidence必须为list")
        cleaned_evidence = []
        evidence_ids = set()
        for ev in evidence:
            if not isinstance(ev, dict):
                raise ValueError("evidence item必须为object")
            eid = str(ev.get("id", ""))
            if eid not in history:
                raise ValueError(f"引用不可见历史:{eid}")
            span = _text(ev.get("output_span"), "output_span", 180)
            historical_output = str(history[eid].get("output", ""))
            if span.casefold() not in historical_output.casefold():
                raise ValueError(f"output_span不是历史输出原文:{eid}:{span}")
            evidence_ids.add(eid)
            cleaned_evidence.append({
                "id": eid,
                "input_cue": _text(ev.get("input_cue"), "input_cue", 260),
                "output_behavior": _text(ev.get("output_behavior"), "output_behavior", 260),
                "output_span": span,
            })
        if len(evidence_ids) < 2:
            raise ValueError("每个信号至少两个不同历史证据")
        applicable = list(dict.fromkeys(str(x) for x in item.get("applicable_parent_ids", [])))
        if set(applicable) - parent_ids:
            raise ValueError("applicable_parent_ids包含未知Parent")
        confidence = float(item.get("confidence", 0.0))
        if not 0 <= confidence <= 1:
            raise ValueError("confidence越界")
        output.append({
            "signal_id": signal_id,
            "dimension": dimension,
            "observation": _text(item.get("observation"), "observation", 300),
            "evidence": cleaned_evidence,
            "query_relevance": _text(item.get("query_relevance"), "query_relevance", 360),
            "applicable_parent_ids": applicable,
            "confidence": confidence,
        })
    return output


def _build_signals(row: dict[str, Any], client: TeacherClient) -> dict[str, Any]:
    payload = {"current_input": str(row["source_text"]), "top8_history": _history(row), "parents": _parents(row)}
    rejected = False
    rejection_reason = ""
    try:
        signals = _request(client, "local_profile_signals", "07_local_profile_signals.txt", payload, lambda x: _validate_signals(x, row), 2)
    except RuntimeError as error:
        # 连续无法给出可核验证据表示该Query没有可靠局部信号。严格Pilot应记录
        # 拒绝并返回空列表，而不是放宽证据约束或终止其他样本。
        signals = []
        rejected = True
        rejection_reason = str(error)
    return {"id": str(row["id"]), "user_id": str(row.get("user_id", row["id"])), "signals": signals, "quality_rejected": rejected, "rejection_reason": rejection_reason}


def _validate_traces(value: Any, row: dict[str, Any], signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("traces"), list):
        raise ValueError("必须包含traces list")
    parents = {str(x["candidate_id"]): str(x["text"]) for x in row["candidates"]}
    values = [x for x in value["traces"] if isinstance(x, dict)]
    by_parent = {str(x.get("parent_id")): x for x in values}
    if len(values) != len(parents) or set(by_parent) != set(parents):
        raise ValueError("必须为每个Parent返回一次")
    signal_map = {str(x["signal_id"]): x for x in signals}
    result = []
    for parent_id, parent in parents.items():
        item = by_parent[parent_id]
        selected = item.get("selected_signals", [])
        edits = item.get("edits", [])
        if not isinstance(selected, list) or not isinstance(edits, list):
            raise ValueError("selected_signals/edits必须为list")
        selected_clean = []
        selected_ids = set()
        for selected_item in selected:
            sid = str(selected_item.get("signal_id", ""))
            if sid not in signal_map or parent_id not in signal_map[sid]["applicable_parent_ids"]:
                raise ValueError(f"应用未供应或不适用信号:{sid}")
            selected_ids.add(sid)
            # 这两个字段只用于离线审计，不进入Student target。允许Teacher提到
            # target/reference，由后续盲审决定实际edit是否有效。
            selected_clean.append({"signal_id": sid, "parent_gap": _text(selected_item.get("parent_gap"), "parent_gap", 300, check_forbidden=False), "expected_benefit": _text(selected_item.get("expected_benefit"), "expected_benefit", 300, check_forbidden=False)})
        if len(edits) > 8:
            raise ValueError("edit数量超过8")
        edits_clean = []
        for edit in edits:
            kind = str(edit.get("type", "")).lower()
            cause = str(edit.get("cause", "")).lower()
            if kind not in OPS or cause not in CAUSES:
                raise ValueError("edit type/cause无效")
            source = _text(edit.get("source"), "source", 180, True)
            target = _text(edit.get("target"), "target", 180, True)
            if source and source.casefold() not in parent.casefold():
                raise ValueError(f"source不是Parent子串:{source}")
            if target and target.casefold() not in str(row["target"]).casefold():
                raise ValueError(f"target不是Gold子串:{target}")
            if source and normalized_text(source) == normalized_text(parent):
                raise ValueError("禁止整Parent source")
            if target and normalized_text(target) == normalized_text(row["target"]):
                raise ValueError("禁止整Gold target")
            sid_value = edit.get("signal_id")
            sid = None if sid_value is None or str(sid_value).lower() in {"", "null", "none"} else str(sid_value)
            if cause == "task" and sid is not None:
                raise ValueError("task edit不能引用signal")
            if cause != "task" and sid not in selected_ids:
                raise ValueError("profile edit必须引用selected signal")
            edits_clean.append({"type": kind, "source": source, "target": target, "cause": cause, "query_basis": _text(edit.get("query_basis"), "query_basis", 300, True, check_forbidden=False), "signal_id": sid})
        if normalized_text(parent) != normalized_text(row["target"]) and not edits_clean:
            raise ValueError("非Gold Parent必须有edit")
        result.append({"parent_id": parent_id, "selected_signals": selected_clean, "edits": edits_clean, "output": str(row["target"])})
    return result


def _align_traces(row: dict[str, Any], signal_row: dict[str, Any], client: TeacherClient) -> dict[str, Any]:
    payload = {"current_input": str(row["source_text"]), "parents": _parents(row), "desired_output": str(row["target"]), "local_signals": signal_row["signals"]}
    rejected = False
    rejection_reason = ""
    try:
        traces = _request(client, "local_profile_trace_align", "08_local_profile_trace_align.txt", payload, lambda x: _validate_traces(x, row, signal_row["signals"]), 1)
    except RuntimeError as error:
        # 无法构造无泄漏、局部可执行轨迹时仍保留精确Gold输出监督，四个Parent
        # 均回退到output-only；不能因追求Trace覆盖率而接收元答案解释。
        traces = [{"parent_id": str(parent["candidate_id"]), "selected_signals": [], "edits": [], "output": str(row["target"])} for parent in row["candidates"]]
        rejected = True
        rejection_reason = str(error)
    return {"id": str(row["id"]), "user_id": str(row.get("user_id", row["id"])), "signals": signal_row["signals"], "traces": traces, "quality_rejected": rejected, "rejection_reason": rejection_reason}


def _validate_judge(value: Any, row: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Judge必须返回object")
    signals = {str(x["signal_id"]): x for x in generated["signals"]}
    signal_values = [x for x in value.get("signal_audits", []) if isinstance(x, dict)]
    signal_map = {str(x.get("signal_id")): x for x in signal_values}
    if set(signal_map) != set(signals):
        raise ValueError("Judge signal集合不一致")
    signal_audits = []
    for sid in signals:
        x = signal_map[sid]
        confidence = float(x.get("confidence", 0.0))
        if not 0 <= confidence <= 1:
            raise ValueError("Judge signal confidence越界")
        signal_audits.append({"signal_id": sid, "inferred_observation": _text(x.get("inferred_observation"), "inferred_observation", 300, True), "evidence_supported": bool(x.get("evidence_supported", False)), "query_relevant": bool(x.get("query_relevant", False)), "confidence": confidence, "reason": _text(x.get("reason"), "reason", 360, True)})
    traces = {str(x["parent_id"]): x for x in generated["traces"]}
    trace_values = [x for x in value.get("trace_audits", []) if isinstance(x, dict)]
    trace_map = {str(x.get("parent_id")): x for x in trace_values}
    if set(trace_map) != set(traces):
        raise ValueError("Judge trace集合不一致")
    trace_audits = []
    for pid, trace in traces.items():
        x = trace_map[pid]
        coverage = float(x.get("changed_content_coverage", 0.0))
        if not 0 <= coverage <= 1:
            raise ValueError("coverage越界")
        edits = [y for y in x.get("edits", []) if isinstance(y, dict)]
        by_index = {int(y.get("index", -1)): y for y in edits}
        if set(by_index) != set(range(len(trace["edits"]))):
            raise ValueError("Judge edit数量不一致")
        clean = []
        for index, edit in enumerate(trace["edits"]):
            y = by_index[index]
            profile_value = y.get("profile_consistent")
            profile_consistent = None if profile_value is None else bool(profile_value)
            if edit["cause"] == "task" and profile_consistent is not None:
                raise ValueError("task edit的profile_consistent必须null")
            if edit["cause"] != "task" and profile_consistent is None:
                raise ValueError("profile edit缺profile_consistent")
            clean.append({"index": index, "task_supported": bool(y.get("task_supported", False)), "profile_consistent": profile_consistent, "reason": _text(y.get("reason"), "edit reason", 300, True)})
        trace_audits.append({"parent_id": pid, "changed_content_coverage": coverage, "edits": clean})
    return {"signal_audits": signal_audits, "trace_audits": trace_audits}


def _judge(row: dict[str, Any], generated: dict[str, Any], client: TeacherClient) -> dict[str, Any]:
    blind_signals = [{"signal_id": x["signal_id"], "dimension": x["dimension"], "evidence_ids": [e["id"] for e in x["evidence"]]} for x in generated["signals"]]
    blind_traces = []
    for trace in generated["traces"]:
        blind_traces.append({"parent_id": trace["parent_id"], "selected_signal_ids": [x["signal_id"] for x in trace["selected_signals"]], "edits": [{"type": e["type"], "source": e["source"], "target": e["target"], "cause": e["cause"], "signal_id": e["signal_id"]} for e in trace["edits"]]})
    payload = {"current_input": str(row["source_text"]), "top8_history": _history(row), "parents": _parents(row), "signal_candidates": blind_signals, "traces": blind_traces}
    rejected = False
    rejection_reason = ""
    try:
        audit = _request(client, "local_profile_trace_blind_judge", "09_local_profile_trace_blind_judge.txt", payload, lambda x: _validate_judge(x, row, generated), 2)
    except RuntimeError as error:
        # Judge漏审任何signal/edit时不能默认通过。构造形状完整的保守失败审计，
        # 使该Query最终回退为output-only并继续其余样本。
        audit = {
            "signal_audits": [
                {"signal_id": signal["signal_id"], "inferred_observation": "", "evidence_supported": False, "query_relevant": False, "confidence": 0.0, "reason": "Blind Judge schema rejected."}
                for signal in generated["signals"]
            ],
            "trace_audits": [
                {"parent_id": trace["parent_id"], "changed_content_coverage": 0.0, "edits": [
                    {"index": index, "task_supported": False, "profile_consistent": (None if edit["cause"] == "task" else False), "reason": "Blind Judge schema rejected."}
                    for index, edit in enumerate(trace["edits"])
                ]}
                for trace in generated["traces"]
            ],
        }
        rejected = True
        rejection_reason = str(error)
    return {"id": str(row["id"]), **audit, "quality_rejected": rejected, "rejection_reason": rejection_reason}


def _run_stage(rows, destination: Path, worker, concurrency: int, label: str, retry_rejected: bool = False, force_ids: set[str] | None = None):
    existing = read_jsonl(destination) if destination.exists() else []
    selected_ids = {str(x["id"]) for x in rows}
    forced = force_ids or set()
    by_id = {str(x["id"]): x for x in existing if str(x["id"]) in selected_ids and str(x["id"]) not in forced and not (retry_rejected and x.get("quality_rejected"))}
    jobs = [x for x in rows if str(x["id"]) not in by_id]
    print(f"{label} jobs={len(jobs)} resume={len(by_id)}/{len(rows)} concurrency={concurrency}", flush=True)
    def checkpoint(): write_jsonl(destination, [by_id[str(x["id"])] for x in rows if str(x["id"]) in by_id])
    def done(row, result, completed):
        by_id[str(row["id"])] = result; checkpoint(); print(f"{label} {completed}/{len(jobs)} sample={row['id']}", flush=True)
    try:
        run_bounded(jobs, worker, done, max_workers=concurrency, thread_name_prefix=label)
    except BoundedJobError as failure:
        checkpoint(); raise RuntimeError(f"{label}失败 sample={failure.job['id']}: {failure.error}") from failure.error
    checkpoint()
    return by_id


def _student_prompt(row: dict[str, Any], parent: dict[str, Any], trace: bool) -> str:
    schema = '{"signals":[...],"edits":[...],"output":"..."}' if trace else '{"output":"..."}'
    payload = {"current_input": str(row["source_text"]), "top8_history": _history(row), "parent": {"parent_id": str(parent["candidate_id"]), "text": str(parent["text"])}}
    return "Use the current input, retrieved history and Parent to produce the best supported personalized output. Return JSON only. REQUIRED_SCHEMA:\n" + schema + "\nPAYLOAD:\n" + json.dumps(payload, ensure_ascii=False) + "\nOUTPUT:\n"


def _report(rows, generated, audits, output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    examples, flat_signals, flat_edits, trace_records = [], [], [], []
    positive, negative = [], []
    for row in rows:
        g = generated[str(row["id"])]
        a = audits[str(row["id"])]
        signal_audit = {x["signal_id"]: x for x in a["signal_audits"]}
        signal_map = {x["signal_id"]: x for x in g["signals"]}
        parent_map = {str(x["candidate_id"]): x for x in row["candidates"]}
        audit_map = {x["parent_id"]: x for x in a["trace_audits"]}
        for signal in g["signals"]:
            audit = signal_audit[signal["signal_id"]]
            valid = audit["evidence_supported"] and audit["query_relevant"] and audit["confidence"] >= 0.7
            flat_signals.append({"sample_id": str(row["id"]), "signal": signal, "audit": audit, "valid": valid})
        for trace in g["traces"]:
            pid = trace["parent_id"]; ta = audit_map[pid]; parent = parent_map[pid]
            kept_edits = []
            used_signal_ids = set()
            task_fail = False
            for edit, ea in zip(trace["edits"], ta["edits"]):
                sid = edit["signal_id"]
                signal_valid = sid is None or (sid in signal_audit and signal_audit[sid]["evidence_supported"] and signal_audit[sid]["query_relevant"] and signal_audit[sid]["confidence"] >= 0.7)
                # 历史只能决定“如何表达”，不能授权引入当前Query不支持的内容。
                # 因此即使是profile-only edit，也必须先通过task-supported门控。
                task_ok = ea["task_supported"]
                profile_ok = edit["cause"] == "task" or (ea["profile_consistent"] is True and signal_valid)
                keep = task_ok and profile_ok
                flat_edits.append({"sample_id": str(row["id"]), "parent_id": pid, "edit": edit, "audit": ea, "signal_valid": signal_valid, "valid": keep})
                if keep:
                    # query_basis是Teacher审计字段，不蒸馏给Student。
                    kept_edits.append({key: edit[key] for key in ("type", "source", "target", "cause", "signal_id")})
                    if sid: used_signal_ids.add(sid)
                elif edit["cause"] == "task":
                    task_fail = True
            trace_ok = not task_fail and ta["changed_content_coverage"] >= 0.6
            if not trace_ok or not kept_edits:
                tier = "output_only"; compact_signals = []; kept_edits = []
            else:
                compact_signals = []
                for sid in sorted(used_signal_ids):
                    s = signal_map[sid]
                    # Student学习独立盲审重新归纳的局部行为，不能沿用可能与证据
                    # 不一致的生成器observation。
                    compact_signals.append({"id": sid, "dimension": s["dimension"], "evidence_ids": [e["id"] for e in s["evidence"]], "observation": signal_audit[sid]["inferred_observation"], "relevance": s["query_relevance"]})
                tier = "profile_trace" if compact_signals else "task_trace"
            if tier == "output_only":
                trace_text = ""; output_text = json.dumps({"output": str(row["target"])}, ensure_ascii=False, separators=(",", ":"))
            else:
                prefix = {"signals": compact_signals, "edits": kept_edits}
                encoded = json.dumps(prefix, ensure_ascii=False, separators=(",", ":"))
                trace_text = encoded[:-1] + ',"output":'; output_text = json.dumps(str(row["target"]), ensure_ascii=False) + "}"
            example = {"example_id": f"{row['id']}:{pid}:local-trace", "sample_id": str(row["id"]), "user_id": str(row.get("user_id", row["id"])), "operation_type": "mutation", "parent_a_id": pid, "parent_b_id": None, "sample_weight": 1.0 / max(len(row["candidates"]), 1), "supervision_tier": tier, "output": str(row["target"]), "prompt": _student_prompt(row, parent, tier != "output_only"), "trace_text": trace_text, "output_text": output_text, "quality_audit": ta}
            examples.append(example)
            rec = {"sample_id": str(row["id"]), "query": str(row["source_text"]), "parent": str(parent["text"]), "gold": str(row["target"]), "signals": g["signals"], "trace": trace, "audit": ta, "tier": tier, "compact_signals": compact_signals, "kept_edits": kept_edits}
            trace_records.append(rec)
            if tier == "profile_trace" and len(positive) < 3: positive.append(rec)
            if any(e["edit"]["cause"] != "task" and not e["valid"] for e in flat_edits if e["sample_id"] == str(row["id"]) and e["parent_id"] == pid) and len(negative) < 3: negative.append(rec)
    write_jsonl(output_dir / "compact_student_sft.jsonl", examples)
    write_json(output_dir / "positive_cases.json", positive); write_json(output_dir / "negative_cases.json", negative)
    profile_edits = [x for x in flat_edits if x["edit"]["cause"] != "task"]
    task_edits = [x for x in flat_edits if x["edit"]["cause"] != "profile"]
    report = {
        "queries": len(rows), "parents": len(trace_records), "teacher_model": config["teacher"]["model"],
        "signals": {"total": len(flat_signals), "queries_with_signal": sum(bool(generated[str(r["id"])]["signals"]) for r in rows), "blind_evidence_supported_rate": sum(x["audit"]["evidence_supported"] for x in flat_signals) / max(len(flat_signals), 1), "blind_query_relevant_rate": sum(x["audit"]["query_relevant"] for x in flat_signals) / max(len(flat_signals), 1), "fully_valid": sum(x["valid"] for x in flat_signals)},
        "edits": {"total": len(flat_edits), "all_edit_task_supported_rate": sum(x["audit"]["task_supported"] for x in flat_edits) / max(len(flat_edits), 1), "task_or_both": len(task_edits), "task_supported_rate": sum(x["audit"]["task_supported"] for x in task_edits) / max(len(task_edits), 1), "profile_or_both": len(profile_edits), "profile_consistent_rate": sum(x["audit"]["profile_consistent"] is True for x in profile_edits) / max(len(profile_edits), 1), "fully_valid_profile_edits": sum(x["valid"] for x in profile_edits)},
        # 覆盖率只查看编辑字段，不能把完整Gold output放进被搜索文本，否则会
        # 产生接近自证循环的虚高覆盖率。
        "trace": {"alignment_rejected_queries": sum(bool(generated[str(r["id"])].get("quality_rejected")) for r in rows), "judge_rejected_queries": sum(bool(audits[str(r["id"])].get("quality_rejected")) for r in rows), "mean_blind_coverage": statistics.mean(x["audit"]["changed_content_coverage"] for x in trace_records), "mean_automatic_changed_token_coverage": statistics.mean(_coverage(x["parent"], x["gold"], json.dumps(x["trace"]["edits"], ensure_ascii=False)) for x in trace_records), "mean_kept_edit_changed_token_coverage": statistics.mean(_coverage(x["parent"], x["gold"], json.dumps(x["kept_edits"], ensure_ascii=False)) for x in trace_records), "teacher_audit_field_meta_mentions": sum(bool(FORBIDDEN.search(json.dumps(x["trace"], ensure_ascii=False))) for x in trace_records), "student_trace_leakage": sum(bool(FORBIDDEN.search(x["trace_text"])) for x in examples), "tiers": {tier: sum(x["tier"] == tier for x in trace_records) for tier in ("profile_trace", "task_trace", "output_only")}},
        "limitations": ["同一qwen3-32b以不同盲审Prompt担任生成器与Judge", "Top-8只表示Query相关局部信号，不证明全局稳定用户偏好"],
    }
    write_json(output_dir / "quality_report.json", report)
    s, e, t = report["signals"], report["edits"], report["trace"]
    lines = ["# Top-8 Local Profile Trace Pilot", "", f"- Query：{report['queries']}；Parent：{report['parents']}；模型：`{report['teacher_model']}`。", "", "| 指标 | 结果 |", "|---|---:|", f"| 生成局部信号 | {s['total']} |", f"| 含信号Query | {s['queries_with_signal']}/{report['queries']} |", f"| 盲审证据支持率 | {s['blind_evidence_supported_rate']:.1%} |", f"| 盲审Query相关率 | {s['blind_query_relevant_rate']:.1%} |", f"| 完全有效信号 | {s['fully_valid']}/{s['total']} |", f"| Trace对齐拒绝Query | {t['alignment_rejected_queries']} |", f"| Judge拒绝Query | {t['judge_rejected_queries']} |", f"| 全部Edit的Query支持率 | {e['all_edit_task_supported_rate']:.1%} |", f"| Task/Task+Profile edit支持率 | {e['task_supported_rate']:.1%} |", f"| Profile edit一致率 | {e['profile_consistent_rate']:.1%} |", f"| 完全有效Profile edit | {e['fully_valid_profile_edits']}/{e['profile_or_both']} |", f"| 盲审变化覆盖率 | {t['mean_blind_coverage']:.3f} |", f"| 原始Edit Token覆盖率 | {t['mean_automatic_changed_token_coverage']:.3f} |", f"| 最终保留Edit Token覆盖率 | {t['mean_kept_edit_changed_token_coverage']:.3f} |", f"| Profile/Task/Output-only | {t['tiers']['profile_trace']}/{t['tiers']['task_trace']}/{t['tiers']['output_only']} |", f"| Teacher审计字段元答案措辞 | {t['teacher_audit_field_meta_mentions']} |", f"| Student Trace泄漏 | {t['student_trace_leakage']} |", "", "## 边界", "", "- 该实验评价局部、Query相关的历史信号，不声称得到稳定全局用户偏好。", "- Teacher与Judge为同一模型，结果仍需人工抽查正反Case。"]
    (output_dir / "quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run(config: dict[str, Any], count: int, concurrency: int, retry_rejected: bool = False, rerun_judge: bool = False, retry_judge: bool = False) -> dict[str, Any]:
    source = resolve_path("dataset/editor_sets/validated_trace_blind_quality_pilot_20/selected_raw_traces.jsonl")
    rows = read_jsonl(source)[:count]
    output_dir = resolve_path(f"dataset/editor_sets/local_profile_trace_pilot_{count}"); output_dir.mkdir(parents=True, exist_ok=True)
    signal_client = _client(config, output_dir / "signal_cache", 3500)
    signals = _run_stage(rows, output_dir / "01_local_signals.jsonl", lambda r: _build_signals(r, signal_client), concurrency, "local-signals")
    align_path = output_dir / "02_aligned_traces.jsonl"
    retry_ids: set[str] = set()
    if retry_rejected and align_path.exists():
        retry_ids = {str(row["id"]) for row in read_jsonl(align_path) if row.get("quality_rejected")}
    align_client = _client(config, output_dir / "align_cache", 5000)
    generated = _run_stage(rows, align_path, lambda r: _align_traces(r, signals[str(r["id"])], align_client), concurrency, "trace-align", retry_rejected=retry_rejected)
    judge_path = output_dir / "03_blind_audits.jsonl"
    judge_client = _client(config, output_dir / "judge_cache", 5000)
    if rerun_judge:
        judge_force_ids = {str(row["id"]) for row in rows}
    elif retry_judge and judge_path.exists():
        judge_force_ids = {str(row["id"]) for row in read_jsonl(judge_path) if row.get("quality_rejected")}
    else:
        judge_force_ids = retry_ids
    audits = _run_stage(rows, judge_path, lambda r: _judge(r, generated[str(r["id"])], judge_client), concurrency, "blind-judge", force_ids=judge_force_ids)
    report = _report(rows, generated, audits, output_dir, config)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-8 Local Profile Trace质量Pilot")
    parser.add_argument("--config", default=str(HERE / "config.yaml")); parser.add_argument("--count", type=int, default=20); parser.add_argument("--concurrency", type=int, default=4); parser.add_argument("--retry-rejected", action="store_true"); parser.add_argument("--rerun-judge", action="store_true"); parser.add_argument("--retry-judge", action="store_true")
    args = parser.parse_args(); run(load_config(args.config), args.count, args.concurrency, args.retry_rejected, args.rerun_judge, args.retry_judge)


if __name__ == "__main__": main()
