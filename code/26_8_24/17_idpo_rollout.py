"""阶段 17：从当前 Editor 策略采样用户级 on-policy responses。

每个 response 都来自同一个 Prompt 的随机采样，只有这样的候选才能形成
合法 DPO pair。Teacher seed 只提供 Parent，不能代替当前 Editor 的 rollout。
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from pipeline_common import (  # noqa: E402
    build_editor_prompt,
    choose_crossover_pairs,
    load_config,
    load_project_stage,
    read_jsonl,
    resolve_path,
    stage_path,
    validate_crossover,
    validate_mutation,
    visible_history,
    write_json,
    write_jsonl,
)
from idpo_common import canonical_response_parts, idpo_path  # noqa: E402


def _limit_rows(rows: list[dict[str, Any]], settings: dict[str, Any]) -> list[dict[str, Any]]:
    users: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        users.setdefault(str(row.get("user_id", row.get("parent_sample_id", ""))), []).append(row)
    ordered_users = sorted(users)
    user_limit = int(settings.get("user_limit", 0))
    if user_limit > 0:
        ordered_users = ordered_users[:user_limit]
    per_user = int(settings.get("pseudo_queries_per_user", 0))
    selected = []
    for user_id in ordered_users:
        values = sorted(users[user_id], key=lambda item: str(item["id"]))
        selected.extend(values[:per_user] if per_user > 0 else values)
    return selected


def _mock_results(
    prompt: str,
    operation_type: str,
    parent_a: dict,
    parent_b: dict | None,
    count: int,
    trace_aware: bool = False,
    simple_trace_aware: bool = False,
) -> list[tuple[dict, str, None]]:
    prompt_payload = json.loads(
        prompt.split("PAYLOAD:\n", 1)[1].rsplit("\n\nOUTPUT:\n", 1)[0]
    )
    history_ids = [str(item["id"]) for item in prompt_payload["retrieved_history"]]
    matched_span = str(prompt_payload["current_input"])[:80].strip()
    values = []
    for index in range(count):
        if simple_trace_aware:
            payload = {
                "evidence_ids": history_ids[:1],
                "edit_reason": "The selected history supports a concise personalized revision that applies to the current input.",
                "edit_action": f"Apply the supported edit variant {index + 1}.",
                "output": f"{parent_a['text']} Variant {index + 1}",
            }
        elif trace_aware:
            payload = {
                "history_analysis": {"evidence_ids": history_ids[:2]},
                "preference": {
                    "condition": "When the input presents a named method for a focused task.",
                    "preferred_behavior": "Name the task and the method directly in the output.",
                },
                "applicability": {
                    "decision": "apply",
                    "matched_input_span": matched_span,
                },
                "edit_plan": [f"Apply personalized edit variant {index + 1}."],
                "output": f"{parent_a['text']} Variant {index + 1}",
            }
        elif operation_type == "mutation":
            payload = {
                "decision": "revise",
                "task_correction": "Preserve the central contribution and improve the title wording.",
                "profile_signal": {"evidence_ids": [], "observation": "No reliable user-specific change is required in this smoke rollout."},
                "edit_action": f"Apply the current editing policy variant {index + 1}.",
                "output": f"{parent_a['text']} Variant {index + 1}",
            }
        else:
            payload = {
                "decision": "merge",
                "profile_signal": {"evidence_ids": [], "observation": "No reliable user-specific change is required in this smoke rollout."},
                "merge_action": f"Combine the parent wording using policy variant {index + 1}.",
                "output": f"{parent_a['text']} Blend {index + 1}",
            }
        values.append((payload, json.dumps(payload, ensure_ascii=False), None))
    return values


def _editor_for_user(config: dict[str, Any], user_id: str):
    """按轮次加载全局策略或该用户上一轮 Adapter。"""

    settings = config["idpo"]
    adapter_root = str(settings.get("policy_adapter_root", "")).strip()
    editor_module = load_project_stage(
        "code/26_8_24/07_generate_editor_pool.py", "idpo_local_editor"
    )
    local = copy.deepcopy(config)
    # 阶段一 simple_conditional_trace 的正式候选生成是 output-only，但
    # Trace-aware IDPO 需要严格保留结构化响应；只在本地解析器层切换为
    # strict mode，不改变已经构造好的 rollout Prompt。
    if str(settings.get("rollout_response_mode", "")) in {
        "conditional_preference_trace",
        "simple_conditional_trace",
    }:
        local.setdefault("sft_data", {})["supervision_mode"] = "gold_aware_trace"
    # Rollout 与后续 DPO 必须使用同一个上下文长度口径。旧实现
    # 遗漏了这一传递，导致 IDPO max_length=1024 时本地 Editor
    # 仍按 SFT 的2048 token生成，无谓增加 KV/activation 显存。
    local.setdefault("training", {})["max_length"] = int(
        settings.get("max_length", local.get("training", {}).get("max_length", 1024))
    )
    # 常规候选生成使用 greedy decoding；IDPO 必须从同一个 Prompt 随机采样，
    # 才能获得来自当前策略分布的不同 chosen/rejected response。
    inference = local.setdefault("inference", {})
    inference["do_sample"] = bool(settings.get("rollout_do_sample", True))
    inference["temperature"] = float(settings.get("rollout_temperature", 0.8))
    inference["top_p"] = float(settings.get("rollout_top_p", 0.95))
    # A rollout query is expanded to ``rollout_samples`` identical prompts.
    # The query batch multiplier lets several independent queries share one
    # forward/generation batch while preserving the same-prompt DPO contract.
    inference["batch_size"] = int(
        settings.get("rollout_batch_size", settings.get("rollout_samples", 1))
        * max(1, int(settings.get("rollout_query_batch_size", 1)))
    )
    if not inference["do_sample"] and int(settings.get("rollout_samples", 1)) > 1:
        raise ValueError("IDPO rollout_samples>1 时 rollout_do_sample 必须为 true")
    if adapter_root:
        adapter_path = resolve_path(adapter_root) / f"user_{user_id}"
        if not adapter_path.exists():
            raise FileNotFoundError(f"缺少上一轮 user Editor adapter: {adapter_path}")
        local.setdefault("model", {})["adapter_path"] = str(adapter_path)
    return editor_module.LocalEditor(local)


def _sample_prompt(editor, config: dict, prompt: str, count: int):
    if bool(config["idpo"].get("mock_editor", False)):
        return None
    # LocalEditor 的批量接口按 prompt 数量返回一条 response；重复同一 Prompt
    # 才能保证 chosen/rejected 共享完全相同的条件。
    return editor.generate_many([prompt] * count)


def _validate_samples(
    raw_values: list[tuple[dict[str, Any] | None, str, Exception | None]],
    operation_type: str,
    parent_a: dict[str, Any],
    parent_b: dict[str, Any] | None,
    evidence: set[str],
    output_only: bool = False,
    trace_aware: bool = False,
    current_input: str = "",
    simple_trace_aware: bool = False,
) -> list[dict[str, Any]]:
    from pipeline_common import normalized_text

    valid = []
    seen = set()
    for payload, raw, error in raw_values:
        if error is not None or payload is None:
            continue
        try:
            value = dict(payload)
            if simple_trace_aware:
                raw_ids = value.get("evidence_ids")
                if not isinstance(raw_ids, list):
                    raise ValueError("evidence_ids必须为list")
                evidence_ids = []
                for item in raw_ids:
                    history_id = str(item).strip()
                    if history_id and history_id not in evidence_ids:
                        evidence_ids.append(history_id)
                if not 1 <= len(evidence_ids) <= 2:
                    raise ValueError("简化Trace必须引用1到2条历史")
                if set(evidence_ids) - evidence:
                    raise ValueError("简化Trace引用了不可见历史ID")
                reason = " ".join(str(value.get("edit_reason", "")).strip().split())
                action = " ".join(str(value.get("edit_action", "")).strip().split())
                if not reason or len(reason) > 600:
                    raise ValueError("edit_reason无效")
                if not action or len(action) > 600:
                    raise ValueError("edit_action无效")
                output = str(value.get("output", "")).strip().strip('"').strip()
                if not output or "\n" in output or len(output) > 300:
                    raise ValueError("简化Trace output无效")
                clean = {
                    "evidence_ids": evidence_ids,
                    "edit_reason": reason,
                    "edit_action": action,
                    "output": output,
                }
            elif trace_aware:
                history_analysis = value.get("history_analysis")
                preference = value.get("preference")
                applicability = value.get("applicability")
                plans = value.get("edit_plan")
                if not isinstance(history_analysis, dict):
                    raise ValueError("history_analysis必须为object")
                evidence_ids = [
                    str(item).strip()
                    for item in history_analysis.get("evidence_ids", [])
                    if str(item).strip()
                ]
                if not 2 <= len(evidence_ids) <= 4 or len(set(evidence_ids)) != len(evidence_ids):
                    raise ValueError("Trace必须引用2到4条不同历史")
                if set(evidence_ids) - evidence:
                    raise ValueError("Trace引用了不可见历史ID")
                if not isinstance(preference, dict):
                    raise ValueError("preference必须为object")
                condition = str(preference.get("condition", "")).strip()
                behavior = str(preference.get("preferred_behavior", "")).strip()
                if not condition or len(condition) > 400:
                    raise ValueError("preference.condition无效")
                if not behavior or len(behavior) > 400:
                    raise ValueError("preference.preferred_behavior无效")
                if not isinstance(applicability, dict) or str(
                    applicability.get("decision", "")
                ).strip().lower() != "apply":
                    raise ValueError("applicability.decision必须为apply")
                matched = str(applicability.get("matched_input_span", "")).strip()
                if not matched or len(matched) > 700:
                    raise ValueError("matched_input_span无效")
                if normalized_text(matched) not in normalized_text(current_input):
                    raise ValueError("matched_input_span不来自当前输入")
                if not isinstance(plans, list) or not 1 <= len(plans) <= 2:
                    raise ValueError("edit_plan必须包含1到2条操作")
                clean_plans = [str(item).strip() for item in plans]
                if any(not item or len(item) > 400 for item in clean_plans):
                    raise ValueError("edit_plan包含无效操作")
                output = str(value.get("output", "")).strip().strip('"').strip()
                if not output or "\n" in output or len(output) > 300:
                    raise ValueError("Trace response output无效")
                clean = {
                    "history_analysis": {"evidence_ids": evidence_ids},
                    "preference": {
                        "condition": condition,
                        "preferred_behavior": behavior,
                    },
                    "applicability": {
                        "decision": "apply",
                        "matched_input_span": matched,
                    },
                    "edit_plan": clean_plans,
                    "output": output,
                }
            elif output_only:
                output = str(value.get("output", "")).strip().strip('"').strip()
                if not output or "\n" in output or len(output) > 300:
                    raise ValueError("output-only response无效")
                clean = {"output": output}
            elif operation_type == "mutation":
                value["parent_id"] = str(parent_a["candidate_id"])
                clean = validate_mutation(value, parent_a, evidence)
                task_correction = str(value.get("task_correction", "")).strip()
                if not task_correction or len(task_correction) > 300:
                    raise ValueError("Mutation task_correction 必须是简短非空文本")
                clean["task_correction"] = task_correction
            else:
                value["parent_a_id"] = str(parent_a["candidate_id"])
                value["parent_b_id"] = str(parent_b["candidate_id"])
                clean = validate_crossover(value, parent_a, parent_b, evidence)
            output_key = normalized_text(clean["output"])
            if not output_key or output_key in seen:
                continue
            seen.add(output_key)
            trace_text, output_text, response_text = canonical_response_parts(
                operation_type, clean
            )
            valid.append(
                {
                    "response_id": f"r{len(valid)}",
                    "output": clean["output"],
                    "response_text": response_text,
                    "trace_text": trace_text,
                    "output_text": output_text,
                    "trace": clean,
                    "raw_response": raw,
                }
            )
        except Exception:
            continue
    return valid


def _build_rollout(row: dict[str, Any], config: dict, editor) -> list[dict[str, Any]]:
    results = []
    for spec in _build_operation_specs(row, config):
        settings = config["idpo"]
        sample_count = int(settings["rollout_samples"])
        if bool(settings.get("mock_editor", False)):
            raw_values = _mock_results(
                spec["prompt"], spec["operation_type"], spec["parent_a"],
                spec["parent_b"], sample_count,
                bool(spec.get("trace_aware", False)),
                bool(spec.get("simple_trace_aware", False)),
            )
        else:
            raw_values = _sample_prompt(editor, config, spec["prompt"], sample_count)
        result = _finalize_rollout(spec, raw_values, config)
        if (
            bool(
                spec.get("trace_aware", False)
                or spec.get("simple_trace_aware", False)
            )
            and not result["minimum_responses_met"]
            and bool(settings.get("trace_fallback_to_output_only", True))
        ):
            fallback = _fallback_spec(spec)
            if bool(settings.get("mock_editor", False)):
                raw_values = _mock_results(
                    fallback["prompt"], fallback["operation_type"],
                    fallback["parent_a"], fallback["parent_b"], sample_count, False,
                )
            else:
                raw_values = _sample_prompt(editor, config, fallback["prompt"], sample_count)
            result = _finalize_rollout(fallback, raw_values, config)
        results.append(result)
    return results


def _build_operation_specs(row: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build target-blind rollout requests without calling the model."""
    settings = config["idpo"]
    maximum_history = int(settings["maximum_history_records"])
    trace_history = int(settings.get("trace_maximum_history_records", maximum_history))
    parents = list(row.get("candidates", []))
    if not parents:
        return []
    enabled = {
        str(value).strip().lower()
        for value in settings.get("rollout_operations", ["mutation", "crossover"])
    }
    unknown = enabled - {"mutation", "crossover"}
    if unknown:
        raise ValueError(f"未知 IDPO rollout_operations={sorted(unknown)}")
    operations = []
    if "mutation" in enabled:
        mutation_parents = max(1, int(settings.get("rollout_mutation_parents", 2)))
        operations.extend(
            ("mutation", parent, None) for parent in parents[:mutation_parents]
        )
    if "crossover" in enabled:
        if len(parents) < 2:
            raise ValueError(f"sample={row['id']} 少于两个 Parent，不能执行 Crossover")
        crossover_count = max(1, int(settings.get("rollout_crossovers", 1)))
        operations.extend(
            ("crossover", left, right)
            for left, right in choose_crossover_pairs(parents, crossover_count)
        )
    if not operations:
        raise ValueError("IDPO rollout_operations 不能为空")
    sample_count = int(settings["rollout_samples"])
    supervision_mode = str(config.get("sft_data", {}).get(
        "supervision_mode", "gold_aware_trace"
    ))
    response_mode = str(settings.get("rollout_response_mode", "")).strip()
    if not response_mode:
        response_mode = (
            "output_only"
            if supervision_mode in {
                "conditional_preference_trace", "simple_conditional_trace"
            }
            else "gold_aware_trace"
        )
    if response_mode not in {
        "output_only", "gold_aware_trace", "conditional_preference_trace",
        "simple_conditional_trace",
    }:
        raise ValueError(f"未知 idpo.rollout_response_mode={response_mode}")
    rollout_prompt_mode = {
        "conditional_preference_trace": "conditional_preference_trace_idpo",
        "simple_conditional_trace": "simple_conditional_trace_idpo",
    }.get(response_mode, response_mode)
    trace_response = response_mode in {
        "conditional_preference_trace", "simple_conditional_trace"
    }
    active_history_count = (
        trace_history
        if trace_response
        else maximum_history
    )
    history = visible_history(row, active_history_count)
    evidence = {str(item["id"]) for item in history}
    fallback_history = visible_history(row, maximum_history)
    fallback_evidence = {str(item["id"]) for item in fallback_history}
    specs = []
    for operation_type, parent_a, parent_b in operations:
        prompt = build_editor_prompt(
            row,
            operation_type,
            parent_a,
            parent_b,
            active_history_count,
            supervision_mode=rollout_prompt_mode,
            history_input_max_chars=int(settings.get("history_input_max_chars", 0)),
            history_output_max_chars=int(settings.get("history_output_max_chars", 0)),
        )
        fallback_prompt = build_editor_prompt(
            row,
            operation_type,
            parent_a,
            parent_b,
            maximum_history,
            supervision_mode="output_only",
            history_input_max_chars=int(settings.get("history_input_max_chars", 0)),
            history_output_max_chars=int(settings.get("history_output_max_chars", 0)),
        )
        specs.append({
            "rollout_id": f"{row['id']}:{operation_type}:{parent_a['candidate_id']}",
            "user_id": str(row.get("user_id", row.get("parent_sample_id", ""))),
            "pseudo_query_id": str(row["id"]),
            "operation_type": operation_type,
            "prompt": prompt,
            "current_input": str(row["source_text"]),
            "retrieved_history": history,
            "parent_a": parent_a,
            "parent_b": parent_b,
            "evidence": evidence,
            "output_only": rollout_prompt_mode == "output_only",
            "trace_aware": response_mode == "conditional_preference_trace",
            "simple_trace_aware": response_mode == "simple_conditional_trace",
            "trace_fallback": False,
            "fallback_prompt": fallback_prompt,
            "fallback_history": fallback_history,
            "fallback_evidence": fallback_evidence,
        })
    return specs


def _fallback_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """把失败的完整 Trace 请求降级为与原实验相同的 output-only 请求。"""

    fallback = dict(spec)
    fallback["prompt"] = str(spec["fallback_prompt"])
    fallback["retrieved_history"] = list(spec["fallback_history"])
    fallback["evidence"] = set(spec["fallback_evidence"])
    fallback["output_only"] = True
    fallback["trace_aware"] = False
    fallback["simple_trace_aware"] = False
    fallback["trace_fallback"] = True
    return fallback


def _finalize_rollout(
    spec: dict[str, Any],
    raw_values: list[tuple[dict[str, Any] | None, str, Exception | None]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate one operation's responses after a shared model batch."""
    settings = config["idpo"]
    valid = _validate_samples(
        raw_values,
        spec["operation_type"],
        spec["parent_a"],
        spec["parent_b"],
        spec["evidence"],
        bool(spec.get("output_only", False)),
        bool(spec.get("trace_aware", False)),
        str(spec.get("current_input", "")),
        bool(spec.get("simple_trace_aware", False)),
    )
    parent_a = spec["parent_a"]
    parent_b = spec["parent_b"]
    return {
        "rollout_id": spec["rollout_id"],
        "user_id": spec["user_id"],
        "pseudo_query_id": spec["pseudo_query_id"],
        "operation_type": spec["operation_type"],
        "prompt": spec["prompt"],
        "current_input": spec["current_input"],
        "retrieved_history": spec["retrieved_history"],
        "parent_a": {"candidate_id": str(parent_a["candidate_id"]), "text": str(parent_a["text"])},
        "parent_b": None if parent_b is None else {
            "candidate_id": str(parent_b["candidate_id"]), "text": str(parent_b["text"])
        },
        "responses": valid,
        "trace_aware": bool(
            spec.get("trace_aware", False) or spec.get("simple_trace_aware", False)
        ),
        "trace_mode": (
            "simple" if spec.get("simple_trace_aware", False)
            else "conditional" if spec.get("trace_aware", False)
            else "output_only"
        ),
        "trace_fallback": bool(spec.get("trace_fallback", False)),
        "minimum_responses_met": len(valid) >= int(settings["minimum_valid_responses"]),
        "gold_visible_during_rollout": False,
        "hidden_target_used": False,
    }


def _run_operation_batch(
    specs: list[dict[str, Any]], editor, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Generate several operations in one batch and restore per-operation groups."""
    settings = config["idpo"]
    sample_count = int(settings["rollout_samples"])
    if bool(settings.get("mock_editor", False)):
        return [
            _finalize_rollout(
                spec,
                _mock_results(
                    spec["prompt"], spec["operation_type"], spec["parent_a"],
                    spec["parent_b"], sample_count,
                    bool(spec.get("trace_aware", False)),
                    bool(spec.get("simple_trace_aware", False)),
                ),
                config,
            )
            for spec in specs
        ]
    prompts = [spec["prompt"] for spec in specs for _ in range(sample_count)]
    raw = editor.generate_many(prompts)
    expected = len(specs) * sample_count
    if len(raw) != expected:
        raise RuntimeError(f"Editor batch returned {len(raw)} responses; expected {expected}")
    results = [
        _finalize_rollout(
            spec,
            raw[index * sample_count : (index + 1) * sample_count],
            config,
        )
        for index, spec in enumerate(specs)
    ]
    fallback_indices = [
        index
        for index, (spec, result) in enumerate(zip(specs, results))
        if bool(
            spec.get("trace_aware", False)
            or spec.get("simple_trace_aware", False)
        )
        and not result["minimum_responses_met"]
        and bool(settings.get("trace_fallback_to_output_only", True))
    ]
    if fallback_indices:
        fallback_specs = [_fallback_spec(specs[index]) for index in fallback_indices]
        fallback_results = _run_operation_batch(fallback_specs, editor, config)
        for index, result in zip(fallback_indices, fallback_results):
            results[index] = result
    return results


def run(config: dict, split: str = "validation") -> dict[str, Any]:
    settings = config["idpo"]
    adaptation_split = f"adaptation_{split}"
    source = read_jsonl(stage_path(config, adaptation_split, "seeds"))
    source = _limit_rows(source, settings)
    output_path = idpo_path(config, int(settings["round"]), f"{split}_rollouts.jsonl")
    existing = read_jsonl(output_path) if bool(settings.get("resume_existing", True)) and output_path.exists() else []
    done = {str(row["rollout_id"]): row for row in existing}
    existing_counts: dict[str, int] = {}
    for item in existing:
        key = str(item.get("pseudo_query_id", ""))
        existing_counts[key] = existing_counts.get(key, 0) + 1
    user_ids = sorted({str(row.get("user_id", row.get("parent_sample_id", ""))) for row in source})
    mock = bool(settings.get("mock_editor", False))
    editor = None
    reports = {
        "split": split,
        "round": int(settings["round"]),
        "queries": len(source),
        "users": len(user_ids),
        "rollouts": 0,
        "valid_rollouts": 0,
        "responses": 0,
        "policy": (
            "mock"
            if mock
            else ("per_user_previous_round" if settings.get("policy_adapter_root") else "global_sft_adapter")
        ),
        "response_mode": str(settings.get("rollout_response_mode", "legacy_default")),
        "sampling": {
            "do_sample": bool(settings.get("rollout_do_sample", True)),
            "temperature": float(settings.get("rollout_temperature", 0.8)),
            "top_p": float(settings.get("rollout_top_p", 0.95)),
            "samples_per_prompt": int(settings["rollout_samples"]),
            "query_batch_size": max(1, int(settings.get("rollout_query_batch_size", 1))),
            "effective_generation_batch": (
                int(settings["rollout_samples"])
                * max(1, int(settings.get("rollout_query_batch_size", 1)))
            ),
        },
    }
    current_user = None
    query_batch_size = max(1, int(settings.get("rollout_query_batch_size", 1)))
    adapter_policy = bool(str(settings.get("policy_adapter_root", "")).strip())
    try:
        index = 0
        position = 0
        while position < len(source):
            row = source[position]
            user_id = str(row.get("user_id", row.get("parent_sample_id", "")))
            expected_rollouts = len(_build_operation_specs(row, config))
            if existing_counts.get(str(row["id"]), 0) >= expected_rollouts:
                index += 1
                position += 1
                print(f"IDPO rollout {index}/{len(source)} user={user_id} source=resume", flush=True)
                continue
            if not mock and (editor is None or (adapter_policy and user_id != current_user)):
                if editor is not None:
                    del editor
                    gc.collect()
                    if __import__("torch").cuda.is_available():
                        __import__("torch").cuda.empty_cache()
                editor = _editor_for_user(config, user_id)
                current_user = user_id
            batch_rows = []
            while position + len(batch_rows) < len(source) and len(batch_rows) < query_batch_size:
                candidate_row = source[position + len(batch_rows)]
                candidate_user = str(candidate_row.get("user_id", candidate_row.get("parent_sample_id", "")))
                if adapter_policy and candidate_user != user_id:
                    break
                if existing_counts.get(str(candidate_row["id"]), 0) >= len(_build_operation_specs(candidate_row, config)):
                    break
                batch_rows.append(candidate_row)
            specs = [spec for batch_row in batch_rows for spec in _build_operation_specs(batch_row, config)]
            for rollout in _run_operation_batch(specs, editor, config):
                done[str(rollout["rollout_id"])] = rollout
            for batch_row in batch_rows:
                index += 1
                position += 1
                print(f"IDPO rollout {index}/{len(source)} user={batch_row.get('user_id', batch_row.get('parent_sample_id', ''))}", flush=True)
            if index % int(settings.get("checkpoint_every", 10)) == 0:
                write_jsonl(output_path, [done[key] for key in sorted(done)])
    finally:
        # OOM/API中断也保存距上一个周期断点之后的已完成 rollout。
        if done:
            write_jsonl(output_path, [done[key] for key in sorted(done)])
        if editor is not None:
            del editor
    ordered = [done[key] for key in sorted(done)]
    write_jsonl(output_path, ordered)
    reports["rollouts"] = len(ordered)
    reports["valid_rollouts"] = sum(bool(item["minimum_responses_met"]) for item in ordered)
    reports["responses"] = sum(len(item["responses"]) for item in ordered)
    reports["trace_rollouts"] = sum(bool(item.get("trace_aware")) for item in ordered)
    reports["trace_fallback_rollouts"] = sum(bool(item.get("trace_fallback")) for item in ordered)
    reports["unique_users"] = len({str(item["user_id"]) for item in ordered})
    report_path = idpo_path(config, int(settings["round"]), f"{split}_rollout_report.json")
    write_json(report_path, reports)
    print(f"IDPO rollouts -> {output_path}; report={reports}")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="17 - IDPO on-policy Editor rollout")
    parser.add_argument("--config", default=str(HERE / "config_idpo.yaml"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    run(load_config(args.config), args.split)


if __name__ == "__main__":
    main()
