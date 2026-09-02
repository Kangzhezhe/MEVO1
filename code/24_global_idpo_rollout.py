"""全局 IDPO 的遗传式 on-policy rollout。

Mutation 由当前 SFT Editor 采样，响应采用半结构化 CoT，并要求一次局部编辑，
对应 EDIT 的“关键编辑步骤”思想。Crossover 由 Teacher 直接对 Mutation 子代做
sequence-level 融合；Gold 只在离线选择阶段出现，默认 crossover 也进入全局 DPO
偏好池，形成完整的 Mutation -> Crossover -> Selection -> Update 迭代。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from idpo_common import canonical_response_parts, idpo_path  # noqa: E402
from pipeline_common import build_configured_editor_prompt, load_config, read_jsonl, stage_path, teacher_client, visible_history, write_jsonl  # noqa: E402


def _editor(config: dict[str, Any]):
    module_path = HERE / "07_generate_editor_pool.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("global_rollout_editor", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    local = copy.deepcopy(config)
    settings = local["global_idpo"]
    rollout_mode = str(settings.get("rollout_response_mode", "trace"))
    output_only = rollout_mode in {"output_only", "plain_output_only"}
    plain_output_only = rollout_mode == "plain_output_only"
    local.setdefault("sft_data", {})["supervision_mode"] = (
        "plain_output_only" if plain_output_only else ("output_only" if output_only else "simple_conditional_trace")
    )
    local.setdefault("training", {})["max_length"] = int(settings.get("max_length", 2048))
    local.setdefault("inference", {}).update({
        "do_sample": True,
        "temperature": float(settings.get("temperature", 0.8)),
        "top_p": float(settings.get("top_p", 0.95)),
        # ``rollout_samples`` is the number of candidates per Query, not the
        # number of sequences that must occupy GPU memory simultaneously.
        # Keeping these coupled made an 8x long-context batch OOM on 30GB cards.
        "batch_size": int(settings.get("rollout_batch_size", 1)),
        "max_new_tokens": int(settings.get("max_new_tokens", 128)),
    })
    return module.LocalEditor(local)


def _compact_payload(
    value: Any,
    parent: dict[str, Any],
    history: list[dict[str, str]],
    output_only: bool = False,
    allow_long_output: bool = False,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    output = " ".join(str(value.get("output", "")).strip().split())
    if not output or "\n" in output or (not allow_long_output and len(output) > 300):
        return None
    if output_only:
        return {"output": output}
    visible = {str(item["id"]) for item in history}
    evidence = []
    raw_ids = value.get("evidence_ids", [])
    if isinstance(raw_ids, list):
        evidence = [str(item).strip() for item in raw_ids if str(item).strip() in visible][:2]
    reason = " ".join(str(value.get("edit_reason", "")).strip().split())[:600].strip()
    action = " ".join(str(value.get("edit_action", "")).strip().split())[:600].strip()
    if not reason:
        reason = "Use a concise local edit that preserves the parent contribution."
    if not action:
        action = "Apply one concrete wording edit supported by the current input."
    return {"evidence_ids": evidence, "edit_reason": reason, "edit_action": action, "output": output}


def _teacher_crossover_prompt(row: dict[str, Any], left: dict[str, Any], right: dict[str, Any], history: list[dict[str, str]]) -> str:
    payload = {
        "current_input": str(row["source_text"]),
        "retrieved_history": history,
        "parent_a": {"id": str(left["candidate_id"]), "text": str(left["text"])},
        "parent_b": {"id": str(right["candidate_id"]), "text": str(right["text"])},
    }
    return (
        "You are a teacher performing sequence-level crossover for a personalized editor. "
        "Fuse the two complete parent sequences into one coherent final title. Preserve "
        "the factual contribution of the better parent, and borrow useful wording from "
        "the other parent only when it fits the current input. Use visible history as "
        "optional evidence. Do not mention a gold answer or metric. Return JSON only.\n\n"
        'REQUIRED_SCHEMA: {"evidence_ids":["..."],"edit_reason":"...",'
        '"edit_action":"...","output":"..."}\n\nPAYLOAD:\n'
        + json.dumps(payload, ensure_ascii=False)
        + "\n\nOUTPUT:\n"
    )


def run(config: dict[str, Any], split: str = "train") -> dict[str, Any]:
    settings = config["global_idpo"]
    plain_output_only = str(settings.get("rollout_response_mode", "trace")) == "plain_output_only"
    base_text_protocol = str(config.get("sft_data", {}).get("prompt_protocol", "")) == "base_text"
    rows = read_jsonl(stage_path(config, "adaptation_train", "seeds"))
    limit = int(settings.get("user_limit", 0))
    if limit > 0:
        users = sorted({str(row.get("user_id", "")) for row in rows})[:limit]
        rows = [row for row in rows if str(row.get("user_id", "")) in users]
    destination = idpo_path(config, 0, "train_rollouts.jsonl")
    existing = read_jsonl(destination) if bool(settings.get("resume_existing", True)) and destination.exists() else []
    done = {str(row["rollout_id"]): row for row in existing}
    editor = _editor(config)
    teacher = teacher_client(config) if bool(settings.get("include_teacher_crossover", True)) else None
    samples = int(settings.get("rollout_samples", 8))
    mutation_parents = max(1, int(settings.get("mutation_parents", 1)))
    for index, row in enumerate(rows, 1):
        rollout_id = f"{row['id']}:global"
        if rollout_id in done:
            print(f"global IDPO rollout {index}/{len(rows)} source=resume", flush=True)
            continue
        history = visible_history(row, int(config["retrieval"]["top_k"]))
        parents = list(row.get("candidates", []))[: max(2, mutation_parents)]
        if not parents:
            continue
        mutation_parent = parents[0]
        rollout_mode = (
            "plain_output_only"
            if str(settings.get("rollout_response_mode", "trace")) == "plain_output_only"
            else "output_only"
            if str(settings.get("rollout_response_mode", "trace")) == "output_only"
            else "global_mutation_edit"
        )
        prompt = build_configured_editor_prompt(
            config,
            row, "mutation", mutation_parent, None,
            int(config["generation"].get("maximum_history_records", 8)),
            supervision_mode=rollout_mode,
            history_input_max_chars=int(config["simple_conditional_trace"].get("history_input_max_chars", 500)),
            history_output_max_chars=int(config["simple_conditional_trace"].get("history_output_max_chars", 300)),
        )
        raw_values = editor.generate_many([prompt] * samples)
        responses: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_payload, raw, error in raw_values:
            if error is not None:
                continue
            payload = _compact_payload(
                raw_payload,
                mutation_parent,
                history,
                output_only=rollout_mode == "output_only",
                allow_long_output=base_text_protocol,
            )
            if payload is None:
                continue
            key = payload["output"].casefold()
            if key in seen:
                continue
            seen.add(key)
            trace_text, output_text, response_text = canonical_response_parts(
                "mutation", payload, plain_output_only=plain_output_only
            )
            responses.append({
                "response_id": f"r{len(responses)}",
                "operation_type": "mutation",
                "output": payload["output"],
                "response_text": response_text,
                "trace_text": trace_text,
                "output_text": output_text,
                "trace": payload,
                "source": "student_on_policy",
                "dpo_eligible": True,
                "raw_response": raw,
            })
        # 遗传式阶段：先由 Student Mutation 产生子代，再让 Teacher 对两个
        # Mutation 子代做 sequence-level crossover。若 Mutation 有效响应不足，
        # 才退回初始 Parent，保证小样本/解析失败时流程仍可继续。
        crossover_sources = [
            {
                "candidate_id": f"{row['id']}:mut:{item['response_id']}",
                "text": str(item["output"]),
            }
            for item in responses
            if item.get("source") == "student_on_policy"
        ]
        if len(crossover_sources) < 2:
            crossover_sources = [
                {"candidate_id": str(item["candidate_id"]), "text": str(item["text"])}
                for item in parents[:2]
            ]
        if teacher is not None and len(crossover_sources) >= 2:
            crossover_count = max(1, int(settings.get("crossover_count", 1)))
            for cross_index in range(crossover_count):
                cross_left, cross_right = crossover_sources[0], crossover_sources[1]
                cross_prompt = _teacher_crossover_prompt(row, cross_left, cross_right, history)
                try:
                    value, raw = teacher.json(
                        f"global_idpo_teacher_crossover:{row['id']}:{cross_index}",
                        cross_prompt,
                        {"current_input": row["source_text"], "parent_a": cross_left, "parent_b": cross_right, "retrieved_history": history},
                    )
                    payload = _compact_payload(value, cross_left, history)
                    if payload is not None and payload["output"].casefold() not in seen:
                        trace_text, output_text, response_text = canonical_response_parts("crossover", payload)
                        responses.append({
                            "response_id": f"r{len(responses)}",
                            "operation_type": "crossover",
                            "output": payload["output"],
                            "response_text": response_text,
                            "trace_text": trace_text,
                            "output_text": output_text,
                            "trace": payload,
                            "source": "teacher_direct_sequence_fusion",
                            "dpo_eligible": bool(settings.get("dpo_include_teacher_crossover", False)),
                            "raw_response": raw,
                        })
                except Exception as error:
                    print(f"teacher crossover skipped sample={row['id']}: {error}", flush=True)
        done[rollout_id] = {
            "rollout_id": rollout_id,
            "user_id": str(row.get("user_id", "")),
            "pseudo_query_id": str(row["id"]),
            "operation_type": "mutation",
            "prompt": prompt,
            "current_input": str(row["source_text"]),
            "retrieved_history": history,
            "parent_a": {"candidate_id": str(mutation_parent["candidate_id"]), "text": str(mutation_parent["text"])},
            "parent_b": None if len(parents) < 2 else {"candidate_id": str(parents[1]["candidate_id"]), "text": str(parents[1]["text"])},
            "responses": responses,
            "minimum_responses_met": len([r for r in responses if r["dpo_eligible"]]) >= int(settings.get("minimum_valid_responses", 4)),
            "gold_visible_during_rollout": False,
        }
        if index % int(settings.get("checkpoint_every", 20)) == 0:
            write_jsonl(destination, [done[key] for key in sorted(done)])
        print(f"global IDPO rollout {index}/{len(rows)} user={row.get('user_id','')} responses={len(responses)}", flush=True)
    write_jsonl(destination, [done[key] for key in sorted(done)])
    report = {
        "split": split,
        "rollouts": len(done),
        "responses": sum(len(row.get("responses", [])) for row in done.values()),
        "valid_rollouts": sum(bool(row.get("minimum_responses_met")) for row in done.values()),
        "teacher_crossover_responses": sum(sum(item.get("source") == "teacher_direct_sequence_fusion" for item in row.get("responses", [])) for row in done.values()),
        "student_mutation_responses": sum(sum(item.get("source") == "student_on_policy" for item in row.get("responses", [])) for row in done.values()),
        "dpo_policy": "global_shared_sft_adapter",
        "teacher_crossover_in_dpo": bool(settings.get("dpo_include_teacher_crossover", False)),
    }
    from pipeline_common import write_json
    write_json(idpo_path(config, 0, "train_rollout_report.json"), report)
    print(f"Global IDPO rollouts -> {destination}; report={report}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="24 - global IDPO rollout")
    parser.add_argument("--config", default=str(HERE.parent / "config_global.yaml"))
    args = parser.parse_args()
    run(load_config(args.config))
