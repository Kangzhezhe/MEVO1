"""阶段 04：为四个 target-blind Parent 构造 Gold-aware 编辑轨迹。

Teacher 在训练阶段可以看到 Gold，但只负责解释 Parent 应如何被修正。最终
output 不由 Teacher 生成，而由程序强制设为数据集 Gold。该文件仅处理 train。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from common.concurrency import BoundedJobError, run_bounded  # noqa: E402
from pipeline_common import (  # noqa: E402
    compact_signal,
    load_config,
    normalized_text,
    read_jsonl,
    render_local_prompt,
    stage_path,
    teacher_client,
    visible_history,
    write_jsonl,
)


def _request_list(
    client,
    task: str,
    prompt: str,
    context: dict[str, Any],
    key: str,
    expected: int,
    retries: int,
    validator: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """JSON repair 由 TeacherClient 处理；这里负责业务 Schema 重试。"""

    error: Exception | None = None
    for attempt in range(retries + 1):
        current = prompt
        if attempt:
            current += (
                "\n\nSCHEMA/COMPLETENESS RETRY: Return all requested items exactly once. "
                "Generate the response independently; no previous answer is included. "
                f"Previous validation error: {error}"
            )
        task_name = f"{task}_retry_{attempt}"
        payload, _ = client.json(task_name, current, context)
        try:
            if not isinstance(payload, dict) or not isinstance(payload.get(key), list):
                raise ValueError(f"response must contain {key} list")
            values = [item for item in payload[key] if isinstance(item, dict)]
            if len(values) != expected:
                raise ValueError(f"expected {expected}, received {len(values)}")
            if validator is not None:
                validator(values)
            return values
        except (TypeError, ValueError) as current_error:
            client.invalidate(task_name, current)
            error = current_error
    raise ValueError(f"Gold-aware Teacher 响应修复失败: {error}")


def _validate_trace(
    value: Any,
    parent: dict[str, Any],
    valid_evidence: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("trace 必须是 object")
    if str(value.get("parent_id", "")) != str(parent["candidate_id"]):
        raise ValueError("trace parent_id 不匹配")
    task_correction = str(value.get("task_correction", "")).strip()
    if not task_correction or len(task_correction) > 300:
        raise ValueError("task_correction 必须是简短非空文本")
    signal = compact_signal(value.get("profile_signal"), valid_evidence)
    edit_action = str(value.get("edit_action", "")).strip()
    if not edit_action or len(edit_action) > 300:
        raise ValueError("edit_action 必须是简短非空文本")
    return {
        "parent_id": str(parent["candidate_id"]),
        "task_correction": task_correction,
        "profile_signal": signal,
        "edit_action": edit_action,
    }


def _mock_traces(row: dict[str, Any], parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    history = visible_history(row, 4)
    evidence = [history[0]["id"]] if history else []
    observation = (
        "The visible history supports a concise recurring output choice."
        if history
        else "No reliable profile-specific change is supported by the visible history."
    )
    return [
        {
            "parent_id": parent["candidate_id"],
            "task_correction": "Preserve the central task contribution and remove unsupported wording.",
            "profile_signal": {
                "evidence_ids": evidence,
                "observation": observation,
            },
            "edit_action": "Revise the parent into the demonstrated target while preserving task fidelity.",
        }
        for parent in parents
    ]


def _build_one(row: dict[str, Any], config: dict, client) -> dict[str, Any]:
    if not str(row.get("target", "")).strip():
        raise ValueError(f"sample={row['id']} 缺少 Gold target")
    settings = config["gold_aware_sft"]
    parents = list(row.get("candidates", []))
    expected = int(config["generation"]["task_seeds"]) + int(
        config["generation"]["profile_seeds"]
    )
    if len(parents) != expected:
        raise ValueError(f"sample={row['id']} 需要 {expected} 个 target-blind Parent")

    history = visible_history(row, int(settings["maximum_history_records"]))
    valid_evidence = {str(item["id"]) for item in history}
    payload = {
        "current_input": str(row["source_text"]),
        "retrieved_history": history,
        "reference_output": str(row["target"]),
        "parents": [
            {"parent_id": item["candidate_id"], "text": item["text"]}
            for item in parents
        ],
    }

    if client.config["provider"] == "mock":
        raw = _mock_traces(row, parents)
    else:
        def validate_batch(values: list[dict[str, Any]]) -> None:
            by_parent = {str(item.get("parent_id")): item for item in values}
            if len(by_parent) != len(values):
                raise ValueError("trace parent_id 必须互不重复")
            for parent in parents:
                _validate_trace(
                    by_parent.get(str(parent["candidate_id"])), parent, valid_evidence
                )

        prompt = render_local_prompt(
            "03_gold_aware_traces.txt",
            payload=json.dumps(payload, ensure_ascii=False),
        )
        raw = _request_list(
            client,
            "gold_aware_traces",
            prompt,
            payload,
            "traces",
            expected,
            int(settings["schema_retries"]),
            validator=validate_batch,
        )

    by_parent = {str(item.get("parent_id")): item for item in raw}
    traces = []
    for parent in parents:
        trace = _validate_trace(
            by_parent.get(str(parent["candidate_id"])), parent, valid_evidence
        )
        trace.update(
            {
                "decision": (
                    "keep"
                    if normalized_text(parent["text"]) == normalized_text(row["target"])
                    else "revise"
                ),
                # Teacher 不生成 output，防止自由答案污染监督目标。
                "output": str(row["target"]),
            }
        )
        traces.append(trace)

    output = dict(row)
    output["gold_aware_traces"] = traces
    output["gold_aware_metadata"] = {
        "protocol": "gold_aware_parent_to_target_trace_v1",
        "teacher_sees_gold": True,
        "teacher_generates_output": False,
        "student_prompt_sees_gold": False,
        "trace_count": len(traces),
        "model": str(client.config["model"]),
    }
    return output


def build(config: dict, split: str) -> Path:
    if split != "train":
        raise ValueError("Gold-aware trace 只能在 train split 上构造")
    source_rows = read_jsonl(stage_path(config, split, "seeds"))
    destination = stage_path(config, split, "gold_traces")
    settings = config["gold_aware_sft"]
    existing = (
        read_jsonl(destination)
        if bool(settings.get("resume_existing", True)) and destination.exists()
        else []
    )
    expected = int(config["generation"]["task_seeds"]) + int(
        config["generation"]["profile_seeds"]
    )
    source_ids = {str(row["id"]) for row in source_rows}
    by_id = {
        str(row["id"]): row
        for row in existing
        if str(row["id"]) in source_ids
        and len(row.get("gold_aware_traces", [])) == expected
    }
    jobs = [row for row in source_rows if str(row["id"]) not in by_id]
    client = teacher_client(config)
    print(
        f"gold-aware traces rows={len(jobs)}, resume={len(by_id)}/{len(source_rows)}, "
        f"parents_per_query={expected}, concurrency={settings['concurrency']}",
        flush=True,
    )

    def checkpoint() -> None:
        ordered = [by_id[str(row["id"])] for row in source_rows if str(row["id"]) in by_id]
        write_jsonl(destination, ordered)

    def done(row: dict[str, Any], result: dict[str, Any], completed: int) -> None:
        by_id[str(row["id"])] = result
        if completed % int(settings.get("checkpoint_every", 10)) == 0:
            checkpoint()
        print(f"gold-aware progress {completed}/{len(jobs)} sample={row['id']}", flush=True)

    try:
        run_bounded(
            jobs,
            lambda row: _build_one(row, config, client),
            done,
            max_workers=int(settings["concurrency"]),
            thread_name_prefix="gold-aware-traces",
        )
    except BoundedJobError as failure:
        checkpoint()
        raise RuntimeError(
            f"Gold-aware trace 失败 sample={failure.job['id']}: {failure.error}"
        ) from failure.error
    checkpoint()
    print(f"gold-aware traces -> {destination}; rows={len(source_rows)}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="04 - 构建 Gold-aware SFT 编辑轨迹")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--split", choices=("train",), default="train")
    args = parser.parse_args()
    build(load_config(args.config), args.split)


if __name__ == "__main__":
    main()
