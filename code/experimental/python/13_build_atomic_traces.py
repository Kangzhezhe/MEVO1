"""阶段 13：为 S2 构造严格门控的原子编辑监督。

输入与 S0/S1 完全相同的 target-blind 四 Parent。Teacher 训练请求额外看到
Gold、真实历史和随机其他用户历史；Gold 只用于解释监督，不会进入 Student
Prompt 或后续候选池。无法通过原子门控的 Query 保留 quality_rejected，SFT
阶段会回退到该 Query 的 S0 output-only 标签。
"""

from __future__ import annotations

import argparse
import copy
import json
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
    read_jsonl,
    resolve_path,
    stage_path,
    visible_history,
    write_jsonl,
)
from supervision_quality_pilot import _atomic_one, _validate_atomic  # noqa: E402


def _contrast_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        for offset in range(1, len(rows)):
            other = rows[(index + offset) % len(rows)]
            if str(other.get("user_id")) != str(row.get("user_id")):
                result[str(row["id"])] = other
                break
        if str(row["id"]) not in result:
            raise ValueError(f"sample={row['id']} 找不到其他用户历史")
    return result


def _mock_one(row: dict[str, Any], contrast: dict[str, Any]) -> dict[str, Any]:
    history = visible_history(row, 4)
    evidence = [str(item["id"]) for item in history[:2]]
    traces = []
    for parent in row["candidates"]:
        task_ops = []
        if str(parent["text"]).strip() != str(row["target"]).strip():
            task_ops = [{
                "type": "replace",
                "source_span": str(parent["text"]).split()[0],
                "target_span": str(row["target"]).split()[0],
            }]
        personal = []
        if len(evidence) >= 2:
            personal = [{
                "type": "format",
                "source_span": "",
                "target_span": "",
                "evidence_ids": evidence,
                "history_pattern": "The two visible titles share a recurring concise structure.",
                "application": "Apply the structure only when it fits the current contribution.",
            }]
        traces.append({
            "parent_id": str(parent["candidate_id"]),
            "task_operations": task_ops,
            "personalized_operations": personal,
            "output": str(row["target"]),
        })
    return {
        "id": str(row["id"]),
        "user_id": str(row.get("user_id", row["id"])),
        "source_text": str(row["source_text"]),
        "target": str(row["target"]),
        "retrieved_profile": row.get("retrieved_profile", []),
        "contrast_user_id": str(contrast.get("user_id", contrast["id"])),
        "candidates": row["candidates"],
        "atomic_traces": _validate_atomic(
            {"traces": traces}, row
        ),
        "atomic_metadata": {
            "protocol": "s2_atomic_trace_v1",
            "teacher_sees_gold": True,
            "quality_rejected": False,
            "mock": True,
        },
    }


def build(config: dict[str, Any]) -> Path:
    split = "train"
    source_rows = read_jsonl(stage_path(config, split, "seeds"))
    destination = stage_path(config, split, "atomic_traces")
    settings = config["atomic_sft"]
    existing = (
        read_jsonl(destination)
        if bool(settings.get("resume_existing", True)) and destination.exists()
        else []
    )
    source_ids = {str(row["id"]) for row in source_rows}
    expected = int(config["generation"]["task_seeds"]) + int(
        config["generation"]["profile_seeds"]
    )
    by_id = {
        str(row["id"]): row
        for row in existing
        if str(row["id"]) in source_ids
        and (
            bool(row.get("quality_rejected"))
            or len(row.get("atomic_traces", [])) == expected
        )
    }
    jobs = [row for row in source_rows if str(row["id"]) not in by_id]
    contrasts = _contrast_rows(source_rows)
    teacher_settings = copy.deepcopy(config["teacher"])
    cache_dir = resolve_path(settings["cache_dir"])
    teacher_settings["cache_dir"] = str(cache_dir)
    client = TeacherClient(teacher_settings, cache_dir)
    print(
        f"atomic traces rows={len(jobs)}, resume={len(by_id)}/{len(source_rows)}, "
        f"concurrency={settings['concurrency']}, retries={settings['schema_retries']}",
        flush=True,
    )

    def checkpoint() -> None:
        ordered = [by_id[str(row["id"])] for row in source_rows if str(row["id"]) in by_id]
        write_jsonl(destination, ordered)

    def worker(row: dict[str, Any]) -> dict[str, Any]:
        contrast = contrasts[str(row["id"])]
        if client.config["provider"] == "mock":
            return _mock_one(row, contrast)
        result = _atomic_one(
            row,
            contrast,
            client,
            int(settings["schema_retries"]),
        )
        result["atomic_metadata"] = {
            "protocol": "s2_atomic_trace_v1",
            "teacher_sees_gold": True,
            "quality_rejected": bool(result.get("quality_rejected", False)),
            "mock": False,
        }
        return result

    def done(row: dict[str, Any], result: dict[str, Any], completed: int) -> None:
        by_id[str(row["id"])] = result
        if completed % int(settings.get("checkpoint_every", 10)) == 0:
            checkpoint()
        rejected = bool(result.get("quality_rejected", False))
        print(
            f"atomic progress {completed}/{len(jobs)} sample={row['id']} "
            f"traces={len(result.get('atomic_traces', []))} rejected={rejected}",
            flush=True,
        )

    try:
        run_bounded(
            jobs,
            worker,
            done,
            max_workers=int(settings["concurrency"]),
            thread_name_prefix="atomic-traces",
        )
    except BoundedJobError as failure:
        checkpoint()
        raise RuntimeError(
            f"Atomic Trace 失败 sample={failure.job['id']}: {failure.error}"
        ) from failure.error
    checkpoint()
    rejected = sum(bool(row.get("quality_rejected", False)) for row in by_id.values())
    print(
        f"atomic traces -> {destination}; rows={len(source_rows)}, rejected={rejected}",
        flush=True,
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="13 - 构建 S2 原子编辑监督")
    parser.add_argument("--config", default=str(HERE / "config_s2.yaml"))
    args = parser.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
