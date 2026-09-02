"""阶段 03：生成 2 个 Task-only 与 2 个直接 Profile-conditioned Seed。

这里没有 Factor。两组 Seed 分两次请求，确保 Task-only Prompt 不会意外看到
Profile，便于后续严格比较个性化历史本身是否有效。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from common.concurrency import BoundedJobError, run_bounded  # noqa: E402
from pipeline_common import (  # noqa: E402
    candidate,
    load_config,
    mock_title,
    normalized_text,
    read_jsonl,
    render_local_prompt,
    resolve_path,
    stage_path,
    teacher_client,
    visible_history,
    write_jsonl,
)


def _candidate_values(payload: Any) -> list[str]:
    """兼容 OpenAI-compatible 服务常见的等价 JSON Schema。

    不在这里放宽候选质量约束。兼容的只是容器字段和列表项表示方式，
    标题仍必须非空、单行、长度合格且互不重复。
    """

    def container(value: Any, depth: int = 0) -> list[Any] | None:
        # IDPO 每个 LOO Query 只要1个 task Parent。若端点返回
        # 单行标题而非容器 JSON，仍可按同样的内容质量约束验证。
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return value
        if not isinstance(value, dict) or depth > 1:
            return None
        for key in ("candidates", "titles", "outputs"):
            if isinstance(value.get(key), list):
                return value[key]
        for key in ("title", "output", "text"):
            if isinstance(value.get(key), str):
                return [value[key]]
        for key in ("data", "result", "response"):
            nested = container(value.get(key), depth + 1)
            if nested is not None:
                return nested
        return None

    items = container(payload)
    if items is None:
        raise ValueError("Teacher 响应中没有可识别的标题列表")
    values = []
    seen = set()
    for item in items:
        if isinstance(item, dict):
            item = next(
                (item[key] for key in ("title", "output", "text") if isinstance(item.get(key), str)),
                "",
            )
        text = str(item).strip().strip('"').strip()
        key = normalized_text(text)
        if text and key and key not in seen and "\n" not in text and len(text) <= 300:
            values.append(text)
            seen.add(key)
    return values


def _request_group(
    client,
    task: str,
    prompt: str,
    context: dict[str, Any],
    count: int,
    retries: int,
) -> list[str]:
    # 快速 IDPO 可以只需要一类 Parent。count=0 时不得为了空结果调用 Teacher。
    if count <= 0:
        return []
    error: Exception | None = None
    values: list[str] = []
    seen: set[str] = set()
    for attempt in range(retries + 1):
        current = prompt
        if attempt:
            missing = count - len(values)
            current += (
                "\n\nCOMPLETENESS RETRY: Return exactly "
                f"{missing} additional distinct non-empty final scholarly paper title(s). Each title "
                "must be one line and at most 30 words; do not return an abstract, "
                "summary, or explanation. Generate a fresh response independently; "
                "no previous answer is provided as context. "
                f"Previous error: {error}"
            )
        payload, _ = client.json(f"{task}_retry_{attempt}", current, context)
        try:
            for value in _candidate_values(payload):
                key = normalized_text(value)
                if key not in seen:
                    values.append(value)
                    seen.add(key)
            if len(values) >= count:
                return values[:count]
            raise ValueError(f"expected {count} unique titles, accumulated {len(values)}")
        except (TypeError, ValueError) as current_error:
            client.invalidate(f"{task}_retry_{attempt}", current)
            error = current_error
    raise ValueError(f"Seed Teacher 响应在修复后仍无效: {error}")


def _generate_one(row: dict[str, Any], config: dict, client) -> dict[str, Any]:
    settings = config["generation"]
    task_count = int(settings["task_seeds"])
    profile_count = int(settings["profile_seeds"])
    history = visible_history(row, int(settings["maximum_history_records"]))
    if client.config["provider"] == "mock":
        task_values = [
            mock_title(row["source_text"], f"Task{i + 1}") for i in range(task_count)
        ]
        profile_values = [
            mock_title(row["source_text"], f"Profile{i + 1}")
            for i in range(profile_count)
        ]
    else:
        task_prompt = render_local_prompt(
            "01_task_seeds.txt",
            count=task_count,
            instruction=str(row.get("instruction", "Generate the requested output")),
            current_input=str(row["source_text"]),
        )
        task_values = _request_group(
            client,
            "factor_free_task_seeds",
            task_prompt,
            {"current_input": row["source_text"], "count": task_count},
            task_count,
            int(settings["schema_retries"]),
        )
        profile_prompt = render_local_prompt(
            "02_profile_seeds.txt",
            count=profile_count,
            instruction=str(row.get("instruction", "Generate the requested output")),
            current_input=str(row["source_text"]),
            history=json.dumps(history, ensure_ascii=False),
        )
        profile_values = _request_group(
            client,
            "factor_free_profile_seeds",
            profile_prompt,
            {
                "current_input": row["source_text"],
                "retrieved_history": history,
                "count": profile_count,
            },
            profile_count,
            int(settings["schema_retries"]),
        )
    row["candidates"] = [
        candidate(f"{row['id']}_task_{index}", "task_seed", text)
        for index, text in enumerate(task_values)
    ] + [
        candidate(f"{row['id']}_profile_{index}", "profile_seed", text)
        for index, text in enumerate(profile_values)
    ]
    row["seed_metadata"] = {
        "task_seed_count": len(task_values),
        "profile_seed_count": len(profile_values),
        "explicit_user_factors": False,
        "gold_visible": False,
        "model": str(client.config["model"]),
    }
    return row


def generate(config: dict, split: str) -> Path:
    source_rows = read_jsonl(stage_path(config, split, "retrieve"))
    destination = stage_path(config, split, "seeds")
    settings = config["generation"]
    existing = (
        read_jsonl(destination)
        if bool(settings.get("resume_existing", True)) and destination.exists()
        else []
    )
    # 不重复付费生成同一个 target-blind Parent。正式 S0 子集可从此前由同一
    # Teacher 构造的完整 split 中按 ID 复用；缺失 ID 仍会正常调用 API。
    reuse_splits = settings.get("reuse_processed_splits", {})
    reuse_name = reuse_splits.get(split) if isinstance(reuse_splits, dict) else None
    if reuse_name:
        reuse_path = (
            resolve_path(config["paths"]["candidate_root"])
            / str(reuse_name)
            / "03_seeds.jsonl"
        )
        if reuse_path.exists():
            existing_by_id = {str(row["id"]): row for row in existing}
            for row in read_jsonl(reuse_path):
                existing_by_id.setdefault(str(row["id"]), row)
            existing = list(existing_by_id.values())
    source_ids = {str(row["id"]) for row in source_rows}
    by_id = {
        str(row["id"]): row
        for row in existing
        if str(row["id"]) in source_ids
        if len(row.get("candidates", []))
        == int(settings["task_seeds"]) + int(settings["profile_seeds"])
    }
    jobs = [row for row in source_rows if str(row["id"]) not in by_id]
    client = teacher_client(config)
    print(
        f"factor-free seed rows={len(jobs)}, resume={len(by_id)}/{len(source_rows)}, "
        f"concurrency={settings['concurrency']}, reuse={reuse_name or 'none'}",
        flush=True,
    )

    def worker(row: dict[str, Any]) -> dict[str, Any]:
        return _generate_one(row, config, client)

    def checkpoint() -> None:
        # 始终按源数据顺序落盘，断点恢复后不会因为线程完成顺序改变样本顺序。
        ordered = [
            by_id[str(source_row["id"])]
            for source_row in source_rows
            if str(source_row["id"]) in by_id
        ]
        write_jsonl(destination, ordered)

    def done(row: dict[str, Any], result: dict[str, Any], completed: int) -> None:
        by_id[str(row["id"])] = result
        if completed % int(settings.get("checkpoint_every", 20)) == 0:
            checkpoint()
        print(f"seed progress {completed}/{len(jobs)} sample={row['id']}", flush=True)

    try:
        run_bounded(
            jobs,
            worker,
            done,
            max_workers=int(settings["concurrency"]),
            thread_name_prefix="factor-free-seed",
        )
    except BoundedJobError as failure:
        # 异常发生时也保留未到 checkpoint_every 边界的已完成请求。
        checkpoint()
        raise RuntimeError(
            f"Seed 生成失败 sample={failure.job['id']}: {failure.error}"
        ) from failure.error
    checkpoint()
    print(f"factor-free seeds -> {destination}; rows={len(source_rows)}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="03 - 生成无 Factor Seed")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "adaptation_train", "adaptation_validation", "adaptation_test"),
        default="train",
    )
    args = parser.parse_args()
    generate(load_config(args.config), args.split)


if __name__ == "__main__":
    main()
