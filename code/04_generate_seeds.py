"""阶段 04：分别生成 task-only 和 factor-conditioned 初始候选。

两类 seed 共用当前 query，但后者额外看到因子方向；这里只产生候选，不进行
ROUGE 过滤，因此可以安全地扩大 Teacher 样本数来做数据规模实验。
"""

from __future__ import annotations

import json
from pathlib import Path
import re

from common.concurrency import BoundedJobError, run_bounded
from common.prompts import render_prompt
from common.teacher import TeacherClient
from common.utils import load_config, read_jsonl, write_jsonl


MISSING_ABSTRACTS = {"no abstract available", "no abstract available."}
INVALID_CANDIDATES = {"", "n/a", "none", "null", "title not available"}


def _drop_removed_factors(row: dict) -> None:
    """Remove legacy fixed summaries before any candidate-generation prompt."""
    row["factors"] = [
        factor
        for factor in row.get("factors", [])
        if str(factor.get("factor_id", "")) != "profile_style"
    ]
    if not row["factors"]:
        raise ValueError(f"No active factors remain for id={row.get('id')}")


def build_task_seed_prompt(row: dict, count: int) -> str:
    return render_prompt(
        "lamp5/02_task_seed_v2.txt", count=count, source_text=row["source_text"]
    )


def build_factor_seed_prompt(row: dict, count: int) -> str:
    factors = [
        {
            "factor_id": factor["factor_id"],
            "type": factor["type"],
            "direction": factor["direction"],
            "condition": factor["condition"],
        }
        for factor in row["factors"]
    ]
    return render_prompt(
        "lamp5/03_factor_seed_v2.txt",
        count=count,
        source_text=row["source_text"],
        factors=json.dumps(factors, ensure_ascii=False),
    )


def build_seed_repair_prompt(
    original_prompt: str,
    invalid_payload,
    error: str,
    count: int,
) -> str:
    return render_prompt(
        "lamp5/04_seed_repair_v2.txt",
        count=count,
        error=error,
        invalid_payload=json.dumps(invalid_payload, ensure_ascii=False),
        original_prompt=original_prompt,
    )

def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _candidate_values(payload: dict, sample_id: str, group: str) -> list[str]:
    texts = payload.get("candidates", [])
    if not isinstance(texts, list):
        raise ValueError(f"{group} candidates must be a list for id={sample_id}")
    seen = set()
    unique = []
    for value in texts:
        text = str(value).strip().strip('"')
        normalized = _key(text)
        if text.casefold() not in INVALID_CANDIDATES and normalized not in seen:
            seen.add(normalized)
            unique.append(text)
    return unique


def _parse_candidates(payload: dict, count: int, sample_id: str, group: str) -> list[str]:
    unique = _candidate_values(payload, sample_id, group)
    if len(unique) != count:
        raise ValueError(
            f"Expected exactly {count} distinct {group} candidates for id={sample_id}, got {len(unique)}"
        )
    return unique


def _request_group(
    row: dict,
    settings: dict,
    client: TeacherClient,
    task: str,
    prompt: str,
    count: int,
    group: str,
) -> tuple[list[str], dict]:
    context = {**row, "seed_count": count}
    payload, raw = client.json(task, prompt, context)
    initial_raw = raw
    repair_count = 0
    while True:
        try:
            texts = _parse_candidates(payload, count, row["id"], group)
            return texts, {
                "count": count,
                "repair_count": repair_count,
                "initial_raw_response": initial_raw,
                "raw_response": raw,
            }
        except (AttributeError, TypeError, ValueError) as error:
            if repair_count >= int(settings.get("schema_retries", 2)):
                partial = _candidate_values(payload, row["id"], group)
                if partial and bool(settings.get("allow_partial_group", False)):
                    return partial, {
                        "count": len(partial),
                        "requested_count": count,
                        "repair_count": repair_count,
                        "partial_fallback": True,
                        "fallback_reason": str(error),
                        "initial_raw_response": initial_raw,
                        "raw_response": raw,
                    }
                raise ValueError(
                    f"Invalid {group} seeds for id={row['id']} after "
                    f"{repair_count} repairs: {error}"
                ) from error
            repair_count += 1
            repair_prompt = build_seed_repair_prompt(prompt, payload, str(error), count)
            payload, raw = client.json(
                f"{task}_repair_{repair_count}", repair_prompt, context
            )


def _generate_one(row: dict, settings: dict, client: TeacherClient) -> dict:
    _drop_removed_factors(row)
    task_count = int(settings["task_seeds"])
    factor_count = int(settings["factor_seeds"])
    task_prompt = build_task_seed_prompt(row, task_count)
    task_texts, task_metadata = _request_group(
        row,
        settings,
        client,
        "seeds_task",
        task_prompt,
        task_count,
        "task-only",
    )
    factor_prompt = build_factor_seed_prompt(row, factor_count)
    if str(row["source_text"]).strip().casefold() in MISSING_ABSTRACTS:
        factor_texts = []
        factor_metadata = {
            "count": 0,
            "requested_count": factor_count,
            "skipped": True,
            "skip_reason": "missing_current_abstract",
        }
    else:
        factor_texts, factor_metadata = _request_group(
            row,
            settings,
            client,
            "seeds_factor",
            factor_prompt,
            factor_count,
            "factor-conditioned",
        )
    factor_ids = [factor["factor_id"] for factor in row["factors"]]
    row["candidates"] = [
        {
            "candidate_id": f"{row['id']}_task_{index}",
            "type": "task_seed",
            "used_factors": [],
            "text": text,
        }
        for index, text in enumerate(task_texts)
    ] + [
        {
            "candidate_id": f"{row['id']}_factor_{index}",
            "type": "factor_seed",
            "used_factors": factor_ids,
            "text": text,
        }
        for index, text in enumerate(factor_texts)
    ]
    row["seed_metadata"] = {
        "prompt_version": settings["prompt_version"],
        "model": client.config["model"],
        "task_only": task_metadata,
        "factor_conditioned": factor_metadata,
    }
    return row


def generate(source: Path, destination: Path, config: dict, client: TeacherClient) -> None:
    source_rows = read_jsonl(source)
    settings = config["generation"]
    task_count = int(settings["task_seeds"])
    factor_count = int(settings["factor_seeds"])
    concurrency = int(settings.get("concurrency", 1))
    if concurrency < 1:
        raise ValueError("generation.concurrency must be at least 1")
    failure_policy = str(settings.get("failure_policy", "error"))
    if failure_policy not in {"error", "skip"}:
        raise ValueError("generation.failure_policy must be 'error' or 'skip'")
    resume_existing = bool(settings.get("resume_existing", False))
    existing_rows = read_jsonl(destination) if resume_existing and destination.exists() else []
    result_by_id = {str(row["id"]): row for row in existing_rows}
    jobs = [row for row in source_rows if str(row["id"]) not in result_by_id]
    print(
        f"seed sample operations={len(jobs)}, concurrency={concurrency}, "
        f"resume={len(result_by_id)}/{len(source_rows)}",
        flush=True,
    )

    def worker(job: dict) -> dict:
        try:
            return _generate_one(job, settings, client)
        except Exception as error:
            if failure_policy == "error":
                raise
            return {
                "_skipped": True,
                "sample_id": str(job["id"]),
                "error": f"{type(error).__name__}: {error}",
            }

    def on_result(job: dict, result: dict, completed: int) -> None:
        row = job
        if result.get("_skipped"):
            print(
                f"seed skipped {completed}/{len(jobs)} sample={row['id']}: "
                f"{result['error']}",
                flush=True,
            )
            return
        result_by_id[str(row["id"])] = result
        print(
            f"seed progress {completed}/{len(jobs)} sample={row['id']}",
            flush=True,
        )

    try:
        run_bounded(
            jobs,
            worker,
            on_result,
            max_workers=concurrency,
            thread_name_prefix="mevo-seed",
        )
    except BoundedJobError as failure:
        row = failure.job
        raise RuntimeError(
            f"Seed generation failed for sample={row['id']}: {failure.error}"
        ) from failure.error
    rows = [row for row in source_rows if str(row["id"]) in result_by_id]
    rows = [result_by_id[str(row["id"])] for row in rows]
    write_jsonl(destination, rows)
    skipped = len(source_rows) - len(rows)
    print(
        f"generated {task_count} task-only + {factor_count} factor-conditioned seeds "
        f"for {len(rows)} samples; skipped {skipped} -> {destination}"
    )
    if skipped and bool(settings.get("require_complete", False)):
        raise RuntimeError(
            f"Seed stage remains incomplete: {len(rows)}/{len(source_rows)} samples; "
            f"retry will process only the {skipped} missing samples"
        )


def main() -> None:
    from common.runtime import config_parser, stage_path, teacher_client

    args = config_parser("04 - Generate task-only and factor-conditioned seeds").parse_args()
    config = load_config(args.config)
    generate(
        stage_path(config, "factors"),
        stage_path(config, "seeds"),
        config,
        teacher_client(config),
    )


if __name__ == "__main__":
    main()
