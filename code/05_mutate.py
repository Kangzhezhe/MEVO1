"""阶段 05：按每个因子改写 seed，形成候选池中的 mutation 子节点。

每次请求只输入 query、父候选、因子及其证据，不输入 gold。请求和 JSON schema
错误会走 repair 重试；``mutation.concurrency`` 可用于控制并发度。
"""

from __future__ import annotations

import json
from pathlib import Path
import re

from common.concurrency import BoundedJobError, run_bounded
from common.prompts import render_prompt
from common.teacher import TeacherClient
from common.utils import load_config, read_jsonl, write_jsonl


_TRACE_OUTPUT_PATTERN = re.compile(
    r'^\s*["\']?output["\']?\s*:\s*(.+?)\s*$',
    flags=re.IGNORECASE | re.MULTILINE,
)


def _clean_output(value) -> str:
    if not isinstance(value, str):
        return ""
    output = value.strip().strip('"').strip()
    if output.casefold() in {"", "none", "null", "n/a"}:
        return ""
    if "\n" in output or len(output) > 300:
        return ""
    return output


def _normalize_payload(payload) -> tuple[object, str | None]:
    """Recover a title misplaced on a final ``output:`` trace line.

    Some compatible endpoints put the requested title at the end of
    ``operation_trace`` while omitting the JSON ``output`` field. Only an
    explicitly labelled, single-line final title is recovered; arbitrary
    reasoning text is never treated as a candidate.
    """

    if not isinstance(payload, dict):
        return payload, None
    normalized = dict(payload)
    output = _clean_output(normalized.get("output"))
    if output:
        normalized["output"] = output
        return normalized, None

    matches = _TRACE_OUTPUT_PATTERN.findall(str(normalized.get("operation_trace", "")))
    if not matches:
        return normalized, None
    recovered = _clean_output(matches[-1].strip().rstrip("}"))
    if not recovered:
        return normalized, None
    normalized["output"] = recovered
    return normalized, "operation_trace_output_line"


def build_mutation_prompt(row: dict, parent: dict, factor: dict) -> str:
    evidence_ids = set(map(str, factor["evidence_ids"]))
    # Dynamic factors cite the retrieved Top-k, while full-profile stable
    # factors may cite any historical record. Prefer retrieved evidence and
    # then fill from the complete target-blind profile.
    evidence = [p for p in row["retrieved_profile"] if str(p["id"]) in evidence_ids]
    seen = {str(item["id"]) for item in evidence}
    evidence.extend(
        item
        for item in row.get("profile", [])
        if str(item.get("id")) in evidence_ids and str(item.get("id")) not in seen
    )
    evidence = evidence[:4]
    return render_prompt(
        "lamp5/05_mutation_v1.txt",
        source_text=row["source_text"],
        parent_text=parent["text"],
        factor=json.dumps(factor, ensure_ascii=False),
        evidence=json.dumps(evidence, ensure_ascii=False),
    )


def build_mutation_repair_prompt(
    row: dict,
    parent: dict,
    factor: dict,
    invalid_payload,
    error: str,
) -> str:
    return render_prompt(
        "lamp5/05_mutation_repair_v1.txt",
        error=error,
        invalid_payload=json.dumps(invalid_payload, ensure_ascii=False),
        source_text=row["source_text"],
        parent_text=parent["text"],
        factor=json.dumps(factor, ensure_ascii=False),
    )


def _validate(payload, row: dict, parent: dict) -> None:
    if not isinstance(payload, dict):
        raise ValueError("mutation response must be a JSON object")
    if payload.get("operation") != "mutation":
        raise ValueError("operation must equal 'mutation'")
    if not _clean_output(payload.get("output")):
        raise ValueError(
            f"mutation output is empty for id={row['id']}, parent={parent['candidate_id']}"
        )


def _mutate_one(
    row: dict,
    parent: dict,
    factor: dict,
    prompt_version: str,
    schema_retries: int,
    client: TeacherClient,
) -> dict:
    prompt = build_mutation_prompt(row, parent, factor)
    context = {**row, "parent": parent["text"], "factor": factor}
    payload, raw = client.json("mutation", prompt, context)
    initial_raw = raw
    repair_count = 0
    schema_recovery = None
    while True:
        payload, current_recovery = _normalize_payload(payload)
        schema_recovery = schema_recovery or current_recovery
        try:
            _validate(payload, row, parent)
            break
        except (TypeError, ValueError) as error:
            if repair_count >= schema_retries:
                raise ValueError(
                    f"Invalid mutation for id={row['id']}, parent={parent['candidate_id']} "
                    f"after {repair_count} repairs: {error}"
                ) from error
            repair_count += 1
            repair_prompt = build_mutation_repair_prompt(row, parent, factor, payload, str(error))
            payload, raw = client.json(
                f"mutation_repair_{repair_count}", repair_prompt, context
            )
    return {
        "candidate_id": f"{parent['candidate_id']}_mut_{factor['factor_id']}",
        "type": "mutation",
        "parent_id": parent["candidate_id"],
        "factor_id": factor["factor_id"],
        "operation_trace": str(payload.get("operation_trace", "")).strip(),
        "text": str(payload["output"]).strip().strip('"'),
        "prompt_version": prompt_version,
        "model": client.config["model"],
        "repair_count": repair_count,
        "schema_recovery": schema_recovery,
        "initial_raw_response": initial_raw,
        "raw_response": raw,
    }


def mutate(source: Path, destination: Path, config: dict, client: TeacherClient) -> None:
    rows = read_jsonl(source)
    settings = config["mutation"]
    concurrency = int(settings.get("concurrency", 1))
    if concurrency < 1:
        raise ValueError("mutation.concurrency must be at least 1")
    max_per_sample = int(settings.get("max_per_sample", 0))
    if max_per_sample < 0:
        raise ValueError("mutation.max_per_sample cannot be negative")
    schema_retries = int(settings.get("schema_retries", 2))
    if schema_retries < 0:
        raise ValueError("mutation.schema_retries cannot be negative")
    failure_policy = str(settings.get("failure_policy", "error"))
    if failure_policy not in {"error", "skip"}:
        raise ValueError("mutation.failure_policy must be 'error' or 'skip'")

    jobs = []
    for row_index, row in enumerate(rows):
        candidates = row["candidates"]
        factors = row["factors"]
        combinations = [
            (candidates[parent_index], factors[(parent_index + round_index) % len(factors)])
            for round_index in range(len(factors))
            for parent_index in range(len(candidates))
        ]
        if max_per_sample:
            combinations = combinations[:max_per_sample]
        row["mutations"] = [None] * len(combinations)
        row["mutation_failures"] = []
        row["mutation_metadata"] = {
            "planned": len(combinations),
            "generated": 0,
            "skipped": 0,
            "failure_policy": failure_policy,
        }
        for slot, (parent, factor) in enumerate(combinations):
            jobs.append((row_index, slot, row, parent, factor))

    print(f"mutation operations={len(jobs)}, concurrency={concurrency}", flush=True)

    def worker(job: tuple[int, int, dict, dict, dict]) -> dict:
        _, _, row, parent, factor = job
        try:
            return _mutate_one(
                row,
                parent,
                factor,
                settings["prompt_version"],
                schema_retries,
                client,
            )
        except ValueError as error:
            if failure_policy == "error":
                raise
            return {
                "_skipped": True,
                "candidate_id": f"{parent['candidate_id']}_mut_{factor['factor_id']}",
                "parent_id": parent["candidate_id"],
                "factor_id": factor["factor_id"],
                "error": str(error),
            }

    def on_result(
        job: tuple[int, int, dict, dict, dict], result: dict, completed: int
    ) -> None:
        row_index, slot, row, parent, factor = job
        if result.get("_skipped"):
            rows[row_index]["mutation_failures"].append({
                key: result[key]
                for key in ("candidate_id", "parent_id", "factor_id", "error")
            })
            rows[row_index]["mutation_metadata"]["skipped"] += 1
            print(
                f"mutation skipped {completed}/{len(jobs)} "
                f"sample={row['id']} parent={parent['candidate_id']} "
                f"factor={factor['factor_id']}: {result['error']}",
                flush=True,
            )
            return
        rows[row_index]["mutations"][slot] = result
        rows[row_index]["mutation_metadata"]["generated"] += 1
        print(
            f"mutation progress {completed}/{len(jobs)} "
            f"sample={row['id']} parent={parent['candidate_id']} "
            f"factor={factor['factor_id']}",
            flush=True,
        )

    try:
        run_bounded(
            jobs,
            worker,
            on_result,
            max_workers=concurrency,
            thread_name_prefix="mevo-mutation",
        )
    except BoundedJobError as failure:
        _, _, row, parent, factor = failure.job
        raise RuntimeError(
            f"Mutation failed for sample={row['id']}, "
            f"parent={parent['candidate_id']}, factor={factor['factor_id']}: "
            f"{failure.error}"
        ) from failure.error

    for row in rows:
        row["mutations"] = [mutation for mutation in row["mutations"] if mutation is not None]
    generated = sum(row["mutation_metadata"]["generated"] for row in rows)
    skipped = sum(row["mutation_metadata"]["skipped"] for row in rows)
    write_jsonl(destination, rows)
    print(
        f"generated {generated}/{len(jobs)} mutations; skipped {skipped} -> {destination}"
    )


def main() -> None:
    from common.runtime import config_parser, stage_path, teacher_client

    args = config_parser("05 - Mutate candidates along factor directions").parse_args()
    config = load_config(args.config)
    mutate(
        stage_path(config, "seeds"),
        stage_path(config, "mutate"),
        config,
        teacher_client(config),
    )


if __name__ == "__main__":
    main()
