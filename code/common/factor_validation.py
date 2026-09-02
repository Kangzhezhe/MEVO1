"""Full-profile rewrite-factor extraction and leave-out validation.

This is intentionally independent from stages 01-08. It tests whether a stable
factor bank extracted from a user's historical profile transfers to held-out
historical abstracts before changing candidate generation or Ranker training.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from common.concurrency import BoundedJobError, run_bounded
from common.metrics import score
from common.prompts import render_prompt
from common.teacher import TeacherClient
from common.utils import load_config, read_jsonl, resolve_path, write_jsonl


_NORMALIZE = re.compile(r"[^a-z0-9]+")
_OPERATION_TYPES = {
    "content_select",
    "content_omit",
    "structure",
    "compression",
    "lexicalization",
    "ordering",
}


def _key(value: str) -> str:
    return _NORMALIZE.sub("", str(value).lower())


def _unique_profile(row: dict[str, Any]) -> list[dict[str, str]]:
    result = []
    seen_abstracts: set[str] = set()
    seen_titles: set[str] = set()
    for index, item in enumerate(row.get("profile", [])):
        abstract = str(item.get("abstract", "")).strip()
        title = str(item.get("title", "")).strip()
        if not abstract or not title:
            continue
        abstract_key = _key(abstract)
        title_key = _key(title)
        if not abstract_key or not title_key or abstract_key in seen_abstracts or title_key in seen_titles:
            continue
        seen_abstracts.add(abstract_key)
        seen_titles.add(title_key)
        result.append({"id": str(item.get("id", index)), "abstract": abstract, "title": title})
    return result


def _split_profile(
    profiles: list[dict[str, str]], user_id: str, holdout_fraction: float, seed: int, min_build: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if len(profiles) <= min_build:
        raise ValueError(f"profile for user={user_id} has only {len(profiles)} records")
    holdout_count = max(1, int(round(len(profiles) * holdout_fraction)))
    holdout_count = min(holdout_count, len(profiles) - min_build)
    indices = list(range(len(profiles)))
    random.Random(f"{seed}:{user_id}:factor-validation").shuffle(indices)
    heldout_indices = set(indices[:holdout_count])
    build = [item for index, item in enumerate(profiles) if index not in heldout_indices]
    heldout = [item for index, item in enumerate(profiles) if index in heldout_indices]
    return build, heldout


def _chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _json_payload(payload: Any, key: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get(key), list):
        raise ValueError(f"Teacher payload must contain list field {key}")
    return [item for item in payload[key] if isinstance(item, dict)]


def _request(
    client: TeacherClient,
    task: str,
    prompt: str,
    context: dict[str, Any],
    key: str,
    retries: int,
) -> list[dict[str, Any]]:
    payload, _ = client.json(task, prompt, context)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return _json_payload(payload, key)
        except (TypeError, ValueError) as error:
            last_error = error
            if attempt >= retries:
                break
            repair_prompt = (
                prompt
                + f"\n\nSchema repair: return a complete JSON object with a list field named {key}."
            )
            payload, _ = client.json(f"{task}_repair_{attempt + 1}", repair_prompt, context)
    raise ValueError(f"invalid Teacher schema for task={task}: {last_error}") from last_error


def _drafts(
    records: list[dict[str, str]], settings: dict[str, Any], client: TeacherClient
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for batch in _chunks(records, int(settings["batch_size"])):
        visible_batch = [{"id": record["id"], "abstract": record["abstract"]} for record in batch]
        prompt = render_prompt("lamp5/09_factor_validation_task_drafts_v1.txt", records=json.dumps(visible_batch, ensure_ascii=False))
        values = _request(client, "factor_validation_task_drafts", prompt, {"records": visible_batch}, "drafts", int(settings["schema_retries"]))
        by_id = {str(item.get("id")): str(item.get("draft", "")).strip() for item in values}
        missing = [record for record in batch if not by_id.get(record["id"])]
        # Batch-compatible endpoints occasionally omit one item near the end
        # of a long response. Retry only those IDs with a one-record request;
        # the successful batch response remains cached and is not repeated.
        for record in missing:
            single = {"id": record["id"], "abstract": record["abstract"]}
            single_prompt = render_prompt(
                "lamp5/09_factor_validation_task_drafts_v1.txt",
                records=json.dumps([single], ensure_ascii=False),
            )
            single_values = []
            for retry in range(int(settings.get("missing_retries", 3)) + 1):
                retry_prompt = single_prompt
                if retry:
                    retry_prompt += "\nReturn exactly one item for the requested id. Do not omit it."
                single_values = _request(
                    client,
                    f"factor_validation_task_drafts_single_{retry}",
                    retry_prompt,
                    {"records": [single]},
                    "drafts",
                    int(settings["schema_retries"]),
                )
                if any(str(item.get("draft", "")).strip() for item in single_values):
                    break
            for item in single_values:
                if str(item.get("id")) == record["id"] and str(item.get("draft", "")).strip():
                    by_id[record["id"]] = str(item["draft"]).strip()
            if record["id"] not in by_id and len(single_values) == 1:
                fallback = str(single_values[0].get("draft", "")).strip()
                if fallback:
                    by_id[record["id"]] = fallback
        for record in batch:
            draft = by_id.get(record["id"], "")
            if not draft:
                raise ValueError(f"missing task-only draft for record={record['id']}")
            output.append({**record, "draft": draft})
    return output


def _extract(
    records: list[dict[str, str]], settings: dict[str, Any], client: TeacherClient
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for batch in _chunks(records, int(settings["batch_size"])):
        prompt = render_prompt("lamp5/10_factor_validation_extract_v1.txt", records=json.dumps(batch, ensure_ascii=False))
        values = _request(client, "factor_validation_extract", prompt, {"records": batch}, "factors", int(settings["schema_retries"]))
        for value in values:
            operation_type = str(value.get("operation_type", "")).strip()
            direction = str(value.get("direction", "")).strip()
            condition = str(value.get("condition", "")).strip()
            evidence_ids = [str(item) for item in value.get("evidence_ids", []) if str(item)]
            valid_ids = {record["id"] for record in batch}
            evidence_ids = [item for item in evidence_ids if item in valid_ids]
            if operation_type not in _OPERATION_TYPES or not direction or not condition or not evidence_ids:
                continue
            proposals.append({
                "operation_type": operation_type,
                "direction": direction,
                "condition": condition,
                "evidence_ids": sorted(set(evidence_ids)),
                "reason": str(value.get("reason", "")).strip(),
            })
    return proposals


def _merge(
    proposals: list[dict[str, Any]], settings: dict[str, Any], client: TeacherClient
) -> list[dict[str, Any]]:
    if not proposals:
        return []

    def reduce_chunks(chunks: Iterable[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        intermediate: list[dict[str, Any]] = []
        child_settings = {**settings, "merge_chunk_size": 0}
        for chunk in chunks:
            chunk_factors = _merge(chunk, child_settings, client)
            intermediate.extend(
                {
                    "operation_type": factor["operation_type"],
                    "direction": factor["direction"],
                    "condition": factor["condition"],
                    "evidence_ids": list(factor["support_ids"]),
                    "reason": "Intermediate factor from hierarchical full-profile merge.",
                }
                for factor in chunk_factors
            )
        if not intermediate:
            return []
        return _merge(intermediate, child_settings, client)

    # Large profiles can yield hundreds of proposals. Asking the Teacher to
    # reduce all of them in one response frequently produces truncated JSON.
    # Reduce bounded chunks first, then merge the compact intermediate factors.
    merge_chunk_size = int(settings.get("merge_chunk_size", 0))
    if merge_chunk_size > 0 and len(proposals) > merge_chunk_size:
        return reduce_chunks(_chunks(proposals, merge_chunk_size))

    def merge_prompt(items: list[dict[str, Any]]) -> str:
        return render_prompt(
            "lamp5/11_factor_validation_merge_v1.txt",
            max_factors=int(settings["max_factors"]),
            min_support=int(settings["min_support"]),
            proposals=json.dumps(items, ensure_ascii=False),
        )

    prompt = merge_prompt(proposals)
    cache_path = getattr(client, "_cache_path", None)
    cached = bool(callable(cache_path) and cache_path("factor_validation_merge", prompt).exists())
    prompt_budget = int(settings.get("merge_max_prompt_chars", 0))
    if prompt_budget > 0 and len(prompt) > prompt_budget and not cached:
        chunks: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for proposal in proposals:
            candidate = [*current, proposal]
            if current and len(merge_prompt(candidate)) > prompt_budget:
                chunks.append(current)
                current = [proposal]
            else:
                current = candidate
        if current:
            chunks.append(current)
        if len(chunks) > 1:
            return reduce_chunks(chunks)

    all_ids = {item for proposal in proposals for item in proposal["evidence_ids"]}

    def validate(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        factors = []
        for value in values[: int(settings["max_factors"])]:
            operation_type = str(value.get("operation_type", "")).strip()
            direction = str(value.get("direction", "")).strip()
            condition = str(value.get("condition", "")).strip()
            # Some compatible endpoints invert the schema and put the complete
            # imperative in ``condition`` while returning a one-word direction.
            # Normalize that response so downstream generation/Ranker inputs see
            # the executable instruction rather than a label such as "remove".
            if len(direction.split()) < 4 and len(condition.split()) >= 4:
                direction, condition = condition, "when applicable to the current abstract"
            support_ids = [
                str(item)
                for item in value.get("support_ids", [])
                if str(item) in all_ids
            ]
            support_ids = sorted(set(support_ids))
            if operation_type not in _OPERATION_TYPES or not direction or not condition:
                continue
            if len(support_ids) < int(settings["min_support"]):
                continue
            if any(factor["operation_type"] == operation_type for factor in factors):
                continue
            factors.append({
                "factor_id": f"f{len(factors) + 1}",
                "type": operation_type,
                "operation_type": operation_type,
                "direction": direction,
                "condition": condition,
                "support_ids": support_ids,
                "support_count": len(support_ids),
            })
        return factors

    context = {"factor_candidates": proposals}
    try:
        values = _request(
            client,
            "factor_validation_merge",
            prompt,
            context,
            "factors",
            int(settings["schema_retries"]),
        )
    except RuntimeError as error:
        # If an endpoint still truncates a bounded request, bisect only that
        # failed merge. Successful cached merges keep their original grouping.
        if "invalid JSON" not in str(error) or len(proposals) < 2:
            raise
        midpoint = len(proposals) // 2
        return reduce_chunks((proposals[:midpoint], proposals[midpoint:]))
    factors = validate(values)
    if factors:
        return factors

    # A response can be valid JSON but still unusable after semantic checks.
    # In particular, compatible endpoints sometimes rewrite opaque record IDs
    # (for example 408017 -> 40817). Stage-level retries then replay the same
    # cached response forever. For repair, replace long IDs with short aliases
    # and map them back locally; this preserves exact evidence without fuzzy
    # matching or trusting the model to reproduce opaque identifiers.
    ordered_ids = sorted(all_ids)
    id_to_alias = {record_id: f"r{index + 1}" for index, record_id in enumerate(ordered_ids)}
    alias_to_id = {alias: record_id for record_id, alias in id_to_alias.items()}
    aliased_proposals = [
        {
            **proposal,
            "evidence_ids": [id_to_alias[item] for item in proposal["evidence_ids"]],
        }
        for proposal in proposals
    ]
    aliased_prompt = render_prompt(
        "lamp5/11_factor_validation_merge_v1.txt",
        max_factors=int(settings["max_factors"]),
        min_support=int(settings["min_support"]),
        proposals=json.dumps(aliased_proposals, ensure_ascii=False),
    )
    merge_retries = int(settings.get("merge_retries", settings["schema_retries"]))
    for attempt in range(1, merge_retries + 1):
        repair_prompt = (
            aliased_prompt
            + "\n\nSEMANTIC REPAIR: The previous response had no factor that passed "
            + f"validation. Every factor needs at least {settings['min_support']} "
            + "distinct support_ids. In this repair prompt, every evidence ID is a "
            + "short alias such as r1. Copy only those aliases exactly; never invent "
            + "or modify one. Keep operation_type in the requested enum and return a "
            + "complete imperative direction and applicability condition."
        )
        aliased_values = _request(
            client,
            f"factor_validation_merge_alias_repair_{attempt}",
            repair_prompt,
            {"factor_candidates": aliased_proposals},
            "factors",
            int(settings["schema_retries"]),
        )
        values = []
        for value in aliased_values:
            decoded = dict(value)
            decoded["support_ids"] = [
                alias_to_id.get(str(item), str(item))
                for item in value.get("support_ids", [])
            ]
            values.append(decoded)
        factors = validate(values)
        if factors:
            return factors
    return []


def _heldout_predictions(
    records: list[dict[str, str]], factors: list[dict[str, Any]], settings: dict[str, Any], client: TeacherClient
) -> list[dict[str, Any]]:
    task_drafts = _drafts(records, settings, client)
    task_by_id = {record["id"]: record["draft"] for record in task_drafts}
    values = []
    visible_records = [{"id": record["id"], "abstract": record["abstract"]} for record in records]
    for batch in _chunks(visible_records, int(settings["batch_size"])):
        prompt = render_prompt(
            "lamp5/12_factor_validation_generate_v1.txt",
            factors=json.dumps(factors, ensure_ascii=False),
            records=json.dumps(batch, ensure_ascii=False),
        )
        batch_values = _request(client, "factor_validation_generate", prompt, {"records": batch, "factors": factors}, "predictions", int(settings["schema_retries"]))
        by_id = {str(item.get("id")): item for item in batch_values}
        for record in batch:
            existing = by_id.get(record["id"], {})
            if str(existing.get("factor_conditioned", "")).strip():
                continue
            single = [record]
            single_prompt = render_prompt(
                "lamp5/12_factor_validation_generate_v1.txt",
                factors=json.dumps(factors, ensure_ascii=False),
                records=json.dumps(single, ensure_ascii=False),
            )
            single_values = []
            for retry in range(int(settings.get("missing_retries", 3)) + 1):
                retry_prompt = single_prompt
                if retry:
                    retry_prompt += "\nReturn exactly one prediction for the requested id. Do not omit it."
                single_values = _request(
                    client,
                    f"factor_validation_generate_single_{retry}",
                    retry_prompt,
                    {"records": single, "factors": factors},
                    "predictions",
                    int(settings["schema_retries"]),
                )
                if any(str(item.get("factor_conditioned", "")).strip() for item in single_values):
                    break
            for item in single_values:
                if str(item.get("id")) == record["id"]:
                    by_id[record["id"]] = item
            if not str(by_id.get(record["id"], {}).get("factor_conditioned", "")).strip() and len(single_values) == 1:
                by_id[record["id"]] = single_values[0]
        values.extend(by_id.values())
    by_id = {str(item.get("id")): item for item in values}
    result = []
    for record in records:
        item = by_id.get(record["id"], {})
        task_only = task_by_id.get(record["id"], "")
        factor_conditioned = str(item.get("factor_conditioned", "")).strip()
        if not task_only or not factor_conditioned:
            raise ValueError(f"missing held-out prediction for record={record['id']}")
        result.append({
            "id": record["id"],
            "abstract": record["abstract"],
            "target": record["title"],
            "task_only": task_only,
            "factor_conditioned": factor_conditioned,
            "used_factor_ids": [str(value) for value in item.get("used_factor_ids", [])],
        })
    return result


def _process_user(row: dict[str, Any], settings: dict[str, Any], client: TeacherClient) -> dict[str, Any]:
    user_id = str(row["id"])
    profiles = _unique_profile(row)
    build, heldout = _split_profile(profiles, user_id, float(settings["holdout_fraction"]), int(settings["seed"]), int(settings["min_build_count"]))
    drafts = _drafts(build, settings, client)
    proposals = _extract(drafts, settings, client)
    factors = _merge(proposals, settings, client)
    predictions = _heldout_predictions(heldout, factors, settings, client) if factors else []
    for item in predictions:
        task_score = score(item["task_only"], item["target"])
        factor_score = score(item["factor_conditioned"], item["target"])
        item["task_score"] = task_score
        item["factor_score"] = factor_score
        item["delta_rouge_1"] = factor_score["rouge_1"] - task_score["rouge_1"]
        item["delta_rouge_l"] = factor_score["rouge_l"] - task_score["rouge_l"]
    return {
        "user_id": user_id,
        "source_sample_id": user_id,
        "build_count": len(build),
        "heldout_count": len(heldout),
        "factor_count": len(factors),
        "factors": factors,
        "proposals": proposals,
        "build_trajectories": drafts,
        "predictions": predictions,
    }


def _select_rows(rows: list[dict[str, Any]], settings: dict[str, Any]) -> list[dict[str, Any]]:
    eligible = [row for row in rows if len(_unique_profile(row)) >= int(settings["min_profile_count"])]
    if len(eligible) < int(settings["users"]):
        raise ValueError(f"only {len(eligible)} users meet min_profile_count={settings['min_profile_count']}")
    rng = random.Random(int(settings["seed"]))
    indices = list(range(len(eligible)))
    rng.shuffle(indices)
    chosen = set(indices[: int(settings["users"])])
    return [row for index, row in enumerate(eligible) if index in chosen]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _paired_bootstrap(values: list[float], seed: int, samples: int) -> dict[str, float]:
    if not values:
        return {"low": 0.0, "mid": 0.0, "high": 0.0}
    rng = random.Random(seed)
    size = len(values)
    means = sorted(
        sum(values[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    return {
        "low": means[int(0.025 * (samples - 1))],
        "mid": _mean(values),
        "high": means[int(0.975 * (samples - 1))],
    }


def _report(results: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
    predictions = [item for result in results for item in result["predictions"]]
    task_r1 = [item["task_score"]["rouge_1"] for item in predictions]
    factor_r1 = [item["factor_score"]["rouge_1"] for item in predictions]
    task_rl = [item["task_score"]["rouge_l"] for item in predictions]
    factor_rl = [item["factor_score"]["rouge_l"] for item in predictions]
    deltas_rl = [item["delta_rouge_l"] for item in predictions]
    wins = sum(delta > 1e-12 for delta in deltas_rl)
    ties = sum(abs(delta) <= 1e-12 for delta in deltas_rl)
    losses = len(deltas_rl) - wins - ties
    factor_support = [factor["support_count"] for result in results for factor in result["factors"]]
    return {
        "protocol": "full_profile_rewrite_factor_leaveout_v1",
        "settings": settings,
        "users": len(results),
        "users_with_factors": sum(result["factor_count"] > 0 for result in results),
        "users_without_factors": sum(result["factor_count"] == 0 for result in results),
        "build_records": sum(result["build_count"] for result in results),
        "heldout_records": len(predictions),
        "factors": {
            "total": sum(result["factor_count"] for result in results),
            "mean_per_user": _mean([result["factor_count"] for result in results]),
            "mean_support_count": _mean(factor_support),
        },
        "task_only": {"rouge_1": _mean(task_r1), "rouge_l": _mean(task_rl)},
        "factor_conditioned": {"rouge_1": _mean(factor_r1), "rouge_l": _mean(factor_rl)},
        "delta": {
            "rouge_1": _mean([factor_r1[i] - task_r1[i] for i in range(len(predictions))]),
            "rouge_l": _mean(deltas_rl),
            "rouge_l_paired_bootstrap_95_ci": _paired_bootstrap(
                deltas_rl,
                int(settings["seed"]),
                int(settings.get("bootstrap_samples", 5000)),
            ),
            "rouge_l_win_tie_loss": {"win": wins, "tie": ties, "loss": losses},
        },
        "factor_usage": {
            "records_with_any_factor": sum(bool(item["used_factor_ids"]) for item in predictions),
            "mean_used_factor_count": _mean([len(item["used_factor_ids"]) for item in predictions]),
        },
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    delta = report["delta"]
    lines = [
        "# Full-profile rewrite factor validation",
        "",
        f"- Protocol: `{report['protocol']}`",
        f"- Users: {report['users']}; build records: {report['build_records']}; held-out records: {report['heldout_records']}",
        f"- Users with/without a valid factor bank: {report['users_with_factors']}/{report['users_without_factors']}",
        "",
        "| Method | ROUGE-1 | ROUGE-L |",
        "|---|---:|---:|",
        f"| Task-only | {report['task_only']['rouge_1']:.6f} | {report['task_only']['rouge_l']:.6f} |",
        f"| Factor-conditioned | {report['factor_conditioned']['rouge_1']:.6f} | {report['factor_conditioned']['rouge_l']:.6f} |",
        f"| Delta | {delta['rouge_1']:+.6f} | {delta['rouge_l']:+.6f} |",
        "",
        f"ROUGE-L win/tie/loss: {delta['rouge_l_win_tie_loss']['win']}/{delta['rouge_l_win_tie_loss']['tie']}/{delta['rouge_l_win_tie_loss']['loss']}.",
        f"Paired bootstrap 95% CI for ROUGE-L delta: [{delta['rouge_l_paired_bootstrap_95_ci']['low']:.6f}, {delta['rouge_l_paired_bootstrap_95_ci']['high']:.6f}].",
        "",
        "This is a historical leave-out validation. Reference titles are used only for offline scoring.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: dict[str, Any]) -> dict[str, Any]:
    settings = dict(config["factor_validation"])
    settings["seed"] = int(config["project"]["seed"])
    source = resolve_path(settings["source"])
    output_dir = resolve_path(settings["output_dir"])
    result_dir = resolve_path(settings["result_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    rows = _select_rows(read_jsonl(source), settings)
    client = TeacherClient(config["teacher"], resolve_path(config["teacher"]["cache_dir"]))
    concurrency = int(settings.get("concurrency", 1))
    results: list[dict[str, Any] | None] = [None] * len(rows)
    jobs = list(enumerate(rows))
    print(f"factor validation users={len(rows)}, concurrency={concurrency}", flush=True)

    def worker(job: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        return _process_user(job[1], settings, client)

    def on_result(job: tuple[int, dict[str, Any]], result: dict[str, Any], completed: int) -> None:
        results[job[0]] = result
        print(f"factor validation progress {completed}/{len(rows)} user={job[1]['id']} factors={result['factor_count']} heldout={result['heldout_count']}", flush=True)

    try:
        run_bounded(jobs, worker, on_result, max_workers=concurrency, thread_name_prefix="mevo-factor-validation")
    except BoundedJobError as failure:
        raise RuntimeError(f"factor validation failed for user={failure.job[1]['id']}: {failure.error}") from failure.error

    complete_results = [result for result in results if result is not None]
    report = _report(complete_results, settings)
    write_jsonl(output_dir / "factor_banks.jsonl", complete_results)
    write_jsonl(output_dir / "heldout_predictions.jsonl", [item for result in complete_results for item in result["predictions"]])
    (result_dir / "factor_validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, result_dir / "factor_validation_report.md")
    print(f"factor validation report -> {result_dir / 'factor_validation_report.md'}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate stable rewrite factors from full user profiles")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
