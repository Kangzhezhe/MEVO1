"""Build target-blind 2-task + 2-factor + 6-mutation Teacher slates."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from common.metrics import score
from common.profile_style import style_summary
from common.prompts import render_prompt
from common.runtime import load_stage
from common.teacher import TeacherClient
from common.utils import load_config, read_jsonl, resolve_path, write_jsonl


MUTATION_PLAN = (
    ("task_0", "f1"),
    ("task_1", "f1"),
    ("factor_0", "f2"),
    ("factor_1", "f2"),
    ("task_0", "f3"),
    ("factor_0", "f3"),
)


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def _directions(profile_titles: list[str]) -> list[dict]:
    style = style_summary(profile_titles)
    return [
        {
            "factor_id": "f1",
            "type": "content",
            "direction": (
                "Preserve the abstract's central technical object, method, and contribution; "
                "never import the topic of an unrelated historical paper."
            ),
            "condition": "content fidelity",
        },
        {
            "factor_id": "f2",
            "type": "style",
            "direction": style,
            "condition": "user historical title style",
        },
        {
            "factor_id": "f3",
            "type": "compression",
            "direction": (
                "Remove generic or redundant wording while retaining the main technical terms; "
                "keep title length compatible with the historical style summary."
            ),
            "condition": "concise scientific title",
        },
    ]


def _parse(payload: dict) -> tuple[list[str], list[str], list[dict]]:
    if not isinstance(payload, dict):
        raise ValueError("Response must be one JSON object")
    task = payload.get("task_seeds")
    factor = payload.get("factor_seeds")
    mutations = payload.get("mutations")
    if not isinstance(task, list) or len(task) != 2:
        raise ValueError("task_seeds must contain exactly two titles")
    if not isinstance(factor, list) or len(factor) != 2:
        raise ValueError("factor_seeds must contain exactly two titles")
    if not isinstance(mutations, list) or len(mutations) != 6:
        raise ValueError("mutations must contain exactly six objects")
    task_titles = [str(value).strip().strip('"') for value in task]
    factor_titles = [str(value).strip().strip('"') for value in factor]
    parsed_mutations = []
    for index, (value, expected) in enumerate(zip(mutations, MUTATION_PLAN)):
        if isinstance(value, str):
            parent, factor_id = expected
            title = value.strip().strip('"')
        elif isinstance(value, dict):
            parent = str(value.get("parent", "")).strip()
            factor_id = str(value.get("factor_id", "")).strip()
            title = str(value.get("title", "")).strip().strip('"')
            if (parent, factor_id) != expected:
                raise ValueError(
                    f"mutation[{index}] must use parent={expected[0]}, factor_id={expected[1]}"
                )
        else:
            raise ValueError(f"mutation[{index}] must be a title string")
        if not title:
            raise ValueError(f"mutation[{index}] has an empty title")
        parsed_mutations.append({"parent": parent, "factor_id": factor_id, "title": title})
    all_titles = task_titles + factor_titles + [value["title"] for value in parsed_mutations]
    if any(not value for value in all_titles):
        raise ValueError("All ten candidate titles must be non-empty")
    return task_titles, factor_titles, parsed_mutations


def _fallback(source_text: str) -> tuple[list[str], list[str], list[dict]]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]+", source_text)
    topic = " ".join(words[:7]).title() or "The Current Scientific Problem"
    task = [f"A Study of {topic}", f"Methods for {topic}"]
    factor = [f"On {topic}", f"{topic}: Methods and Analysis"]
    templates = [
        f"Technical Methods for {topic}",
        f"Analyzing {topic}",
        f"Toward {topic}",
        f"New Perspectives on {topic}",
        topic,
        f"{topic} in Practice",
    ]
    mutations = [
        {"parent": parent, "factor_id": factor_id, "title": title}
        for (parent, factor_id), title in zip(MUTATION_PLAN, templates)
    ]
    return task, factor, mutations


def _unique_titles(values: list[str], count: int, seen: set[str] | None = None) -> list[str]:
    selected = []
    known = set(seen or ())
    for value in values:
        title = str(value).strip()
        key = _norm(title)
        if title and key and key not in known:
            selected.append(title)
            known.add(key)
            if len(selected) == count:
                break
    return selected


def _merge_diverse_candidates(
    row: dict,
    task_titles: list[str],
    factor_titles: list[str],
    mutations: list[dict],
) -> tuple[list[str], list[str], list[dict], bool]:
    """Reuse the previous target-blind slate as a diversity reservoir.

    The reservoir contains candidate text only; its target and offline scores
    are never copied. Content/user-style candidates supply the four seeds,
    while the new prompt remains the primary source for six mutation titles.
    """
    augmentation = row.get("_augmentation_candidates", [])
    content = [value["text"] for value in augmentation if value["type"] == "slate_content"]
    user_style = [
        value["text"] for value in augmentation if value["type"] == "slate_user_style"
    ]
    compressed = [
        value["text"] for value in augmentation if value["type"] == "slate_compressed"
    ]
    merged_task = _unique_titles(content + task_titles, 2)
    seed_norms = {_norm(value) for value in merged_task}
    merged_factor = _unique_titles(user_style + factor_titles + content, 2, seed_norms)
    seed_norms.update(_norm(value) for value in merged_factor)
    generated_mutations = [value["title"] for value in mutations]
    mutation_pool = (
        generated_mutations
        + compressed
        + task_titles
        + factor_titles
        + [value["text"] for value in augmentation]
    )
    fallback_task, fallback_factor, fallback_mutations = _fallback(row["source_text"])
    if len(merged_task) < 2:
        merged_task = _unique_titles(merged_task + fallback_task, 2)
    seed_norms = {_norm(value) for value in merged_task}
    if len(merged_factor) < 2:
        merged_factor = _unique_titles(
            merged_factor + fallback_factor, 2, seed_norms
        )
    seed_norms.update(_norm(value) for value in merged_factor)
    mutation_pool += [value["title"] for value in fallback_mutations]
    mutation_titles = _unique_titles(mutation_pool, 6, seed_norms)
    if len(merged_task) != 2 or len(merged_factor) != 2 or len(mutation_titles) != 6:
        raise ValueError(f"Could not construct ten unique candidates for sample={row['id']}")
    merged_mutations = [
        {"parent": parent, "factor_id": factor_id, "title": title}
        for (parent, factor_id), title in zip(MUTATION_PLAN, mutation_titles)
    ]
    return merged_task, merged_factor, merged_mutations, bool(augmentation)


def _one(row: dict, config: dict, client: TeacherClient) -> dict:
    settings = config["pipeline_slate_generation"]
    target_norm = _norm(row["target"])
    retrieved_titles = [
        str(item.get("title", "")).strip()
        for item in row["retrieved_profile"]
        if str(item.get("title", "")).strip()
        and _norm(item.get("title", "")) != target_norm
    ]
    full_titles = [
        str(item.get("title", "")).strip()
        for item in row.get("profile", [])
        if str(item.get("title", "")).strip()
        and _norm(item.get("title", "")) != target_norm
    ]
    factors = _directions(full_titles)
    prompt = render_prompt(
        "lamp5/08_pipeline_slate_v1.txt",
        source_text=row["source_text"],
        factors=json.dumps(factors, ensure_ascii=False),
        profile_titles=json.dumps(retrieved_titles, ensure_ascii=False),
    )
    context = {**row, "pipeline_slate": True}
    payload, raw = client.json("candidate_pipeline_slate_v1", prompt, context)
    repairs = 0
    fallback = False
    while True:
        try:
            task_titles, factor_titles, mutations = _parse(payload)
            break
        except (AttributeError, TypeError, ValueError) as error:
            if repairs >= int(settings.get("schema_retries", 2)):
                task_titles, factor_titles, mutations = _fallback(row["source_text"])
                fallback = True
                break
            repairs += 1
            repair_prompt = (
                prompt
                + "\n\nSCHEMA REPAIR: "
                + str(error)
                + ". Return the complete required JSON object again with ten distinct titles."
            )
            payload, raw = client.json(
                f"candidate_pipeline_slate_repair_{repairs}", repair_prompt, context
            )

    task_titles, factor_titles, mutations, augmented = _merge_diverse_candidates(
        row, task_titles, factor_titles, mutations
    )

    candidates = []
    seed_by_slot: dict[str, str] = {}
    for index, title in enumerate(task_titles):
        candidate_id = f"{row['id']}_task_{index}"
        seed_by_slot[f"task_{index}"] = candidate_id
        candidates.append({
            "candidate_id": candidate_id,
            "type": "task_seed",
            "used_factors": [],
            "text": title,
            "scores": score(title, row["target"]),
        })
    for index, title in enumerate(factor_titles):
        candidate_id = f"{row['id']}_factor_{index}"
        seed_by_slot[f"factor_{index}"] = candidate_id
        candidates.append({
            "candidate_id": candidate_id,
            "type": "factor_seed",
            "used_factors": ["f1", "f2", "f3"],
            "text": title,
            "scores": score(title, row["target"]),
        })
    mutation_rows = []
    for index, value in enumerate(mutations):
        parent_id = seed_by_slot[value["parent"]]
        candidate_id = f"{row['id']}_{value['parent']}_mut_{value['factor_id']}_{index}"
        mutation_rows.append({
            "candidate_id": candidate_id,
            "type": "mutation",
            "parent_id": parent_id,
            "factor_id": value["factor_id"],
            "operation_trace": "One-call target-blind mutation using the declared factor.",
            "text": value["title"],
            "scores": score(value["title"], row["target"]),
        })
    visible_row = {key: value for key, value in row.items() if not key.startswith("_")}
    return {
        **visible_row,
        "factors": factors,
        "factor_metadata": {"method": "stable_profile_content_style_compression"},
        "candidates": candidates,
        "mutations": mutation_rows,
        "preferences": [],
        "seed_metadata": {
            "prompt_version": "lamp5-pipeline-slate-v1",
            "model": client.config["model"],
            "schema_repairs": repairs,
            "fallback": fallback,
            "diversity_reservoir_used": augmented,
            "raw_response": raw,
        },
        "metric_metadata": {
            "primary": "rouge_l",
            "implementation": "lamp_official_rouge",
        },
    }


def build(config: dict) -> Path:
    settings = config["pipeline_slate_generation"]
    data = config["data"]
    raw = resolve_path(data["raw_root"]) / "train"
    prepare = load_stage("01_prepare.py")
    retrieve = load_stage("02_retrieve.py")
    rows = prepare.normalize_lamp5(
        raw / "train_questions.json",
        raw / "train_outputs.json",
        "train",
        int(settings.get("limit", 2000)),
        int(config["project"]["seed"]),
    )
    retrieval = config["retrieval"]
    for row in rows:
        row["retrieved_profile"] = retrieve.rank_profile(
            row["source_text"],
            row["profile"],
            int(retrieval["top_k"]),
            float(retrieval["k1"]),
            float(retrieval["b"]),
        )
    augmentation_path = settings.get("augmentation_source")
    if augmentation_path:
        augmentation_by_id = {
            str(value["id"]): [
                {
                    "type": str(candidate.get("type", "")),
                    "text": str(candidate.get("text", "")).strip(),
                }
                for candidate in value.get("candidates", [])
            ]
            for value in read_jsonl(resolve_path(augmentation_path))
        }
        missing = [str(row["id"]) for row in rows if str(row["id"]) not in augmentation_by_id]
        if missing:
            raise ValueError(
                f"augmentation_source is missing {len(missing)} requested IDs; first={missing[0]}"
            )
        for row in rows:
            row["_augmentation_candidates"] = augmentation_by_id[str(row["id"])]
    client = TeacherClient(
        config["teacher"],
        resolve_path(settings.get("cache_dir", "dataset/cache/teacher-pipeline-slate")),
    )
    concurrency = int(settings.get("concurrency", 16))
    results: list[dict | None] = [None] * len(rows)
    print(f"Teacher pipeline-slate operations={len(rows)}, concurrency={concurrency}", flush=True)
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="pipeline-slate") as executor:
        futures: dict[Future, tuple[int, str]] = {
            executor.submit(_one, row, config, client): (index, str(row["id"]))
            for index, row in enumerate(rows)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            index, sample_id = futures[future]
            try:
                results[index] = future.result()
            except Exception as error:
                raise RuntimeError(
                    f"Teacher pipeline slate failed for sample={sample_id}: {error}"
                ) from error
            if completed % 10 == 0 or completed == len(rows):
                print(
                    f"Teacher pipeline-slate progress {completed}/{len(rows)} sample={sample_id}",
                    flush=True,
                )
    output_dir = resolve_path(data["processed_root"]) / str(data["processed_split"])
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "06_scored.jsonl"
    write_jsonl(destination, results)
    print(f"Teacher pipeline slates -> {destination}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one-call pipeline-shaped Teacher slates")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
