"""Target-blind Teacher listwise ranking over a frozen candidate pool."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from common.evaluate import evaluate
from common.prompts import render_prompt
from common.teacher import TeacherClient
from common.utils import load_config, read_jsonl, resolve_path, write_jsonl


def _prompt(group: dict, aliases: list[dict]) -> str:
    return render_prompt(
        "lamp5/06_teacher_rank_v1.txt",
        source_text=group["source_text"],
        factors=json.dumps(
            [factor.get("direction", "") for factor in group.get("factors", [])],
            ensure_ascii=False,
        ),
        profile_titles=json.dumps(group.get("profile_titles", []), ensure_ascii=False),
        candidates=json.dumps(
            [{"candidate_id": item["alias"], "text": item["candidate"]["text"]} for item in aliases],
            ensure_ascii=False,
        ),
    )


def _validate(payload: dict, valid_ids: set[str]) -> list[dict]:
    ranking = payload.get("ranking")
    if not isinstance(ranking, list) or len(ranking) != len(valid_ids):
        raise ValueError("ranking must contain every candidate exactly once")
    ids = [str(item.get("candidate_id", "")) for item in ranking if isinstance(item, dict)]
    if len(ids) != len(ranking) or set(ids) != valid_ids or len(ids) != len(set(ids)):
        raise ValueError("ranking candidate IDs are not an exact permutation")
    if str(payload.get("selected_id", "")) != ids[0]:
        raise ValueError("selected_id must equal ranking[0].candidate_id")
    for item in ranking:
        item["score"] = float(item.get("score", 0.0))
        item["reason"] = str(item.get("reason", "")).strip()
    return ranking


def _complete_partial_ranking(payload: dict, ordered_ids: list[str]) -> tuple[list[dict], int]:
    """Preserve valid Teacher order and append omitted aliases at the tail."""
    valid_ids = set(ordered_ids)
    ranking = payload.get("ranking", []) if isinstance(payload, dict) else []
    completed = []
    seen = set()
    if isinstance(ranking, list):
        for item in ranking:
            if not isinstance(item, dict):
                continue
            candidate_id = str(item.get("candidate_id", ""))
            if candidate_id not in valid_ids or candidate_id in seen:
                continue
            seen.add(candidate_id)
            completed.append({
                "candidate_id": candidate_id,
                "score": float(item.get("score", 0.0)),
                "reason": str(item.get("reason", "")).strip(),
            })
    missing = [candidate_id for candidate_id in ordered_ids if candidate_id not in seen]
    completed.extend({
        "candidate_id": candidate_id,
        "score": 0.0,
        "reason": "Omitted by Teacher; appended deterministically after schema retries.",
    } for candidate_id in missing)
    return completed, len(missing)


def _one(group: dict, client: TeacherClient, schema_retries: int) -> dict:
    # Preserve candidate-list order while hiding candidate type, parent and
    # factor provenance from the Teacher.
    aliases = [
        {"alias": f"c{index}", "candidate": candidate}
        for index, candidate in enumerate(group["candidates"])
    ]
    alias_to_candidate = {item["alias"]: item["candidate"] for item in aliases}
    prompt = _prompt(group, aliases)
    payload, raw = client.json(
        "teacher_listwise_rank_v1",
        prompt,
        {"candidates": [item["alias"] for item in aliases]},
    )
    repairs = 0
    fallback_appended = 0
    while True:
        try:
            ranking = _validate(payload, set(alias_to_candidate))
            break
        except (TypeError, ValueError) as error:
            if repairs >= schema_retries:
                ranking, fallback_appended = _complete_partial_ranking(
                    payload, list(alias_to_candidate)
                )
                break
            repairs += 1
            repair_prompt = (
                prompt
                + "\n\nSCHEMA REPAIR: The previous payload was invalid: "
                + str(error)
                + ". Return all candidate IDs exactly once and no extra text."
            )
            payload, raw = client.json(
                f"teacher_listwise_rank_repair_{repairs}",
                repair_prompt,
                {"candidates": list(alias_to_candidate)},
            )
    ranked = []
    for rank, judgment in enumerate(ranking, 1):
        candidate = alias_to_candidate[str(judgment["candidate_id"])]
        ranked.append({
            **candidate,
            "ranker_score": float(len(ranking) - rank + 1),
            "teacher_score": float(judgment["score"]),
            "teacher_reason": judgment["reason"],
        })
    return {
        "sample_id": group["sample_id"],
        "selected_id": ranked[0]["candidate_id"],
        "prediction": ranked[0]["text"],
        "ranked_candidates": ranked,
        "protocol": "target_blind_teacher_listwise_ranker",
        "teacher_metadata": {
            "model": client.config["model"],
            "prompt_version": "lamp5-teacher-rank-v1",
            "schema_repairs": repairs,
            "fallback_candidates_appended": fallback_appended,
            "raw_response": raw,
        },
    }


def run(config: dict) -> Path:
    settings = config["teacher_judge"]
    candidates_path = resolve_path(settings["candidates_path"])
    groups = read_jsonl(candidates_path)
    output_dir = resolve_path(config["experiment"]["result_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    client = TeacherClient(config["teacher"], resolve_path(settings.get("cache_dir", "dataset/cache/teacher-ranker")))
    concurrency = int(settings.get("concurrency", 8))
    results: list[dict | None] = [None] * len(groups)
    print(f"Teacher listwise operations={len(groups)}, concurrency={concurrency}", flush=True)
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="teacher-ranker") as executor:
        futures: dict[Future, tuple[int, str]] = {
            executor.submit(_one, group, client, int(settings.get("schema_retries", 2))):
            (index, str(group["sample_id"]))
            for index, group in enumerate(groups)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            index, sample_id = futures[future]
            try:
                results[index] = future.result()
            except Exception as error:
                raise RuntimeError(f"Teacher ranking failed for sample={sample_id}: {error}") from error
            print(f"Teacher ranking progress {completed}/{len(groups)} sample={sample_id}", flush=True)
    destination = output_dir / "validation_predictions.jsonl"
    write_jsonl(destination, results)
    evaluate(
        destination,
        candidates_path,
        resolve_path(settings["labels_path"]),
        resolve_path(settings["manifest_path"]),
        output_dir / "validation_report.json",
        seed=int(config["project"]["seed"]),
        label_temperature=float(config["ranker"].get("listwise_temperature", 0.1)),
        bootstrap_samples=int(config["ranker"].get("evaluation_bootstrap_samples", 5000)),
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Target-blind Teacher candidate Ranker")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
