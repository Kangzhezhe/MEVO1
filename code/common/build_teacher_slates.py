"""Generate one diverse target-blind Teacher candidate slate per train query."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from common.metrics import score
from common.profile_style import style_summary
from common.prompts import render_prompt
from common.runtime import load_stage
from common.teacher import TeacherClient
from common.utils import load_config, resolve_path, write_jsonl


EXPECTED = {"content": 2, "user_style": 2, "compressed": 2}


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def _parse(payload: dict, count: int) -> list[dict]:
    values = payload.get("candidates")
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"Expected exactly {count} candidate objects")
    parsed = []
    seen = set()
    strategies = Counter()
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("Each candidate must be an object")
        strategy = str(value.get("strategy", "")).strip()
        title = str(value.get("title", "")).strip().strip('"')
        if strategy not in EXPECTED or not title or _norm(title) in seen:
            raise ValueError("Candidate strategy/title is invalid or duplicated")
        seen.add(_norm(title)); strategies[strategy] += 1
        parsed.append({"strategy": strategy, "title": title})
    if strategies != EXPECTED:
        raise ValueError(f"Strategy counts must be {EXPECTED}, got {dict(strategies)}")
    return parsed


def _fallback_titles(source_text: str, existing: list[dict], count: int) -> list[dict]:
    """Deterministic last resort used only after all Teacher schema repairs."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]+", source_text)
    topic = " ".join(words[:8]).title() or "The Current Scientific Problem"
    templates = {
        "content": [f"A Study of {topic}", f"Methods for {topic}", f"Technical Analysis of {topic}"],
        "user_style": [f"On {topic}", f"{topic}: Methods and Analysis", f"Toward {topic}"],
        "compressed": [topic, f"Analyzing {topic}", f"{topic} in Practice"],
    }
    by_strategy = {strategy: [] for strategy in EXPECTED}
    seen = set()
    for item in existing:
        strategy = str(item.get("strategy", "")); title = str(item.get("title", "")).strip()
        key = _norm(title)
        if strategy in EXPECTED and title and key not in seen and len(by_strategy[strategy]) < EXPECTED[strategy]:
            by_strategy[strategy].append(title); seen.add(key)
    for strategy, required in EXPECTED.items():
        for title in templates[strategy]:
            if len(by_strategy[strategy]) >= required:
                break
            if _norm(title) not in seen:
                by_strategy[strategy].append(title); seen.add(_norm(title))
    result = [
        {"strategy": strategy, "title": title}
        for strategy in ("content", "user_style", "compressed")
        for title in by_strategy[strategy]
    ]
    if len(result) != count:
        raise ValueError(f"Fallback produced {len(result)} candidates, expected {count}")
    return result


def _one(row: dict, config: dict, client: TeacherClient) -> dict:
    settings = config["slate_generation"]
    count = int(settings.get("candidate_count", 6))
    history_titles = [
        str(item.get("title", "")).strip()
        for item in row["retrieved_profile"]
        if str(item.get("title", "")).strip() and _norm(item.get("title", "")) != _norm(row["target"])
    ]
    summary = style_summary([
        str(item.get("title", "")).strip()
        for item in row.get("profile", [])
        if str(item.get("title", "")).strip() and _norm(item.get("title", "")) != _norm(row["target"])
    ])
    prompt = render_prompt(
        "lamp5/07_candidate_slate_v1.txt",
        count=count,
        source_text=row["source_text"],
        style_summary=summary,
        profile_titles=json.dumps(history_titles, ensure_ascii=False),
    )
    payload, raw = client.json("candidate_slate_v1", prompt, {**row, "slate_count": count})
    repairs = 0
    fallback = False
    while True:
        try:
            generated = _parse(payload, count)
            break
        except (AttributeError, TypeError, ValueError) as error:
            if repairs >= int(settings.get("schema_retries", 2)):
                partial = []
                if isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
                    for value in payload["candidates"]:
                        if isinstance(value, dict):
                            strategy = str(value.get("strategy", "")); title = str(value.get("title", "")).strip()
                            if strategy in EXPECTED and title:
                                partial.append({"strategy": strategy, "title": title})
                generated = _fallback_titles(row["source_text"], partial, count)
                if len(generated) != count:
                    raise ValueError(f"Could not construct {count} unique slate candidates") from error
                fallback = True
                break
            repairs += 1
            repair_prompt = (
                prompt + "\n\nSCHEMA REPAIR: " + str(error)
                + ". Return exactly two candidates for each required strategy and no extra text."
            )
            payload, raw = client.json(
                f"candidate_slate_repair_{repairs}", repair_prompt, {**row, "slate_count": count}
            )
    factors = [{
        "factor_id": "profile_style",
        "type": "style",
        "evidence_ids": [str(item["id"]) for item in row["retrieved_profile"]],
        "evidence_summary": "Deterministic aggregate over target-filtered historical titles.",
        "direction": summary,
        "condition": "user historical title style",
    }]
    candidates = []
    for index, item in enumerate(generated):
        values = score(item["title"], row["target"])
        candidates.append({
            "candidate_id": f"{row['id']}_slate_{index}",
            "type": f"slate_{item['strategy']}",
            "text": item["title"],
            "scores": values,
        })
    return {
        **row,
        "factors": factors,
        "factor_metadata": {"method": "target_blind_profile_style_summary"},
        "candidates": candidates,
        "mutations": [],
        "preferences": [],
        "seed_metadata": {
            "prompt_version": "lamp5-candidate-slate-v1",
            "model": client.config["model"],
            "schema_repairs": repairs,
            "fallback": fallback,
            "raw_response": raw,
        },
        "metric_metadata": {"primary": "rouge_l", "implementation": "lamp_official_rouge"},
    }


def build(config: dict) -> Path:
    settings = config["slate_generation"]
    data = config["data"]
    raw = resolve_path(data["raw_root"]) / "train"
    prepare = load_stage("01_prepare.py")
    retrieve = load_stage("02_retrieve.py")
    rows = prepare.normalize_lamp5(
        raw / "train_questions.json", raw / "train_outputs.json", "train",
        int(settings.get("limit", 2000)), int(config["project"]["seed"]),
    )
    retrieval = config["retrieval"]
    for row in rows:
        row["retrieved_profile"] = retrieve.rank_profile(
            row["source_text"], row["profile"], int(retrieval["top_k"]),
            float(retrieval["k1"]), float(retrieval["b"]),
        )
    client = TeacherClient(config["teacher"], resolve_path(settings.get("cache_dir", "dataset/cache/teacher-slate")))
    concurrency = int(settings.get("concurrency", 16))
    results: list[dict | None] = [None] * len(rows)
    print(f"Teacher slate operations={len(rows)}, concurrency={concurrency}", flush=True)
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="teacher-slate") as executor:
        futures: dict[Future, tuple[int, str]] = {
            executor.submit(_one, row, config, client): (index, str(row["id"]))
            for index, row in enumerate(rows)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            index, sample_id = futures[future]
            try:
                results[index] = future.result()
            except Exception as error:
                raise RuntimeError(f"Teacher slate failed for sample={sample_id}: {error}") from error
            if completed % 10 == 0 or completed == len(rows):
                print(f"Teacher slate progress {completed}/{len(rows)} sample={sample_id}", flush=True)
    output_dir = resolve_path(data["processed_root"]) / str(data["processed_split"])
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "06_scored.jsonl"
    write_jsonl(destination, results)
    print(f"Teacher slates -> {destination}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one-call Teacher candidate slates")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
