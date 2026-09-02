"""Build a large, API-free LaMP-5 Ranker warm-up set.

The gold train title is a supervised positive. Negatives deliberately separate
content from personalization: user-history titles preserve author style but
usually mismatch the current paper, while cross-user titles are selected for
lexical content overlap but come from another author. Gold titles never enter
dev/test inputs and this warm-up consumes the official train split only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from common.metrics import score
from common.profile_style import style_summary
from common.runtime import load_stage
from common.utils import load_config, resolve_path, write_jsonl


WORD = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "of", "on", "or", "the", "to", "using", "via", "we", "with",
}


def _tokens(text: str) -> set[str]:
    return {token for token in WORD.findall(str(text).lower()) if token not in STOP and len(token) > 1}


def _norm(text: str) -> str:
    # Deliberately discard punctuation/hyphen/LaTeX markup so variants such as
    # ``K-User``/``$K$-User`` or ``SuperTrust``/``Supertrust`` are treated as
    # the same title for leakage and duplicate filtering.
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def _profile_rank(row: dict) -> list[dict]:
    query = _tokens(row["source_text"])
    target = _norm(row["target"])
    ranked = []
    for item in row.get("profile", []):
        title = str(item.get("title", "")).strip()
        if not title or _norm(title) == target:
            continue
        evidence = _tokens(item.get("abstract", "")) | _tokens(title)
        similarity = len(query & evidence) / max(1, len(query | evidence))
        ranked.append((similarity, str(item.get("id", "")), item))
    ranked.sort(key=lambda value: (-value[0], value[1]))
    return [item for _, _, item in ranked]


def _content_index(rows: list[dict]) -> tuple[dict[str, list[int]], dict[str, float], list[set[str]]]:
    title_tokens = [_tokens(row["target"]) for row in rows]
    postings: dict[str, list[int]] = defaultdict(list)
    for index, tokens in enumerate(title_tokens):
        for token in tokens:
            postings[token].append(index)
    count = len(rows)
    idf = {token: math.log((count + 1) / (len(indices) + 1)) + 1.0 for token, indices in postings.items()}
    return postings, idf, title_tokens


def _content_negatives(
    index: int,
    row: dict,
    rows: list[dict],
    postings: dict[str, list[int]],
    idf: dict[str, float],
    title_tokens: list[set[str]],
    count: int,
) -> list[str]:
    scores: Counter[int] = Counter()
    for token in _tokens(row["source_text"]):
        for candidate_index in postings.get(token, []):
            if candidate_index != index:
                scores[candidate_index] += idf[token] / math.sqrt(max(1, len(title_tokens[candidate_index])))
    result = []
    seen = {_norm(row["target"])}
    for candidate_index, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
        title = str(rows[candidate_index]["target"]).strip()
        if _norm(title) in seen:
            continue
        seen.add(_norm(title)); result.append(title)
        if len(result) >= count:
            break
    # Deterministic fallback for abstracts whose vocabulary does not occur in
    # another title.
    cursor = (index + 1) % len(rows)
    while len(result) < count:
        title = str(rows[cursor]["target"]).strip()
        if cursor != index and _norm(title) not in seen:
            seen.add(_norm(title)); result.append(title)
        cursor = (cursor + 1) % len(rows)
    return result


def _group(
    row: dict,
    split: str,
    profile_negative_count: int,
    content_negative_titles: list[str],
    profile_context_count: int,
) -> tuple[dict, dict, list[dict]]:
    ranked_profile = _profile_rank(row)
    all_history_titles = [str(item.get("title", "")).strip() for item in ranked_profile]
    profile_titles = all_history_titles[:profile_context_count]
    factors = [{
        "factor_id": "profile_style",
        "type": "style",
        "direction": style_summary(all_history_titles),
        "condition": "user historical title style",
    }]
    candidates = [{
        "candidate_id": f"{row['id']}_gold",
        "type": "gold_positive",
        "text": str(row["target"]),
    }]
    seen = {_norm(row["target"])}
    for item in ranked_profile:
        title = str(item.get("title", "")).strip()
        if _norm(title) in seen:
            continue
        seen.add(_norm(title))
        candidates.append({
            "candidate_id": f"{row['id']}_profile_neg_{len(candidates)}",
            "type": "profile_negative",
            "text": title,
        })
        if sum(candidate["type"] == "profile_negative" for candidate in candidates) >= profile_negative_count:
            break
    for title in content_negative_titles:
        if _norm(title) in seen:
            continue
        seen.add(_norm(title))
        candidates.append({
            "candidate_id": f"{row['id']}_content_neg_{len(candidates)}",
            "type": "content_negative",
            "text": title,
        })
    sample_id = f"warmup_{split}:{row['id']}"
    base = {
        "sample_id": sample_id,
        "source_text": str(row["source_text"]),
        "factors": factors,
        "profile_titles": profile_titles,
        "profile_target_matches_filtered": len(row.get("profile", [])) - len(ranked_profile),
        "candidates": candidates,
    }
    label_scores = [score(candidate["text"], row["target"])["rouge_l"] for candidate in candidates]
    listwise = {**base, "label_scores": label_scores}
    positive = candidates[0]
    pairs = []
    for candidate, candidate_score in zip(candidates[1:], label_scores[1:]):
        pairs.append({
            **{key: base[key] for key in ("sample_id", "source_text", "factors", "profile_titles", "profile_target_matches_filtered")},
            "pair_id": f"{sample_id}:{positive['candidate_id']}>{candidate['candidate_id']}",
            "operation": "warmup_title_ranking",
            "chosen": positive,
            "rejected": candidate,
            "margin": 1.0 - float(candidate_score),
            "metric": "rouge_l",
        })
    return base, listwise, pairs


def build(config: dict) -> dict:
    settings = config["ranker_warmup"]
    data = config["data"]
    raw = resolve_path(data["raw_root"]) / "train"
    prepare = load_stage("01_prepare.py")
    rows = prepare.normalize_lamp5(
        raw / "train_questions.json", raw / "train_outputs.json", "train", 0,
        int(config["project"]["seed"]),
    )
    postings, idf, title_tokens = _content_index(rows)
    indices = list(range(len(rows)))
    random.Random(int(config["project"]["seed"])).shuffle(indices)
    validation_count = round(len(rows) * float(settings.get("validation_fraction", 0.1)))
    validation_indices = set(indices[:validation_count])
    output_dir = resolve_path(config["ranker"]["data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {split: {"groups": [], "listwise": [], "pairs": [], "labels": []} for split in ("train", "validation")}
    profile_negatives = int(settings.get("profile_negatives", 2))
    content_negatives = int(settings.get("content_negatives", 2))
    profile_context = int(settings.get("profile_context_titles", 8))
    for index, row in enumerate(rows):
        split = "validation" if index in validation_indices else "train"
        negatives = _content_negatives(index, row, rows, postings, idf, title_tokens, content_negatives)
        group, listwise, pairs = _group(row, split, profile_negatives, negatives, profile_context)
        target = artifacts[split]
        target["groups"].append(group); target["listwise"].append(listwise); target["pairs"].extend(pairs)
        target["labels"].append({"sample_id": group["sample_id"], "target": row["target"]})
    counts = {}
    for split, values in artifacts.items():
        write_jsonl(output_dir / f"{split}_candidates.jsonl", values["groups"])
        write_jsonl(output_dir / f"{split}_listwise.jsonl", values["listwise"])
        write_jsonl(output_dir / f"{split}_pairs.jsonl", values["pairs"])
        write_jsonl(output_dir / f"{split}_labels.jsonl", values["labels"])
        counts[split] = {"samples": len(values["groups"]), "pairs": len(values["pairs"]), "listwise_groups": len(values["listwise"])}
    manifest = {
        "protocol": "lamp5_train_title_warmup",
        "seed": int(config["project"]["seed"]),
        "source": str(raw),
        "profile_negatives": profile_negatives,
        "content_negatives": content_negatives,
        "profile_context_titles": profile_context,
        "train_ids": [row["sample_id"] for row in artifacts["train"]["groups"]],
        "validation_ids": [row["sample_id"] for row in artifacts["validation"]["groups"]],
        "counts": counts,
        "warning": "Warm-up uses official train gold titles as supervised positives; dev/test are untouched.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ranker warm-up data -> {output_dir}; counts={counts}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LaMP-5 title-ranking warm-up data")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
