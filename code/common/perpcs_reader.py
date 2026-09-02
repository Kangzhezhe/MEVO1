"""阶段 01：读取 LaMP-5 文件并统一成后续阶段使用的 JSONL。

支持官方 questions/outputs 文件和 Per-Pcs 按用户聚合的处理版本。这里只做
格式归一化和样本截断，不调用 Teacher，也不使用 gold title 生成任何输入；
gold 仅作为离线监督字段保存，供阶段 06 评分。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import ijson

from common.utils import load_config, read_json, resolve_path, write_jsonl


PREFIX = re.compile(
    r"^\s*Generate\s+(?:a|the)\s+title\s+for\s+the\s+following\s+abstract\s+of\s+a\s+paper\s*:\s*",
    re.IGNORECASE,
)

PERPCS_PARTITIONS = {
    "base": Path("user_base_LLM.json"),
    "reserve": Path("user_reserve_10_percent.json"),
    "anchor": Path("user_anchor_candidate.json"),
    "test": Path("test_100/user_test_100.json"),
}
MISSING_PROFILE_ABSTRACTS = {
    "",
    "without abstract",
    "no abstract available",
    "no abstract available.",
}


def _as_rows(payload: Any, label: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in (label, "data", "outputs", "questions", "golds"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Could not find a list of {label} in LaMP file")


def _question_rows(path: str | Path, limit: int) -> list[dict[str, Any]]:
    """Stream the large official questions array and stop early for pilot runs."""
    rows = []
    with Path(path).open("rb") as handle:
        for row in ijson.items(handle, "item"):
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        # Keeps compatibility with a possible wrapped questions payload.
        return _as_rows(read_json(path), "questions")
    return rows


def _target(row: dict[str, Any]) -> str:
    for key in ("output", "target", "answer", "title", "gold"):
        if key in row:
            value = row[key]
            if isinstance(value, list):
                value = value[0]
            return str(value)
    raise ValueError(f"Output id={row.get('id')} has no output field")


def _profile_rows(
    payload: Any,
    *,
    sample_label: str,
    drop_missing_abstracts: bool,
) -> tuple[list[dict[str, str]], int]:
    if not isinstance(payload, list):
        raise ValueError(f"{sample_label} profile must be a list")
    profile = []
    filtered = 0
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{sample_label} profile[{index}] must be an object")
        title = str(item.get("title", "")).strip()
        abstract = str(item.get("abstract", "")).strip()
        if not title or (
            drop_missing_abstracts and abstract.casefold() in MISSING_PROFILE_ABSTRACTS
        ):
            filtered += 1
            continue
        profile.append(
            {
                "id": str(item.get("id", index)),
                "title": title,
                "abstract": abstract,
            }
        )
    return profile, filtered


def normalize_lamp5(
    questions_path: str | Path,
    outputs_path: str | Path,
    split: str,
    limit: int = 0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    questions = _question_rows(questions_path, limit)
    outputs = _as_rows(read_json(outputs_path), "outputs")
    output_by_id = {str(row["id"]): _target(row) for row in outputs}
    normalized = []
    for question in questions:
        sample_id = str(question["id"])
        if sample_id not in output_by_id:
            raise ValueError(f"Question id={sample_id} has no matching output")
        query = str(question.get("input", question.get("query", ""))).strip()
        if not query:
            raise ValueError(f"Question id={sample_id} has an empty input")
        profile = []
        for index, item in enumerate(question.get("profile", [])):
            profile.append(
                {
                    "id": str(item.get("id", index)),
                    "title": str(item.get("title", "")).strip(),
                    "abstract": str(item.get("abstract", "")).strip(),
                }
            )
        normalized.append(
            {
                "id": sample_id,
                "instruction": "Generate a title for the following abstract of a paper",
                "query": query,
                "source_text": PREFIX.sub("", query, count=1).strip(),
                "target": output_by_id[sample_id].strip(),
                "profile": profile,
                "split": split,
            }
        )
    # Streaming already selects the first N official samples. With limit=0 the
    # full split is retained; keeping official order makes runs reproducible.
    return normalized


def normalize_perpcs_lamp5(
    source_path: str | Path,
    split: str,
    partition: str,
    limit: int = 0,
    seed: int = 42,
    drop_missing_profile_abstracts: bool = True,
) -> list[dict[str, Any]]:
    """Flatten Per-Pcs user records into one target-isolated row per query.

    ``limit`` counts flattened queries rather than users. The source order is
    retained, matching the official reader's deterministic pilot behavior.
    ``seed`` remains in the interface so both input formats share one contract.
    """
    del seed
    if partition not in PERPCS_PARTITIONS:
        raise ValueError(
            f"Unknown Per-Pcs partition={partition!r}; expected one of "
            f"{sorted(PERPCS_PARTITIONS)}"
        )

    normalized = []
    seen_users: set[str] = set()
    seen_samples: set[str] = set()
    with Path(source_path).open("rb") as handle:
        for user_index, user in enumerate(ijson.items(handle, "item")):
            if not isinstance(user, dict):
                raise ValueError(f"Per-Pcs user[{user_index}] must be an object")
            user_id = str(user.get("user_id", "")).strip()
            if not user_id:
                raise ValueError(f"Per-Pcs user[{user_index}] has no user_id")
            if user_id in seen_users:
                raise ValueError(f"Per-Pcs partition={partition} repeats user_id={user_id}")
            seen_users.add(user_id)

            profile, filtered = _profile_rows(
                user.get("profile", []),
                sample_label=f"user={user_id}",
                drop_missing_abstracts=drop_missing_profile_abstracts,
            )
            queries = user.get("query", [])
            if not isinstance(queries, list) or not queries:
                raise ValueError(f"Per-Pcs user={user_id} has no query records")

            for query_index, query_row in enumerate(queries):
                if not isinstance(query_row, dict):
                    raise ValueError(
                        f"Per-Pcs user={user_id} query[{query_index}] must be an object"
                    )
                sample_id = str(query_row.get("id", "")).strip()
                if not sample_id:
                    raise ValueError(f"Per-Pcs user={user_id} query[{query_index}] has no id")
                if sample_id in seen_samples:
                    raise ValueError(
                        f"Per-Pcs partition={partition} repeats query id={sample_id}"
                    )
                query = str(query_row.get("input", "")).strip()
                target = _target(query_row).strip()
                source_text = PREFIX.sub("", query, count=1).strip()
                if not query or not source_text:
                    raise ValueError(f"Per-Pcs query id={sample_id} has an empty input")
                if not target:
                    raise ValueError(f"Per-Pcs query id={sample_id} has an empty gold title")

                normalized.append(
                    {
                        "id": sample_id,
                        "user_id": user_id,
                        "instruction": "Generate a title for the following abstract of a paper",
                        "query": query,
                        "source_text": source_text,
                        "target": target,
                        "profile": profile,
                        "split": split,
                        "source_format": "perpcs_lamp5",
                        "source_partition": partition,
                        "filtered_profile_records": filtered,
                    }
                )
                seen_samples.add(sample_id)
                if limit > 0 and len(normalized) >= limit:
                    return normalized
    if not normalized:
        raise ValueError(f"Per-Pcs source={source_path} contains no query records")
    return normalized


def prepare(questions: Path, outputs: Path, destination: Path, split: str, limit: int, seed: int) -> None:
    rows = normalize_lamp5(questions, outputs, split, limit, seed)
    write_jsonl(destination, rows)
    print(f"prepared {len(rows)} LaMP-5 samples -> {destination}")


def prepare_perpcs(
    source: Path,
    destination: Path,
    split: str,
    partition: str,
    limit: int,
    seed: int,
    drop_missing_profile_abstracts: bool,
) -> None:
    rows = normalize_perpcs_lamp5(
        source,
        split,
        partition,
        limit,
        seed,
        drop_missing_profile_abstracts,
    )
    write_jsonl(destination, rows)
    users = {row["user_id"] for row in rows}
    filtered_by_user = {
        row["user_id"]: int(row["filtered_profile_records"]) for row in rows
    }
    filtered = sum(filtered_by_user.values())
    print(
        f"prepared {len(rows)} Per-Pcs LaMP-5 samples from {len(users)} users "
        f"(partition={partition}, filtered_profile_records={filtered}) -> {destination}"
    )


def main() -> None:
    from common.runtime import config_parser, stage_path

    parser = config_parser("01 - Normalize official or Per-Pcs LaMP-5 data")
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--outputs", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    split = str(data["split"])
    source_format = str(data.get("source_format", "official_lamp5"))
    destination = stage_path(config, "prepare")
    if source_format == "perpcs_lamp5":
        partition = str(data["perpcs_partition"])
        if partition not in PERPCS_PARTITIONS:
            raise ValueError(
                f"data.perpcs_partition must be one of {sorted(PERPCS_PARTITIONS)}"
            )
        perpcs_root = resolve_path(data["perpcs_root"])
        source = args.source or perpcs_root / PERPCS_PARTITIONS[partition]
        prepare_perpcs(
            source,
            destination,
            split,
            partition,
            int(data["limit"]),
            int(config["project"]["seed"]),
            bool(data.get("drop_missing_profile_abstracts", True)),
        )
        return
    if source_format != "official_lamp5":
        raise ValueError(
            "data.source_format must be 'official_lamp5' or 'perpcs_lamp5'"
        )
    raw = resolve_path(data["raw_root"]) / split
    prepare(
        args.questions or raw / f"{split}_questions.json",
        args.outputs or raw / f"{split}_outputs.json",
        destination,
        split,
        int(data["limit"]),
        int(config["project"]["seed"]),
    )


if __name__ == "__main__":
    main()
