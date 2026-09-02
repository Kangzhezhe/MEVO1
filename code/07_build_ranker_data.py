"""阶段 07：把评分后的候选池转换为 Ranker 训练/验证数据。

这里执行 target isolation：输入保留 query、abstract、factor directions 和
候选文本，gold 分数只作为 label，绝不写进 Ranker 输入；因此该阶段是防止
答案泄露的关键检查点。
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from common.utils import load_config, read_jsonl, resolve_path, write_jsonl
from common.runtime import GLOBAL_CONFIG


# ---------- Ranker 可见的数据视图 ----------
# 这些 view 函数有意丢弃 target 和 candidate.scores，防止 gold 派生信息进入输入。
def _factor_view(factor: dict) -> dict:
    return {
        "factor_id": str(factor["factor_id"]),
        "type": str(factor["type"]),
        "direction": str(factor["direction"]),
        "condition": str(factor.get("condition", "")),
    }


def _candidate_view(candidate: dict) -> dict:
    result = {
        "candidate_id": str(candidate["candidate_id"]),
        "type": str(candidate["type"]),
        "text": str(candidate["text"]),
    }
    for key in ("used_factors", "parent_id", "factor_id", "operation_trace"):
        if key in candidate:
            result[key] = candidate[key]
    return result


def _normalize_title(text: str) -> str:
    """Normalize only for duplicate/leakage filtering, never as model input."""
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def _profile_titles_view(row: dict) -> tuple[list[str], int]:
    """Expose target-blind user history while removing exact target duplicates.

    LaMP profiles should represent past publications, but the current train500
    snapshot contains one history record whose title exactly equals the target.
    Such a record would make a profile-aware Ranker trivially leak the answer.
    The target is used only as a deny-list here and is never copied to Ranker
    inputs or outputs.
    """
    target = _normalize_title(row.get("target", ""))
    titles = []
    filtered = 0
    for item in row.get("retrieved_profile", []):
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        if target and _normalize_title(title) == target:
            filtered += 1
            continue
        titles.append(title)
    return titles, filtered


def _factor_views(row: dict) -> list[dict]:
    # Compatibility filter: old factor files may still contain the removed
    # fixed profile-style summary. It must not enter Ranker inputs.
    return [
        _factor_view(factor)
        for factor in row.get("factors", [])
        if str(factor.get("factor_id", "")) != "profile_style"
    ]


def _sample_id(row: dict) -> str:
    return str(row.get("_ranker_sample_id", row["id"]))


def _user_id(row: dict) -> str | None:
    value = row.get("user_id")
    return None if value is None else str(value)


def _group_view(row: dict) -> dict:
    candidates = [
        _candidate_view(candidate)
        for candidate in row["candidates"] + row.get("mutations", [])
    ]
    profile_titles, filtered_profile_target_matches = _profile_titles_view(row)
    result = {
        "sample_id": _sample_id(row),
        "source_text": str(row["source_text"]),
        "factors": _factor_views(row),
        "profile_titles": profile_titles,
        "profile_target_matches_filtered": filtered_profile_target_matches,
        "candidates": candidates,
    }
    if _user_id(row) is not None:
        result["user_id"] = _user_id(row)
    return result


def _listwise_view(row: dict, metric: str) -> dict:
    """Gold-derived scores are loss labels and never enter candidate text."""
    group = _group_view(row)
    raw_candidates = row["candidates"] + row.get("mutations", [])
    score_by_id = {
        str(candidate["candidate_id"]): float(candidate["scores"][metric])
        for candidate in raw_candidates
    }
    group["label_scores"] = [
        score_by_id[str(candidate["candidate_id"])] for candidate in group["candidates"]
    ]
    return group


def _pair_record(
    row: dict,
    candidate_by_id: dict[str, dict],
    pair_id: str,
    chosen_id: str,
    rejected_id: str,
    margin: float,
    metric: str,
) -> dict:
    profile_titles, filtered_profile_target_matches = _profile_titles_view(row)
    context = {
        "sample_id": _sample_id(row),
        "source_text": str(row["source_text"]),
        "factors": _factor_views(row),
        "profile_titles": profile_titles,
        "profile_target_matches_filtered": filtered_profile_target_matches,
    }
    if _user_id(row) is not None:
        context["user_id"] = _user_id(row)
    return {
        **context,
        "pair_id": pair_id,
        "operation": "candidate_ranking",
        "chosen": candidate_by_id[chosen_id],
        "rejected": candidate_by_id[rejected_id],
        "margin": margin,
        "metric": metric,
    }


def _pair_rows(
    row: dict,
    strategy: str,
    metric: str,
    minimum_margin: float,
    max_pairs_per_sample: int,
) -> list[dict]:
    """按配置将一个候选组转换为偏好 pair。

    parent_child 只比较 mutation 与其父候选；all_pairs 比较所有分差足够大的
    候选；hard_negative 保留 label 分差最小的 pair；top_pairs 只保留
    oracle 候选与非 oracle 候选的比较。配置 max_pairs_per_sample 用来避免
    大候选池主导训练。
    """
    raw_candidates = row["candidates"] + row.get("mutations", [])
    candidate_by_id = {
        candidate["candidate_id"]: _candidate_view(candidate) for candidate in raw_candidates
    }
    pairs = []
    if strategy == "parent_child":
        for preference in row.get("preferences", []):
            chosen_id = str(preference["chosen_id"])
            rejected_id = str(preference["rejected_id"])
            if chosen_id not in candidate_by_id or rejected_id not in candidate_by_id:
                raise ValueError(f"Preference for sample={row['id']} references an unknown candidate")
            pairs.append(
                _pair_record(
                    row,
                    candidate_by_id,
                    str(preference["id"]),
                    chosen_id,
                    rejected_id,
                    float(preference["margin"]),
                    str(preference["metric"]),
                )
            )
    elif strategy in {"all_pairs", "hard_negative", "top_pairs"}:
        oracle_score = max(float(candidate["scores"][metric]) for candidate in raw_candidates)
        for left_index, left in enumerate(raw_candidates):
            for right in raw_candidates[left_index + 1 :]:
                left_score = float(left["scores"][metric])
                right_score = float(right["scores"][metric])
                margin = abs(left_score - right_score)
                if margin < minimum_margin:
                    continue
                chosen, rejected = (left, right) if left_score > right_score else (right, left)
                if strategy == "top_pairs" and float(chosen["scores"][metric]) < oracle_score - 1.0e-8:
                    continue
                pairs.append(
                    _pair_record(
                        row,
                        candidate_by_id,
                        f"{row['id']}:{chosen['candidate_id']}>{rejected['candidate_id']}",
                        str(chosen["candidate_id"]),
                        str(rejected["candidate_id"]),
                        margin,
                        metric,
                    )
                )
        # The closest score gaps are the hardest negatives. Stable IDs break
        # ties deterministically so experiment reruns use identical examples.
        if strategy in {"hard_negative", "top_pairs"}:
            pairs.sort(key=lambda pair: (pair["margin"], pair["pair_id"]))
    else:
        raise ValueError(f"Unknown ranker pair strategy: {strategy}")

    if max_pairs_per_sample > 0:
        pairs = pairs[:max_pairs_per_sample]
    return pairs


def build(
    source: Path,
    output_dir: Path,
    validation_fraction: float,
    seed: int,
    pair_strategy: str = "parent_child",
    metric: str = "rouge_l",
    minimum_margin: float = 0.02,
    max_pairs_per_sample: int = 0,
) -> dict:
    """从单一 scored 文件随机切分 train/validation（主要用于小型试验）。"""
    rows = read_jsonl(source)
    if len(rows) < 2:
        raise ValueError("Ranker data construction requires at least two samples")
    sample_ids = [str(row["id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Ranker source contains duplicate sample IDs")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("ranker.validation_fraction must be between 0 and 1")

    shuffled = list(sample_ids)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, min(len(rows) - 1, round(len(rows) * validation_fraction)))
    validation_ids = set(shuffled[:validation_count])
    train_ids = set(shuffled[validation_count:])
    if train_ids & validation_ids:
        raise AssertionError("Ranker sample split overlap")

    by_split = {
        "train": [row for row in rows if str(row["id"]) in train_ids],
        "validation": [row for row in rows if str(row["id"]) in validation_ids],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, split_rows in by_split.items():
        groups = [_group_view(row) for row in split_rows]
        listwise_groups = [_listwise_view(row, metric) for row in split_rows]
        pairs = [
            pair
            for row in split_rows
            for pair in _pair_rows(
                row,
                pair_strategy,
                metric,
                minimum_margin,
                max_pairs_per_sample,
            )
        ]
        labels = []
        for row in split_rows:
            label = {"sample_id": str(row["id"]), "target": str(row["target"])}
            if _user_id(row) is not None:
                label["user_id"] = _user_id(row)
            labels.append(label)
        write_jsonl(output_dir / f"{split}_candidates.jsonl", groups)
        write_jsonl(output_dir / f"{split}_listwise.jsonl", listwise_groups)
        write_jsonl(output_dir / f"{split}_pairs.jsonl", pairs)
        # Gold labels are deliberately isolated from model inputs.
        write_jsonl(output_dir / f"{split}_labels.jsonl", labels)
        counts[split] = {
            "samples": len(groups),
            "pairs": len(pairs),
            "listwise_groups": len(listwise_groups),
            "profile_target_matches_filtered": sum(
                int(group["profile_target_matches_filtered"]) for group in groups
            ),
        }

    manifest = {
        "protocol": "grouped_ranker_pilot",
        "seed": seed,
        "source": str(source),
        "validation_fraction": validation_fraction,
        "pair_strategy": pair_strategy,
        "metric": metric,
        "minimum_margin": minimum_margin,
        "max_pairs_per_sample": max_pairs_per_sample,
        "train_ids": sorted(train_ids),
        "validation_ids": sorted(validation_ids),
        "counts": counts,
        "warning": "Pilot split for plumbing only; do not report as a formal held-out benchmark.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"ranker data -> {output_dir}; counts={counts}")
    return manifest


def build_from_splits(
    train_source: Path,
    validation_source: Path,
    output_dir: Path,
    seed: int,
    pair_strategy: str = "all_pairs",
    metric: str = "rouge_l",
    minimum_margin: float = 0.02,
    max_pairs_per_sample: int = 0,
    test_source: Path | None = None,
) -> dict:
    """从明确分离的 train/dev 候选池构建数据，是正式实验推荐入口。

    manifest 会记录两侧 sample id 并检查交集；labels 与模型可见的 candidate
    group 分文件保存，便于审计 target isolation。
    """
    """Build a formal Ranker dataset from disjoint official data splits.

    IDs are namespaced even when the upstream benchmark already uses disjoint
    IDs. This makes the split boundary explicit in every generated artifact and
    prevents a later benchmark version with reused IDs from silently leaking.
    """
    raw_by_split = {
        "train": read_jsonl(train_source),
        "validation": read_jsonl(validation_source),
    }
    if test_source is not None:
        raw_by_split["test"] = read_jsonl(test_source)
    by_split = {}
    for split, rows in raw_by_split.items():
        if not rows:
            raise ValueError(f"Ranker {split} source is empty")
        raw_ids = [str(row["id"]) for row in rows]
        if len(raw_ids) != len(set(raw_ids)):
            raise ValueError(f"Ranker {split} source contains duplicate sample IDs")
        namespace = {"train": "train", "validation": "dev", "test": "test"}[split]
        by_split[split] = [
            {**row, "_ranker_sample_id": f"{namespace}:{row['id']}"} for row in rows
        ]

    ids_by_split = {
        split: {_sample_id(row) for row in rows} for split, rows in by_split.items()
    }
    split_names = list(ids_by_split)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            if ids_by_split[left] & ids_by_split[right]:
                raise AssertionError(
                    f"Ranker official {left}/{right} split overlap"
                )

    has_user_ids = all(
        _user_id(row) is not None for rows in by_split.values() for row in rows
    )
    users_by_split = (
        {
            split: {_user_id(row) for row in rows}
            for split, rows in by_split.items()
        }
        if has_user_ids
        else {}
    )
    if users_by_split:
        for left_index, left in enumerate(split_names):
            for right in split_names[left_index + 1 :]:
                overlap = users_by_split[left] & users_by_split[right]
                if overlap:
                    examples = sorted(overlap)[:5]
                    raise ValueError(
                        f"Ranker {left}/{right} user overlap: count={len(overlap)}, "
                        f"examples={examples}"
                    )

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, split_rows in by_split.items():
        groups = [_group_view(row) for row in split_rows]
        labels = []
        for row in split_rows:
            label = {"sample_id": _sample_id(row), "target": str(row["target"])}
            if _user_id(row) is not None:
                label["user_id"] = _user_id(row)
            labels.append(label)
        write_jsonl(output_dir / f"{split}_candidates.jsonl", groups)
        write_jsonl(output_dir / f"{split}_labels.jsonl", labels)
        # Test labels are isolated solely for final offline metrics. Test never
        # contributes pair/listwise records and therefore cannot affect model
        # training or checkpoint selection.
        if split == "test":
            pairs = []
            listwise_groups = []
        else:
            listwise_groups = [_listwise_view(row, metric) for row in split_rows]
            pairs = [
                pair
                for row in split_rows
                for pair in _pair_rows(
                    row,
                    pair_strategy,
                    metric,
                    minimum_margin,
                    max_pairs_per_sample,
                )
            ]
            write_jsonl(output_dir / f"{split}_listwise.jsonl", listwise_groups)
            write_jsonl(output_dir / f"{split}_pairs.jsonl", pairs)
        counts[split] = {
            "samples": len(groups),
            "pairs": len(pairs),
            "listwise_groups": len(listwise_groups),
            "profile_target_matches_filtered": sum(
                int(group["profile_target_matches_filtered"]) for group in groups
            ),
        }

    manifest = {
        "protocol": "official_split_ranker_experiment",
        "seed": seed,
        "train_source": str(train_source),
        "validation_source": str(validation_source),
        "pair_strategy": pair_strategy,
        "metric": metric,
        "minimum_margin": minimum_margin,
        "max_pairs_per_sample": max_pairs_per_sample,
        "train_ids": sorted(ids_by_split["train"]),
        "validation_ids": sorted(ids_by_split["validation"]),
        "counts": counts,
        "warning": "Dev selects checkpoints; test is evaluated once and never used for training.",
    }
    if users_by_split:
        manifest["user_isolation"] = True
        manifest["user_counts"] = {
            split: len(users) for split, users in users_by_split.items()
        }
    if "test" in ids_by_split:
        manifest["test_source"] = str(test_source)
        manifest["test_ids"] = sorted(ids_by_split["test"])
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"official-split ranker data -> {output_dir}; counts={counts}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build target-isolated pairwise ranker data")
    parser.add_argument("--config", default=GLOBAL_CONFIG)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--train-source", type=Path)
    parser.add_argument("--validation-source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["ranker"]
    data = config["data"]
    processed = resolve_path(data["processed_root"]) / str(
        data.get("processed_split", data["split"])
    )
    train_source = args.train_source or settings.get("train_source")
    validation_source = args.validation_source or settings.get("validation_source")
    test_source = settings.get("test_source")
    common = {
        "output_dir": args.output_dir or resolve_path(settings["data_dir"]),
        "seed": int(config["project"]["seed"]),
        "pair_strategy": str(settings.get("pair_strategy", "parent_child")),
        "metric": str(config["metric"]["primary"]),
        "minimum_margin": float(
            settings.get("pair_minimum_margin", config["metric"]["preference_margin"])
        ),
        "max_pairs_per_sample": int(settings.get("max_pairs_per_sample", 0)),
    }
    if bool(train_source) != bool(validation_source):
        raise ValueError("Set both ranker.train_source and ranker.validation_source")
    if train_source and validation_source:
        build_from_splits(
            resolve_path(train_source),
            resolve_path(validation_source),
            test_source=resolve_path(test_source) if test_source else None,
            **common,
        )
    else:
        build(
            args.source or processed / "06_scored.jsonl",
            validation_fraction=float(settings["validation_fraction"]),
            **common,
        )


if __name__ == "__main__":
    main()
