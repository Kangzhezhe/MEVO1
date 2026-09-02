"""阶段 10：从每个评估用户的历史构造 Leave-One-Out 适配 Query。

每条历史论文依次充当伪 Query/Gold，且从该伪 Query 的可见 Profile 中移除。
当前测试 Query 及其 Gold 均不参与用户 Head 训练。
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline_common import (  # noqa: E402
    load_config,
    load_project_stage,
    read_jsonl,
    stage_path,
    write_json,
    write_jsonl,
)


NORMALIZE = re.compile(r"[^a-z0-9]+")


def _key(value: str) -> str:
    return NORMALIZE.sub("", str(value).casefold())


def _unique_profiles(row: dict[str, Any]) -> list[dict[str, str]]:
    result, abstracts, titles = [], set(), set()
    for index, item in enumerate(row.get("profile", [])):
        abstract = str(item.get("abstract", "")).strip()
        title = str(item.get("title", "")).strip()
        abstract_key, title_key = _key(abstract), _key(title)
        if not abstract_key or not title_key:
            continue
        if abstract_key in abstracts or title_key in titles:
            continue
        abstracts.add(abstract_key)
        titles.add(title_key)
        result.append(
            {"id": str(item.get("id", index)), "abstract": abstract, "title": title}
        )
    return result


def build(config: dict, source_split: str) -> dict[str, Any]:
    if source_split not in {"validation", "test"}:
        raise ValueError("source_split 必须是 validation 或 test")
    adaptation_split = f"adaptation_{source_split}"
    source_rows = read_jsonl(stage_path(config, source_split, "prepare"))
    settings = config["profile_augmentation"]
    requested_count = settings["profiles_per_user"]
    use_all_profiles = str(requested_count).strip().lower() == "all"
    count = 0 if use_all_profiles else int(requested_count)
    minimum = int(settings["minimum_profile_count"])
    compact_top_k = int(settings.get("compact_profile_top_k", 0))
    ranker = None
    if compact_top_k > 0:
        ranker = load_project_stage(
            "code/common/bm25_retriever.py", "loo_compact_profile_retriever"
        )
    # Per-Pcs 会为一个用户提供多条当前 Query。这里先按真实 user_id 合并，确保
    # 一个用户只抽取一次历史、只训练一个 Head，而不是每条 Query 各训练一个。
    current_rows_by_user: dict[str, list[dict[str, Any]]] = {}
    for current in source_rows:
        user_id = str(current.get("user_id", current["id"]))
        current_rows_by_user.setdefault(user_id, []).append(current)

    # IDPO pilot 可以先固定一小组用户，避免为全量 validation 预先构造大量
    # Leave-One-Out Query；不设置时保持原有全量 per-user Head 协议。
    idpo_user_limit = int(config.get("idpo", {}).get("user_limit", 0))
    if idpo_user_limit > 0:
        selected_users = sorted(current_rows_by_user)[:idpo_user_limit]
        current_rows_by_user = {
            user_id: current_rows_by_user[user_id] for user_id in selected_users
        }

    rows = []
    query_counts_by_user = {}
    profile_counts_by_user = {}
    for user_id, current_rows in current_rows_by_user.items():
        query_counts_by_user[user_id] = len(current_rows)
        profiles = _unique_profiles(
            {
                "profile": [
                    profile
                    for current in current_rows
                    for profile in current.get("profile", [])
                ]
            }
        )
        if len(profiles) < minimum:
            raise ValueError(
                f"user={user_id} 只有 {len(profiles)} 条有效历史，少于 {minimum}"
            )
        profile_counts_by_user[user_id] = len(profiles)
        if use_all_profiles:
            selected = list(profiles)
        else:
            rng = random.Random(
                f"{config['project']['seed']}:{user_id}:factor-free-loo"
            )
            selected = rng.sample(profiles, count)
            selected.sort(key=profiles.index)
        for history in selected:
            remaining = [
                item for item in profiles
                if item["id"] != history["id"]
                and _key(item["abstract"]) != _key(history["abstract"])
                and _key(item["title"]) != _key(history["title"])
            ]
            # 全历史 LOO 若在每行重复保存其余完整 Profile，会产生数 GB 的高度
            # 冗余 JSON。这里先让全部剩余历史参与 BM25，再只保存实际会进入
            # Editor 的 Top-k；后续 Stage 02 对这 k 条重排不改变候选集合。
            if ranker is not None:
                retrieval = config["retrieval"]
                remaining = ranker.rank_profile(
                    history["abstract"],
                    remaining,
                    compact_top_k,
                    float(retrieval["k1"]),
                    float(retrieval["b"]),
                )
            rows.append(
                {
                    "id": f"{user_id}:profile:{history['id']}",
                    # common.user_head 继续使用 parent_sample_id 分组，但其值现在
                    # 是真实 user_id；同一用户的全部当前 Query 共享这个 Head。
                    "parent_sample_id": user_id,
                    "user_id": user_id,
                    "current_query_ids": [str(item["id"]) for item in current_rows],
                    "profile_source_id": history["id"],
                    "full_profile_count": len(profiles),
                    "profile_compacted_after_full_retrieval": compact_top_k > 0,
                    "sample_origin": "profile_leave_one_out",
                    "instruction": "Generate a title for the following abstract of a paper",
                    "query": history["abstract"],
                    "source_text": history["abstract"],
                    "target": history["title"],
                    "profile": remaining,
                    "split": adaptation_split,
                }
            )
    expected = (
        sum(profile_counts_by_user.values())
        if use_all_profiles
        else len(current_rows_by_user) * count
    )
    if len(rows) != expected:
        raise AssertionError(f"适配 Query 应为 {expected}，实际为 {len(rows)}")
    destination = stage_path(config, adaptation_split, "prepare")
    write_jsonl(destination, rows)
    report = {
        "protocol": "perpcs_factor_free_profile_leave_one_out",
        "source_split": source_split,
        "users": len(current_rows_by_user),
        "current_queries": len(source_rows),
        "queries_per_user": query_counts_by_user,
        "profiles_per_user": "all" if use_all_profiles else count,
        "profile_counts_by_user": profile_counts_by_user,
        "compact_profile_top_k": compact_top_k,
        "all_remaining_history_considered_by_retrieval": compact_top_k > 0,
        "adaptation_queries": len(rows),
        "current_query_used": False,
        "current_gold_used": False,
        "explicit_user_factors": False,
        "head_key": "user_id",
    }
    write_json(destination.parent / "adaptation_manifest.json", report)
    print(f"Leave-One-Out adaptation queries -> {destination}; report={report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="10 - 构建 per-user Leave-One-Out 数据")
    parser.add_argument("--config", default=str(HERE / "config_idpo.yaml"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    build(load_config(args.config), args.split)


if __name__ == "__main__":
    main()
