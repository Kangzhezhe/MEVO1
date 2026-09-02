"""构造全局 IDPO 的训练用户 Leave-One-Out Query。

每条用户历史记录轮流作为伪 Query/Gold；Gold 只写入 adaptation_train 的
prepare 文件供离线打分，绝不进入 rollout prompt。与 per-user IDPO 不同，所有
用户的 pair 最终合并训练一个共享 Adapter。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline_common import load_config, load_project_stage, read_jsonl, resolve_path, stage_path, write_json, write_jsonl  # noqa: E402

NORMALIZE = re.compile(r"[^a-z0-9]+")


def _key(value: str) -> str:
    return NORMALIZE.sub("", str(value).casefold())


def _unique_profiles(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    seen_abstract: set[str] = set()
    seen_title: set[str] = set()
    for row in rows:
        for index, item in enumerate(row.get("profile", [])):
            abstract = str(item.get("abstract", "")).strip()
            title = str(item.get("title", "")).strip()
            if not abstract or not title:
                continue
            ak, tk = _key(abstract), _key(title)
            if not ak or not tk or ak in seen_abstract or tk in seen_title:
                continue
            seen_abstract.add(ak)
            seen_title.add(tk)
            values.append({"id": str(item.get("id", index)), "abstract": abstract, "title": title})
    return values


def build(config: dict) -> dict[str, Any]:
    source = read_jsonl(stage_path(config, "train", "prepare"))
    by_user: dict[str, list[dict[str, Any]]] = {}
    for row in source:
        by_user.setdefault(str(row.get("user_id", row["id"])), []).append(row)
    limit = int(config.get("global_idpo", {}).get("user_limit", 0))
    if limit > 0:
        by_user = {key: by_user[key] for key in sorted(by_user)[:limit]}
    settings = config.get("global_idpo", {})
    compact_top_k = int(settings.get("compact_profile_top_k", 8))
    minimum = int(settings.get("minimum_profile_count", 10))
    # Smoke/小规模验证可以限制每个用户展开的 LOO Query 数量；默认 0
    # 表示保留完整用户历史，不改变正式实验口径。
    max_adaptation_queries = int(settings.get("max_adaptation_queries", 0))
    retriever = load_project_stage("code/common/bm25_retriever.py", "global_loo_bm25")
    retrieval = config["retrieval"]
    output: list[dict[str, Any]] = []
    profile_counts: dict[str, int] = {}
    query_counts: dict[str, int] = {}
    for user_id in sorted(by_user):
        profiles = _unique_profiles(by_user[user_id])
        if len(profiles) < minimum:
            continue
        profile_counts[user_id] = len(profiles)
        query_counts[user_id] = len(by_user[user_id])
        held_out_profiles = profiles
        if max_adaptation_queries > 0:
            held_out_profiles = profiles[:max_adaptation_queries]
        for held_out in held_out_profiles:
            remaining = [
                item for item in profiles
                if item["id"] != held_out["id"]
                and _key(item["abstract"]) != _key(held_out["abstract"])
                and _key(item["title"]) != _key(held_out["title"])
            ]
            visible = retriever.rank_profile(
                held_out["abstract"], remaining, compact_top_k,
                float(retrieval["k1"]), float(retrieval["b"]),
            )
            output.append({
                "id": f"{user_id}:profile:{held_out['id']}",
                "user_id": user_id,
                "parent_sample_id": user_id,
                "profile_source_id": held_out["id"],
                "source_text": held_out["abstract"],
                "target": held_out["title"],
                "instruction": "Generate a title for the following abstract of a paper",
                "profile": remaining,
                "retrieved_profile": visible,
                "split": "adaptation_train",
                "current_query_used": False,
                "current_gold_used": False,
            })
    destination = stage_path(config, "adaptation_train", "prepare")
    write_jsonl(destination, output)
    report = {
        "protocol": "global_train_user_leave_one_out",
        "users_in_source": len(by_user),
        "users_with_minimum_history": len(profile_counts),
        "source_queries": len(source),
        "adaptation_queries": len(output),
        "profile_counts": profile_counts,
        "query_counts": query_counts,
        "profiles_per_user": "all",
        "compact_profile_top_k": compact_top_k,
        "max_adaptation_queries": max_adaptation_queries,
        "gold_visible_during_rollout": False,
        "target_user_adapter": False,
    }
    write_json(destination.parent / "global_loo_manifest.json", report)
    print(f"Global LOO queries -> {destination}; report={report}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="10 - build global IDPO LOO queries")
    parser.add_argument("--config", default=str(HERE.parent / "config_global.yaml"))
    args = parser.parse_args()
    build(load_config(args.config))
