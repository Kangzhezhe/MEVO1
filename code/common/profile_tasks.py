"""构造 per-user 适配数据：从 profile 留一法生成伪 query/target。"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from common.utils import load_config, read_jsonl, resolve_path, write_jsonl
from common.runtime import USER_CONFIG


NORMALIZE = re.compile(r"[^a-z0-9]+")


def _key(value: str) -> str:
    return NORMALIZE.sub("", str(value).lower())


def _unique_valid_profiles(row: dict) -> list[dict]:
    profiles = []
    seen_abstracts = set()
    seen_titles = set()
    for profile in row.get("profile", []):
        abstract = str(profile.get("abstract", "")).strip()
        title = str(profile.get("title", "")).strip()
        abstract_key = _key(abstract)
        title_key = _key(title)
        if not abstract_key or not title_key:
            continue
        # Avoid constructing a pseudo target that still appears under another
        # duplicate profile record after leave-one-out.
        if abstract_key in seen_abstracts or title_key in seen_titles:
            continue
        seen_abstracts.add(abstract_key)
        seen_titles.add(title_key)
        profiles.append(
            {
                "id": str(profile["id"]),
                "title": title,
                "abstract": abstract,
            }
        )
    return profiles


def build(
    source: Path,
    destination: Path,
    augmented_users: int,
    profiles_per_user: int,
    minimum_profile_count: int,
    seed: int,
) -> dict:
    if augmented_users < 1 or profiles_per_user < 1:
        raise ValueError("augmented_users and profiles_per_user must be positive")
    if minimum_profile_count <= profiles_per_user:
        raise ValueError("minimum_profile_count must exceed profiles_per_user")

    rows = read_jsonl(source)
    eligible = []
    for row in rows:
        profiles = _unique_valid_profiles(row)
        if len(profiles) >= minimum_profile_count:
            eligible.append((row, profiles))
    if len(eligible) < augmented_users:
        raise ValueError(
            f"Requested {augmented_users} augmented users, but only {len(eligible)} are eligible"
        )

    chooser = random.Random(seed)
    selected_indices = set(chooser.sample(range(len(eligible)), augmented_users))
    selected = [eligible[index] for index in range(len(eligible)) if index in selected_indices]
    augmented_rows = []
    selected_user_ids = []
    for row, profiles in selected:
        user_id = str(row["id"])
        selected_user_ids.append(user_id)
        profile_rng = random.Random(f"{seed}:{user_id}:profile-augmentation")
        source_profiles = profile_rng.sample(profiles, profiles_per_user)
        source_profiles.sort(key=lambda profile: profiles.index(profile))
        for source_profile in source_profiles:
            source_id = str(source_profile["id"])
            source_abstract_key = _key(source_profile["abstract"])
            source_title_key = _key(source_profile["title"])
            remaining = [
                profile
                for profile in profiles
                if str(profile["id"]) != source_id
                and _key(profile["abstract"]) != source_abstract_key
                and _key(profile["title"]) != source_title_key
            ]
            if not remaining:
                raise ValueError(f"Leave-one-out removed every profile for user={user_id}")
            augmented_rows.append(
                {
                    "id": f"{user_id}:profile:{source_id}",
                    "parent_sample_id": user_id,
                    "profile_source_id": source_id,
                    "sample_origin": "profile_leave_one_out",
                    "instruction": "Generate a title for the following abstract of a paper",
                    "query": source_profile["abstract"],
                    "source_text": source_profile["abstract"],
                    "target": source_profile["title"],
                    "profile": remaining,
                    "split": "train_profile_augmented",
                }
            )

    expected = augmented_users * profiles_per_user
    if len(augmented_rows) != expected:
        raise AssertionError(f"Expected {expected} augmented rows, got {len(augmented_rows)}")
    sample_ids = [row["id"] for row in augmented_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Profile augmentation produced duplicate sample IDs")

    write_jsonl(destination, augmented_rows)
    manifest = {
        "protocol": "profile_as_query_leave_one_out",
        "source": str(source),
        "seed": seed,
        "augmented_users": augmented_users,
        "profiles_per_user": profiles_per_user,
        "minimum_profile_count": minimum_profile_count,
        "selected_user_ids": selected_user_ids,
        "augmented_samples": len(augmented_rows),
        "source_profile_excluded": True,
        "exact_title_abstract_deduplication": True,
    }
    manifest_path = destination.parent / "profile_augmentation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"profile augmentation -> {destination}; manifest={manifest_path}; samples={len(augmented_rows)}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leave-one-out LaMP-5 profile pseudo queries")
    parser.add_argument("--config", default=USER_CONFIG)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["profile_augmentation"]
    data = config["data"]
    root = resolve_path(data["processed_root"])
    split = str(data.get("processed_split", data["split"]))
    build(
        args.source or resolve_path(settings["source"]),
        args.destination or root / split / "01_prepared.jsonl",
        int(settings["augmented_users"]),
        int(settings["profiles_per_user"]),
        int(settings["minimum_profile_count"]),
        int(config["project"]["seed"]),
    )


if __name__ == "__main__":
    main()
