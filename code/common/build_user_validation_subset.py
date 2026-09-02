"""Build a fixed validation-candidate subset for selected profile users."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.utils import load_config, read_jsonl, resolve_path, write_jsonl


FILES = (
    "validation_candidates.jsonl",
    "validation_labels.jsonl",
    "validation_listwise.jsonl",
    "validation_pairs.jsonl",
)


def _user_id(sample_id: str) -> str:
    return str(sample_id).split(":", 1)[-1]


def build(config: dict) -> dict:
    settings = config["user_validation_subset"]
    source_dir = resolve_path(settings["source_dir"])
    destination = resolve_path(settings["output_dir"])
    profile_manifest = json.loads(
        resolve_path(settings["profile_manifest"]).read_text(encoding="utf-8")
    )
    selected_users = set(map(str, profile_manifest["selected_user_ids"]))
    if not selected_users:
        raise ValueError("Profile manifest selected no users")
    destination.mkdir(parents=True, exist_ok=True)
    counts = {}
    validation_ids = None
    for filename in FILES:
        rows = [
            row
            for row in read_jsonl(source_dir / filename)
            if _user_id(row["sample_id"]) in selected_users
        ]
        write_jsonl(destination / filename, rows)
        counts[filename] = len(rows)
        if filename == "validation_candidates.jsonl":
            validation_ids = [str(row["sample_id"]) for row in rows]
    found_users = {_user_id(sample_id) for sample_id in validation_ids or []}
    if found_users != selected_users:
        raise ValueError(
            f"Validation subset users do not match profile users: "
            f"missing={sorted(selected_users - found_users)}, "
            f"extra={sorted(found_users - selected_users)}"
        )
    source_manifest = json.loads(
        (source_dir / "manifest.json").read_text(encoding="utf-8")
    )
    manifest = {
        **source_manifest,
        "protocol": "official_split_per_user_subset",
        "subset_source": str(source_dir),
        "profile_manifest": str(resolve_path(settings["profile_manifest"])),
        "train_ids": [],
        "validation_ids": validation_ids,
        "selected_user_ids": sorted(selected_users),
        "counts": {
            "train": {"samples": 0, "pairs": 0, "listwise_groups": 0},
            "validation": {
                "samples": len(validation_ids or []),
                "pairs": counts["validation_pairs.jsonl"],
                "listwise_groups": counts["validation_listwise.jsonl"],
            },
        },
        "warning": "Per-user development subset; official test split remains untouched.",
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"per-user validation subset -> {destination}; "
        f"users={len(selected_users)}, counts={counts}"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build selected-user validation subset")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
