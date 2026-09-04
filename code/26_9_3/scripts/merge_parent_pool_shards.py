#!/usr/bin/env python3
"""合并无 Seed 共享 Parent Pool 分片并验证固定宽度契约。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    args = parser.parse_args()
    root, output_dir = Path(args.shard_root), Path(args.output_dir)
    rows: list[dict[str, Any]] = []
    for index in range(args.expected_shards):
        path = root / f"shard_{index}" / f"{args.split}_parent_pool.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"缺少 Parent Pool 分片：{path}")
        rows.extend(read(path))
    rows.sort(key=lambda item: str(item.get("id", "")))
    ids = [str(item.get("id", "")) for item in rows]
    if len(rows) != args.expected:
        raise ValueError(f"{args.split} 数量错误：{len(rows)} != {args.expected}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{args.split} 存在重复 Query ID")
    bad = [str(item.get("id", "")) for item in rows if len(item.get("parent_pool", [])) != 5]
    if bad:
        raise ValueError(f"{args.split} 存在非固定5槽位 Parent：{bad[:5]}")
    if any(bool(item.get("seed_parent_used", False)) for item in rows):
        raise ValueError(f"{args.split} Parent Pool 标记为 seed_parent_used")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{args.split}_parent_pool.jsonl"
    with destination.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    unique_mean = sum(item.get("unique_parent_count", 0) for item in rows) / max(len(rows), 1)
    print(f"MERGED_PARENT_POOL split={args.split} rows={len(rows)} unique_mean={unique_mean:.3f} output={destination}", flush=True)


if __name__ == "__main__":
    main()
