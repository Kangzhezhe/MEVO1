#!/usr/bin/env python3
"""独立构建无 Seed 的共享 Parent Pool。

每个 Query 只调用一次冻结 Base Llama2-7B：

    Query + Top-8 History -> 1 greedy + 4 sampling Parents

输出固定为五个槽位；重复 Parent 保留，空/明显格式污染的槽位复制第一个
有效 Parent 补齐。该 Pool 可被 Direct-Parent Editor、Multitask Editor 和
Gold Crossover 共同复用，避免各实验重新采样导致输入不一致。

No-history 消融使用：

    Query -> 1 greedy + 4 sampling Parents

通过 ``--ablation no_history`` 生成独立 Pool，不能与带 History 的 Pool 混用。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CODE = HERE.parent
PROJECT = CODE.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(PROJECT))

from pipeline_common import load_config, read_jsonl, resolve_path, write_json  # noqa: E402
from run_gold_multi_parent_crossover import (  # noqa: E402
    build_parent_pool,
    configure,
    load_source_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建无 Seed 共享 Parent Pool")
    parser.add_argument("--config", default=str(PROJECT / "config_global_llama2_7b_visgpt_prime_matched.yaml"))
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--ablation", choices=("main", "no_history"), default="main")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-end", type=int, default=0)
    parser.add_argument("--test-start", type=int, default=0)
    parser.add_argument("--test-end", type=int, default=0)
    parser.add_argument(
        "--shard-name",
        default="",
        help="分片输出目录名；设置后输出 train_parent_pool.jsonl/test_parent_pool.jsonl",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = configure(load_config(args.config))
    stamp = args.run_name or (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_shared_parent_pool_{args.ablation}"
    )
    data_dir = resolve_path(
        args.data_dir or f"/data/liux/MEVO_global_cot/dataset/parent_pools/{stamp}"
    )
    if args.shard_name:
        data_dir = data_dir / args.shard_name
    data_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "protocol": f"shared_base_parent_pool_fixed5_{args.ablation}_v1",
        "ablation": args.ablation,
        "requested_parent_count": 5,
        "teacher_used": False,
        "seed_parent_used": False,
        "source_stage": "02_retrieved",
        "history_used": args.ablation != "no_history",
    }

    for split, limit, filename in (
        ("train", args.limit, "train_parent_pool.jsonl"),
        ("test", args.test_limit, "test_parent_pool.jsonl"),
    ):
        if args.split not in {split, "all"}:
            continue
        rows = load_source_rows(config, "", split)
        start = args.train_start if split == "train" else args.test_start
        end = args.train_end if split == "train" else args.test_end
        if end > 0:
            if start < 0 or start >= end:
                raise ValueError(f"无效 {split} 分片区间：{start}:{end}")
            rows = rows[start:end]
        pool_rows = build_parent_pool(
            rows,
            config,
            data_dir / filename,
            pool_source="base_model",
            limit=limit,
            ablation=args.ablation,
        )
        counts: dict[str, int] = {}
        unique_counts: dict[str, int] = {}
        for row in pool_rows:
            count = str(len(row.get("parent_pool", [])))
            unique = str(row.get("unique_parent_count", 0))
            counts[count] = counts.get(count, 0) + 1
            unique_counts[unique] = unique_counts.get(unique, 0) + 1
        report[split] = {
            "queries": len(pool_rows),
            "parent_count_distribution": counts,
            "unique_parent_count_distribution": unique_counts,
            "path": str(data_dir / filename),
        }

    write_json(data_dir / "manifest.json", report)
    print(f"SHARED_PARENT_POOL_DONE={report}", flush=True)


if __name__ == "__main__":
    main()
