"""阶段 01：把 Per-Pcs 用户分区规范化为一条 Query 一行的 JSONL。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline_common import (  # noqa: E402
    load_config,
    load_project_stage,
    resolve_path,
    stage_path,
)

# 保留旧根入口公开的数据读取 API，避免既有测试和外部脚本因正式流程提升而失效。
_READER = load_project_stage("code/common/perpcs_reader.py", "perpcs_reader")
PERPCS_PARTITIONS = _READER.PERPCS_PARTITIONS
normalize_lamp5 = _READER.normalize_lamp5
normalize_perpcs_lamp5 = _READER.normalize_perpcs_lamp5
prepare_perpcs = _READER.prepare_perpcs


def prepare(config: dict | Path, *args) -> Path | None:
    """执行当前 Per-Pcs 准备阶段，并兼容旧官方 LaMP 调用签名。

    当前签名是 ``prepare(config, split)``；旧测试仍使用
    ``prepare(questions, outputs, destination, split, limit, seed)``。
    """

    if not isinstance(config, dict):
        if len(args) != 5:
            raise TypeError(
                "legacy prepare expects questions, outputs, destination, split, limit, seed"
            )
        outputs, destination, split, limit, seed = args
        _READER.prepare(config, outputs, destination, split, limit, seed)
        return None

    if len(args) != 1:
        raise TypeError("current prepare expects config and split")
    split = str(args[0])
    spec = config["splits"][split]
    partition = str(spec["partition"])
    relative = _READER.PERPCS_PARTITIONS[partition]
    source = resolve_path(config["data"]["perpcs_root"]) / relative
    destination = stage_path(config, split, "prepare")
    _READER.prepare_perpcs(
        source=source,
        destination=destination,
        split=split,
        partition=partition,
        limit=int(spec.get("limit", 0)),
        seed=int(config["project"]["seed"]),
        drop_missing_profile_abstracts=bool(
            config["data"].get("drop_missing_profile_abstracts", True)
        ),
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="01 - 准备 Per-Pcs 数据")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    args = parser.parse_args()
    prepare(load_config(args.config), args.split)


if __name__ == "__main__":
    main()
