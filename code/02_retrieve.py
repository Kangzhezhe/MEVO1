"""阶段 02：从目标隔离的完整 Profile 中检索当前 Query 的 Top-k 历史。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline_common import (  # noqa: E402
    load_config,
    load_project_stage,
    read_jsonl,
    stage_path,
    write_jsonl,
)

# 保留旧根入口公开的 BM25 工具 API。
_RETRIEVER = load_project_stage("code/common/bm25_retriever.py", "bm25_retriever")
tokenize = _RETRIEVER.tokenize
rank_profile = _RETRIEVER.rank_profile


def retrieve(config: dict | Path, *args) -> Path | None:
    """执行当前检索阶段，并兼容旧 ``retrieve(source, destination, config)``。"""

    if not isinstance(config, dict):
        if len(args) != 2:
            raise TypeError("legacy retrieve expects source, destination, config")
        destination, legacy_config = args
        _RETRIEVER.retrieve(config, destination, legacy_config)
        return None

    if len(args) != 1:
        raise TypeError("current retrieve expects config and split")
    split = str(args[0])
    settings = config["retrieval"]
    rows = read_jsonl(stage_path(config, split, "prepare"))
    for row in rows:
        row["retrieved_profile"] = _RETRIEVER.rank_profile(
            str(row["source_text"]),
            list(row.get("profile", [])),
            int(settings["top_k"]),
            float(settings["k1"]),
            float(settings["b"]),
        )
    destination = stage_path(config, split, "retrieve")
    write_jsonl(destination, rows)
    print(f"retrieved profiles for {len(rows)} factor-free samples -> {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="02 - 检索用户历史")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "adaptation_train", "adaptation_validation", "adaptation_test"),
        default="train",
    )
    args = parser.parse_args()
    retrieve(load_config(args.config), args.split)


if __name__ == "__main__":
    main()
