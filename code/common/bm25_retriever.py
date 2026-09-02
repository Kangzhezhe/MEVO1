"""阶段 02：用轻量 BM25 风格检索得到与当前 query 最相关的用户历史。

检索结果是后续因子构建的唯一 profile 证据来源；调参优先修改配置中的
``retrieval.top_k``，若替换检索器则保持 ``retrieved_profile`` 字段格式不变。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from common.utils import load_config, read_jsonl, write_jsonl
TOKEN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def rank_profile(query: str, profile: list[dict[str, Any]], top_k: int, k1: float, b: float) -> list[dict[str, Any]]:
    documents = [tokenize(f"{item.get('title', '')} {item.get('abstract', '')}") for item in profile]
    if not documents:
        return []
    frequencies = [Counter(document) for document in documents]
    average_length = sum(map(len, documents)) / len(documents) or 1.0
    document_frequency = Counter(token for document in documents for token in set(document))
    query_terms = tokenize(query)
    scores = []
    for index, (document, frequency) in enumerate(zip(documents, frequencies)):
        score = 0.0
        for term in query_terms:
            if not frequency[term]:
                continue
            df = document_frequency[term]
            idf = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
            denominator = frequency[term] + k1 * (1 - b + b * len(document) / average_length)
            score += idf * frequency[term] * (k1 + 1) / denominator
        scores.append((score, index))
    scores.sort(key=lambda pair: (-pair[0], pair[1]))
    result = []
    for rank, (score, index) in enumerate(scores[:top_k], 1):
        result.append({"rank": rank, "score": round(score, 6), **profile[index]})
    return result


def retrieve(source: Path, destination: Path, config: dict) -> None:
    rows = read_jsonl(source)
    retrieval = config["retrieval"]
    for row in rows:
        row["retrieved_profile"] = rank_profile(
            row["source_text"], row["profile"], retrieval["top_k"], retrieval["k1"], retrieval["b"]
        )
    write_jsonl(destination, rows)
    print(f"retrieved profiles for {len(rows)} samples -> {destination}")


def main() -> None:
    from common.runtime import config_parser, stage_path

    args = config_parser("02 - Retrieve relevant profile items").parse_args()
    config = load_config(args.config)
    retrieve(stage_path(config, "prepare"), stage_path(config, "retrieve"), config)


if __name__ == "__main__":
    main()
