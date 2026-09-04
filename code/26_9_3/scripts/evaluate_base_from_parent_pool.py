#!/usr/bin/env python3
"""用共享 Pool 的第一个 greedy Parent 生成无 Seed Base 评估报告。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from common.metrics import corpus_bleu, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-users", type=int, default=100)
    parser.add_argument("--expected-queries", type=int, default=608)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    with Path(args.pool).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            parents = item.get("parent_pool", [])
            if len(parents) != 5:
                raise ValueError(f"sample={item.get('id')} Parent 数不是5")
            prediction = str(parents[0].get("text", "")).strip()
            rows.append({"id": str(item.get("id", "")), "user_id": str(item.get("user_id", "")), "source_text": str(item.get("source_text", "")), "target": str(item.get("target", "")), "parent": prediction, "prediction": prediction, "error": None, "seed_parent_used": False})
    users = {row["user_id"] for row in rows if row["user_id"]}
    if len(rows) != args.expected_queries or len(users) != args.expected_users:
        raise ValueError(f"Base 评估口径错误：{len(users)} users/{len(rows)} queries")
    metrics = [score(row["prediction"], row["target"]) for row in rows]
    bleu = corpus_bleu([row["prediction"] for row in rows], [row["target"] for row in rows])
    report = {"protocol": "noseed_base_from_shared_greedy_parent_v1", "users": len(users), "queries": len(rows), "valid_predictions": sum(bool(row["prediction"]) for row in rows), "rouge_1": statistics.mean(item["rouge_1"] for item in metrics), "rouge_l": statistics.mean(item["rouge_l"] for item in metrics), "sacrebleu": float(bleu["score"]), "seed_parent_used": False, "input": "query+top8_history"}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "global_test_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"BASE_FROM_POOL_DONE={report}", flush=True)


if __name__ == "__main__":
    main()
