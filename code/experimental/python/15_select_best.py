"""按 Dev 共享 Ranker 指标选择全量实验配置。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def key(item: dict) -> tuple[float, float, float]:
    validation = (item.get("validation") or {}).get("scorers", {}).get("shared", {})
    ranking = validation.get("ranking", {})
    return (
        float(validation.get("rouge_l", -1.0)),
        float(ranking.get("hit_at", {}).get("1", -1.0)),
        -float(ranking.get("mean_regret", 1.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    best, values = max(report["methods"].items(), key=lambda pair: key(pair[1]))
    Path(args.output).write_text(
        json.dumps({"best_method": best, "config": values["config"], "key": key(values)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(best)
    print(f"BEST_METHOD={best} KEY={key(values)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
