"""阶段 06：使用 LaMP 官方兼容 ROUGE 计算候选分数并生成偏好对。

gold title 首次在这里用于离线监督。只有超过 ``metric.preference_margin`` 的
父子差异才写入 preferences，避免把近似等价样本当成强监督。
"""

from pathlib import Path

from common.utils import load_config, read_jsonl, write_jsonl
from common.metrics import score


def evaluate(source: Path, destination: Path, config: dict) -> None:
    rows = read_jsonl(source)
    primary = config["metric"]["primary"]
    margin = float(config["metric"]["preference_margin"])
    preference_count = 0
    for row in rows:
        parents = {candidate["candidate_id"]: candidate for candidate in row["candidates"]}
        for parent in parents.values():
            parent["scores"] = score(parent["text"], row["target"])
        preferences = []
        for child in row.get("mutations", []):
            child["scores"] = score(child["text"], row["target"])
            parent = parents[child["parent_id"]]
            delta = child["scores"][primary] - parent["scores"][primary]
            child["delta"] = round(delta, 8)
            if abs(delta) < margin:
                continue
            chosen, rejected = (child, parent) if delta > 0 else (parent, child)
            preferences.append(
                {
                    "id": f"{row['id']}:{child['candidate_id']}",
                    "sample_id": row["id"],
                    "chosen_id": chosen["candidate_id"],
                    "chosen": chosen["text"],
                    "rejected_id": rejected["candidate_id"],
                    "rejected": rejected["text"],
                    "metric": primary,
                    "margin": round(abs(delta), 8),
                }
            )
        row["preferences"] = preferences
        row["metric_metadata"] = {
            "implementation": config["metric"].get("implementation", "lamp_official_rouge"),
            "primary": primary,
            "preference_margin": margin,
        }
        preference_count += len(preferences)
    write_jsonl(destination, rows)
    print(f"scored {len(rows)} samples; kept {preference_count} preferences -> {destination}")


def main() -> None:
    from common.runtime import config_parser, stage_path

    args = config_parser("06 - Score candidate pools with official-compatible ROUGE").parse_args()
    config = load_config(args.config)
    evaluate(stage_path(config, "mutate"), stage_path(config, "score"), config)


if __name__ == "__main__":
    main()
