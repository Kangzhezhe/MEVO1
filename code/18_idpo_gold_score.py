"""阶段 18：用历史 Leave-One-Out Gold 为 on-policy rollout 构造偏好。

Gold 只在本阶段从 adaptation prepare 文件读取，用于计算本地 ROUGE；它从不
进入 Editor Prompt、rollout 文件或最终 DPO Prompt。每个 rollout 只选择最高分
与最低分响应，并过滤奖励差距过小的噪声 Pair。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from idpo_common import idpo_path  # noqa: E402
from pipeline_common import (  # noqa: E402
    load_config,
    read_jsonl,
    score,
    stage_path,
    write_json,
    write_jsonl,
)


def _score_one(
    rollout: dict[str, Any], target: str, settings: dict[str, Any]
) -> dict[str, Any]:
    if not bool(rollout.get("minimum_responses_met", False)):
        return {
            **rollout,
            "preference_status": "insufficient_responses",
            "preference": None,
        }

    metric_name = str(settings.get("gold_reward_metric", "rouge_l"))
    if metric_name not in {"rouge_1", "rouge_l"}:
        raise ValueError("idpo.gold_reward_metric 必须是 rouge_1 或 rouge_l")
    scored = []
    for response in rollout["responses"]:
        metrics = score(str(response["output"]), target)
        scored.append(
            {
                "response_id": str(response["response_id"]),
                "reward": float(metrics[metric_name]),
                "metrics": {key: float(value) for key, value in metrics.items()},
            }
        )
    scored.sort(key=lambda item: (item["reward"], item["response_id"]))
    rejected, chosen = scored[0], scored[-1]
    margin = float(chosen["reward"] - rejected["reward"])
    minimum_margin = float(settings.get("minimum_reward_margin", 0.03))
    if chosen["response_id"] == rejected["response_id"] or margin < minimum_margin:
        return {
            **rollout,
            "preference_status": "low_margin",
            "preference": None,
            "response_rewards": scored,
            "reward_margin": margin,
            "hidden_target_used": True,
        }
    preference = {
        "source": f"loo_gold_{metric_name}",
        "chosen_id": chosen["response_id"],
        "rejected_id": rejected["response_id"],
        "chosen_reward": chosen["reward"],
        "rejected_reward": rejected["reward"],
        "chosen_metrics": chosen["metrics"],
        "rejected_metrics": rejected["metrics"],
        "margin": margin,
        "reason": f"Leave-One-Out Gold {metric_name} margin={margin:.6f}",
    }
    return {
        **rollout,
        "preference_status": "accepted",
        "preference": preference,
        "response_rewards": scored,
        "reward_margin": margin,
        "hidden_target_used": True,
    }


def run(config: dict, split: str = "validation") -> dict[str, Any]:
    settings = config["idpo"]
    round_index = int(settings["round"])
    rollouts = read_jsonl(idpo_path(config, round_index, f"{split}_rollouts.jsonl"))
    adaptation_rows = read_jsonl(stage_path(config, f"adaptation_{split}", "prepare"))
    targets = {str(row["id"]): str(row["target"]) for row in adaptation_rows}
    rows = []
    for index, rollout in enumerate(rollouts, 1):
        pseudo_query_id = str(rollout["pseudo_query_id"])
        if pseudo_query_id not in targets:
            raise KeyError(f"rollout={rollout['rollout_id']} 找不到 LOO Gold")
        result = _score_one(rollout, targets[pseudo_query_id], settings)
        rows.append(result)
        if index % int(settings.get("checkpoint_every", 10)) == 0:
            print(
                f"IDPO Gold score {index}/{len(rollouts)} "
                f"status={result['preference_status']}",
                flush=True,
            )
    destination = idpo_path(config, round_index, f"{split}_preferences.jsonl")
    write_jsonl(destination, rows)
    report = {
        "round": round_index,
        "split": split,
        "rollouts": len(rows),
        "accepted_pairs": sum(row["preference_status"] == "accepted" for row in rows),
        "low_margin": sum(row["preference_status"] == "low_margin" for row in rows),
        "insufficient_responses": sum(
            row["preference_status"] == "insufficient_responses" for row in rows
        ),
        "preference_source": "loo_gold",
        "reward_metric": str(settings.get("gold_reward_metric", "rouge_l")),
        "minimum_reward_margin": float(settings.get("minimum_reward_margin", 0.03)),
        "gold_visible_during_rollout": False,
        "hidden_target_used_for_label_only": True,
    }
    write_json(idpo_path(config, round_index, f"{split}_preference_report.json"), report)
    print(f"IDPO Gold preferences -> {destination}; report={report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="18 - Score IDPO rollouts with LOO Gold")
    parser.add_argument(
        "--config", default=str(HERE / "config_simple_trace_top8_idpo_first50.yaml")
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    run(load_config(args.config), args.split)


if __name__ == "__main__":
    main()
