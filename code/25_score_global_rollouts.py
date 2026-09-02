"""用训练用户 LOO Gold 离线给全局 rollout 打偏好标签。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from idpo_common import idpo_path  # noqa: E402
from pipeline_common import load_config, read_jsonl, score, stage_path, write_jsonl, write_json  # noqa: E402


def run(config: dict[str, Any]) -> dict[str, Any]:
    settings = config["global_idpo"]
    rollouts = read_jsonl(idpo_path(config, 0, "train_rollouts.jsonl"))
    targets = {str(row["id"]): str(row["target"]) for row in read_jsonl(stage_path(config, "adaptation_train", "prepare"))}
    output = []
    accepted = 0
    for rollout in rollouts:
        target = targets.get(str(rollout["pseudo_query_id"]))
        if target is None:
            raise KeyError(f"缺少 LOO Gold: {rollout['pseudo_query_id']}")
        eligible = [item for item in rollout.get("responses", []) if bool(item.get("dpo_eligible", True))]
        scored = []
        for item in eligible:
            metrics = score(str(item["output"]), target)
            scored.append({"response_id": str(item["response_id"]), "reward": float(metrics[settings.get("reward_metric", "rouge_l")]), "metrics": metrics})
        scored.sort(key=lambda item: (item["reward"], item["response_id"]))
        chosen = scored[-1] if scored else None
        rejected = scored[0] if scored else None
        margin = float(chosen["reward"] - rejected["reward"]) if chosen and rejected else 0.0
        status = "accepted" if chosen and rejected and chosen["response_id"] != rejected["response_id"] and margin >= float(settings.get("minimum_reward_margin", 0.03)) else "low_margin"
        preference = None
        if status == "accepted":
            accepted += 1
            preference = {"source": "global_train_loo_gold", "chosen_id": chosen["response_id"], "rejected_id": rejected["response_id"], "chosen_reward": chosen["reward"], "rejected_reward": rejected["reward"], "chosen_metrics": chosen["metrics"], "rejected_metrics": rejected["metrics"], "margin": margin}
        output.append({**rollout, "preference_status": status, "preference": preference, "response_rewards": scored, "hidden_target_used": True})
    destination = idpo_path(config, 0, "train_preferences.jsonl")
    write_jsonl(destination, output)
    report = {"rollouts": len(output), "accepted_pairs": accepted, "low_margin": len(output) - accepted, "reward_metric": settings.get("reward_metric", "rouge_l"), "minimum_reward_margin": float(settings.get("minimum_reward_margin", 0.03)), "gold_visible_during_rollout": False}
    write_json(idpo_path(config, 0, "train_preference_report.json"), report)
    print(f"Global rollout preferences -> {destination}; report={report}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="25 - score global rollouts")
    parser.add_argument("--config", default=str(HERE.parent / "config_global.yaml"))
    args = parser.parse_args()
    run(load_config(args.config))
