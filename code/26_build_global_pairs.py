"""将全局 rollout 的离线偏好转换为共享 DPO 训练对。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from idpo_common import idpo_path  # noqa: E402
from pipeline_common import load_config, read_jsonl, write_json, write_jsonl  # noqa: E402


def build(config: dict[str, Any]) -> dict[str, Any]:
    rows = read_jsonl(idpo_path(config, 0, "train_preferences.jsonl"))
    pairs = []
    for row in rows:
        preference = row.get("preference")
        if row.get("preference_status") != "accepted" or not preference:
            continue
        by_id = {str(item["response_id"]): item for item in row.get("responses", []) if bool(item.get("dpo_eligible", True))}
        chosen = by_id.get(str(preference["chosen_id"]))
        rejected = by_id.get(str(preference["rejected_id"]))
        if chosen is None or rejected is None or str(row.get("prompt", "")) == "":
            continue
        # 同一个 rollout 的 mutation response 共享 prompt；Teacher crossover 默认
        # dpo_eligible=false，若显式打开则仍会被记录，但报告会单独标注。
        pairs.append({
            "pair_id": f"{row['rollout_id']}:{chosen['response_id']}>{rejected['response_id']}",
            "user_id": str(row.get("user_id", "")),
            "pseudo_query_id": str(row["pseudo_query_id"]),
            "operation_type": "evolution",
            "prompt": str(row["prompt"]),
            "chosen": str(chosen["response_text"]),
            "rejected": str(rejected["response_text"]),
            "chosen_trace_text": str(chosen.get("trace_text", "")),
            "rejected_trace_text": str(rejected.get("trace_text", "")),
            "chosen_output_text": str(chosen.get("output_text", "")),
            "rejected_output_text": str(rejected.get("output_text", "")),
            "chosen_output": str(chosen["output"]),
            "rejected_output": str(rejected["output"]),
            "chosen_reward": preference.get("chosen_reward"),
            "rejected_reward": preference.get("rejected_reward"),
            "preference_margin": preference.get("margin"),
            "preference_source": "global_train_loo_gold",
            "chosen_source": chosen.get("source"),
            "rejected_source": rejected.get("source"),
        })
    if not pairs:
        raise ValueError("没有可用于全局 IDPO 的 accepted pair")
    destination = idpo_path(config, 0, "train_pairs.jsonl")
    write_jsonl(destination, pairs)
    report = {
        "pairs": len(pairs),
        "users": len({item["user_id"] for item in pairs}),
        "pairs_per_user": {user: sum(item["user_id"] == user for item in pairs) for user in sorted({item["user_id"] for item in pairs})},
        "global_shared_adapter": True,
        "teacher_crossover_pairs": sum(item.get("chosen_source") == "teacher_direct_sequence_fusion" or item.get("rejected_source") == "teacher_direct_sequence_fusion" for item in pairs),
    }
    write_json(idpo_path(config, 0, "train_pair_report.json"), report)
    print(f"Global IDPO pairs -> {destination}; report={report}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="26 - build global DPO pairs")
    parser.add_argument("--config", default=str(HERE.parent / "config_global.yaml"))
    args = parser.parse_args()
    build(load_config(args.config))
