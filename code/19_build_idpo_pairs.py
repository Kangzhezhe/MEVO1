"""阶段 19：把 Gold scorer 或 Teacher Judge 结果转换为 per-user DPO pair。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline_common import load_config, read_jsonl, write_json  # noqa: E402
from idpo_common import group_pairs_by_user, idpo_path  # noqa: E402


def build(config: dict, split: str = "validation") -> dict[str, Any]:
    settings = config["idpo"]
    round_index = int(settings["round"])
    source_name = (
        f"{split}_preferences.jsonl"
        if str(settings.get("preference_source", "teacher_judge")) == "loo_gold"
        else f"{split}_judged.jsonl"
    )
    source = read_jsonl(idpo_path(config, round_index, source_name))
    pairs = []
    seen = set()
    for rollout in source:
        preference = rollout.get("preference")
        if preference is None and rollout.get("judge_status") == "accepted":
            preference = rollout.get("judge")
        status = rollout.get("preference_status", rollout.get("judge_status"))
        if status != "accepted" or not preference:
            continue
        by_id = {str(item["response_id"]): item for item in rollout["responses"]}
        chosen = by_id.get(str(preference["chosen_id"]))
        rejected = by_id.get(str(preference["rejected_id"]))
        if chosen is None or rejected is None or chosen is rejected:
            continue
        pair_id = f"{rollout['rollout_id']}:{chosen['response_id']}>{rejected['response_id']}"
        if pair_id in seen:
            continue
        seen.add(pair_id)
        pairs.append(
            {
                "pair_id": pair_id,
                "user_id": str(rollout["user_id"]),
                "pseudo_query_id": str(rollout["pseudo_query_id"]),
                "operation_type": str(rollout["operation_type"]),
                "prompt": str(rollout["prompt"]),
                "chosen": str(chosen["response_text"]),
                "rejected": str(rejected["response_text"]),
                "chosen_trace_text": str(chosen.get("trace_text", "")),
                "rejected_trace_text": str(rejected.get("trace_text", "")),
                "chosen_output_text": str(chosen.get("output_text", chosen["response_text"])),
                "rejected_output_text": str(rejected.get("output_text", rejected["response_text"])),
                "chosen_output": str(chosen["output"]),
                "rejected_output": str(rejected["output"]),
                "preference_source": str(
                    preference.get("source", settings.get("preference_source", "teacher_judge"))
                ),
                "preference_margin": float(preference.get("margin", 0.0)),
                "chosen_reward": preference.get("chosen_reward"),
                "rejected_reward": preference.get("rejected_reward"),
                "chosen_metrics": preference.get("chosen_metrics"),
                "rejected_metrics": preference.get("rejected_metrics"),
                # 保留旧字段，兼容已有的 Teacher-Judge 分析脚本。
                "judge_confidence": float(preference.get("confidence", 0.0)),
                "judge_evidence_ids": list(preference.get("evidence_ids", [])),
                "judge_reason": str(preference.get("reason", "")),
            }
        )
    if not pairs:
        raise ValueError("没有可用于 IDPO 的 accepted pair")
    grouped = group_pairs_by_user(pairs)
    destination = idpo_path(config, round_index, f"{split}_pairs.jsonl")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in pairs) + "\n",
        encoding="utf-8",
    )
    report = {
        "round": round_index,
        "split": split,
        "pairs": len(pairs),
        "users": len(grouped),
        "pairs_per_user": {user: len(values) for user, values in sorted(grouped.items())},
        "prompt_shared_between_chosen_rejected": True,
        "preference_source": str(settings.get("preference_source", "teacher_judge")),
        "hidden_target_used_for_label_only": str(settings.get("preference_source")) == "loo_gold",
        "gold_visible_during_rollout": False,
    }
    write_json(idpo_path(config, round_index, f"{split}_pair_report.json"), report)
    print(f"IDPO pairs -> {destination}; report={report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="19 - Build IDPO DPO pairs")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    build(load_config(args.config), args.split)


if __name__ == "__main__":
    main()
