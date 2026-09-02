"""从已有 Global rollout 构造严格的 Mutation-only、output-only DPO pairs。

只保留学生 Mutation 响应，并要求 chosen 同时满足：
1. chosen 与 rejected 的 ROUGE-L 差距达到阈值；
2. chosen 比原始 Parent 至少提升指定阈值。

这样可以避免 DPO 学习两个都劣于 Parent 的坏候选，也不需要重新调用 Teacher。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from idpo_common import idpo_path  # noqa: E402
from pipeline_common import (  # noqa: E402
    build_editor_prompt,
    load_config,
    read_jsonl,
    resolve_path,
    score,
    stage_path,
    write_json,
    write_jsonl,
)


def _output_response(text: str) -> str:
    return json.dumps({"output": str(text).strip()}, ensure_ascii=False, separators=(",", ":"))


def build(config: dict[str, Any]) -> dict[str, Any]:
    source_root = resolve_path(config["comparison"]["source_idpo_dir"])
    source_rollouts = source_root / "round_0" / "train_rollouts.jsonl"
    source_seeds = resolve_path(config["comparison"]["source_candidate_root"]) / str(
        config["comparison"]["source_adaptation_split"]
    ) / "03_seeds.jsonl"
    rollouts = read_jsonl(source_rollouts)
    seeds = {str(row["id"]): row for row in read_jsonl(source_seeds)}
    settings = config["global_idpo"]
    margin_threshold = float(settings.get("minimum_reward_margin", 0.03))
    parent_threshold = float(settings.get("minimum_parent_improvement", 0.02))
    pairs: list[dict[str, Any]] = []
    skipped = {"missing_seed": 0, "too_few_mutations": 0, "margin": 0, "parent": 0}
    for row in rollouts:
        seed = seeds.get(str(row["pseudo_query_id"]))
        if seed is None:
            skipped["missing_seed"] += 1
            continue
        mutations = [item for item in row.get("responses", []) if item.get("source") == "student_on_policy"]
        if len(mutations) < 2:
            skipped["too_few_mutations"] += 1
            continue
        target = str(seed["target"])
        parent = str(row["parent_a"]["text"])
        parent_reward = float(score(parent, target)["rouge_l"])
        scored = sorted(
            ((float(score(item.get("output", ""), target)["rouge_l"]), item) for item in mutations),
            key=lambda value: value[0],
        )
        rejected_reward, rejected = scored[0]
        chosen_reward, chosen = scored[-1]
        margin = chosen_reward - rejected_reward
        if margin < margin_threshold:
            skipped["margin"] += 1
            continue
        if chosen_reward - parent_reward < parent_threshold:
            skipped["parent"] += 1
            continue
        prompt = build_editor_prompt(
            seed,
            "mutation",
            {"candidate_id": row["parent_a"]["candidate_id"], "text": parent},
            None,
            int(config["generation"].get("maximum_history_records", 8)),
            supervision_mode="output_only",
            history_input_max_chars=int(config["simple_conditional_trace"].get("history_input_max_chars", 500)),
            history_output_max_chars=int(config["simple_conditional_trace"].get("history_output_max_chars", 300)),
        )
        pairs.append({
            "pair_id": f"{row['rollout_id']}:gated_output",
            "user_id": str(row.get("user_id", "")),
            "pseudo_query_id": str(row["pseudo_query_id"]),
            "operation_type": "mutation",
            "prompt": prompt,
            "chosen": _output_response(chosen["output"]),
            "rejected": _output_response(rejected["output"]),
            "chosen_trace_text": "",
            "rejected_trace_text": "",
            "chosen_output_text": _output_response(chosen["output"]),
            "rejected_output_text": _output_response(rejected["output"]),
            "chosen_output": str(chosen["output"]),
            "rejected_output": str(rejected["output"]),
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
            "parent_reward": parent_reward,
            "preference_margin": margin,
            "parent_improvement": chosen_reward - parent_reward,
            "preference_source": "mutation_only_parent_gated_rouge_l",
            "chosen_source": "student_on_policy",
            "rejected_source": "student_on_policy",
        })
    destination = idpo_path(config, 0, "train_pairs.jsonl")
    write_jsonl(destination, pairs)
    report = {
        "pairs": len(pairs),
        "users": len({item["user_id"] for item in pairs}),
        "minimum_reward_margin": margin_threshold,
        "minimum_parent_improvement": parent_threshold,
        "mutation_only": True,
        "output_only": True,
        "skipped": skipped,
    }
    write_json(idpo_path(config, 0, "train_pair_report.json"), report)
    print(f"Gated output-only pairs -> {destination}; report={report}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="30 - build gated output-only pairs")
    parser.add_argument("--config", default=str(HERE.parent / "config_global_same_mevo_gated_output.yaml"))
    args = parser.parse_args()
    build(load_config(args.config))
