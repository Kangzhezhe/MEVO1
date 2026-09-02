"""阶段 21：在未参与 LOO 适配的当前 dev Query 上评估 per-user Editor。

共享 S1 与每用户 IDPO Adapter 使用完全相同的4个 target-blind Seed 和候选预算。
本阶段报告候选 Oracle，专门回答 IDPO 是否改善了候选空间；它不混入 Ranker。
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from idpo_common import idpo_path  # noqa: E402
from pipeline_common import (  # noqa: E402
    load_config,
    load_project_stage,
    read_jsonl,
    stage_path,
    write_json,
    write_jsonl,
)


def _pool(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(row.get("candidates", [])) + list(row.get("mutations", []))


def _oracle(row: dict[str, Any], metric: str) -> tuple[float, str]:
    values = _pool(row)
    if not values:
        raise ValueError(f"sample={row.get('id')} 候选池为空")
    best = max(values, key=lambda item: float(item["scores"][metric]))
    return float(best["scores"][metric]), str(best["text"])


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("没有可评估的 IDPO 当前 Query")
    rouge_1 = [_oracle(row, "rouge_1")[0] for row in rows]
    rouge_l = [_oracle(row, "rouge_l")[0] for row in rows]
    invalid = sum(
        int(row.get("editor_metadata", {}).get("invalid_generations", 0))
        for row in rows
    )
    calls = sum(len(row.get("mutations", [])) for row in rows)
    return {
        "queries": len(rows),
        "oracle_rouge_1": sum(rouge_1) / len(rouge_1),
        "oracle_rouge_l": sum(rouge_l) / len(rouge_l),
        "invalid_generations": invalid,
        "json_valid_rate": (calls - invalid) / calls if calls else 0.0,
    }


def evaluate(config: dict, split: str = "validation") -> dict[str, Any]:
    import torch

    settings = config["idpo"]
    round_index = int(settings["round"])
    adaptation = read_jsonl(stage_path(config, f"adaptation_{split}", "prepare"))
    selected_users = sorted({str(row["user_id"]) for row in adaptation})
    source = [
        row
        for row in read_jsonl(stage_path(config, split, "seeds"))
        if str(row.get("user_id", row["id"])) in selected_users
    ]
    baseline_by_id = {
        str(row["id"]): row
        for row in read_jsonl(stage_path(config, split, "editor"))
        if str(row.get("user_id", row["id"])) in selected_users
    }
    if {str(row["id"]) for row in source} - set(baseline_by_id):
        raise ValueError("共享 S1 baseline 缺少所选用户的当前 Query")

    destination = idpo_path(
        config, round_index, f"{split}_current_editor_scored.jsonl"
    )
    existing = read_jsonl(destination) if destination.exists() else []
    done = {str(row["id"]): row for row in existing}
    by_user: dict[str, list[dict[str, Any]]] = {}
    for row in source:
        if str(row["id"]) in done:
            continue
        user_id = str(row.get("user_id", row["id"]))
        by_user.setdefault(user_id, []).append(row)

    editor_module = load_project_stage(
        "code/07_generate_editor_pool.py", "idpo_user_editor_eval"
    )
    adapter_root = idpo_path(config, round_index, "user_adapters")

    def checkpoint() -> None:
        write_jsonl(
            destination,
            [done[str(row["id"])] for row in source if str(row["id"]) in done],
        )

    for index, user_id in enumerate(sorted(by_user), 1):
        adapter = adapter_root / f"user_{user_id}"
        if not adapter.exists():
            print(f"IDPO eval skip user={user_id}: adapter不存在", flush=True)
            continue
        local = copy.deepcopy(config)
        local.setdefault("model", {})["adapter_path"] = str(adapter)
        editor = editor_module.LocalEditor(local)
        try:
            # Keep one user Adapter loaded and batch all of this user's current
            # Query prompts together.  The helper runs mutation and crossover
            # as two cross-Query batches, then restores row order.
            generated = editor_module.generate_many_rows(by_user[user_id], local, editor)
            for row, result in zip(by_user[user_id], generated):
                done[str(row["id"])] = result
        finally:
            del editor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        checkpoint()
        print(
            f"IDPO current-query eval user {index}/{len(by_user)} "
            f"user={user_id} queries={len(by_user[user_id])}",
            flush=True,
        )
    checkpoint()

    adapted = [done[str(row["id"])] for row in source if str(row["id"]) in done]
    baseline = [baseline_by_id[str(row["id"])] for row in adapted]
    baseline_summary = _summary(baseline)
    adapted_summary = _summary(adapted)
    changes = []
    for before, after in zip(baseline, adapted):
        before_score, before_title = _oracle(before, "rouge_l")
        after_score, after_title = _oracle(after, "rouge_l")
        changes.append(
            {
                "sample_id": str(after["id"]),
                "user_id": str(after.get("user_id", after["id"])),
                "baseline_oracle_rouge_l": before_score,
                "idpo_oracle_rouge_l": after_score,
                "delta": after_score - before_score,
                "baseline_oracle_title": before_title,
                "idpo_oracle_title": after_title,
            }
        )
    deltas = [row["delta"] for row in changes]
    report = {
        "protocol": "per_user_editor_idpo_current_query_oracle_v1",
        "split": split,
        "users_requested": len(selected_users),
        "users_evaluated": len({row["user_id"] for row in changes}),
        "queries": len(changes),
        "baseline_shared_s1": baseline_summary,
        "per_user_idpo": adapted_summary,
        "oracle_rouge_l_delta": sum(deltas) / len(deltas),
        "improved_queries": sum(value > 1.0e-12 for value in deltas),
        "degraded_queries": sum(value < -1.0e-12 for value in deltas),
        "unchanged_queries": sum(abs(value) <= 1.0e-12 for value in deltas),
        "loo_gold_used_for_training_labels": True,
        "current_query_gold_used_for_generation": False,
        "ranker_used": False,
        "changes": changes,
    }
    report_path = idpo_path(
        config, round_index, f"{split}_current_editor_evaluation.json"
    )
    write_json(report_path, report)
    print(
        "IDPO current-query evaluation -> "
        f"{report_path}; oracle_rl_delta={report['oracle_rouge_l_delta']:.6f}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="21 - Evaluate per-user IDPO Editor")
    parser.add_argument(
        "--config", default=str(HERE / "config_simple_trace_top8_idpo_first50.yaml")
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    evaluate(load_config(args.config), args.split)


if __name__ == "__main__":
    main()
