"""阶段 22：在 IDPO 候选上训练并评估 per-user Ranker Head。

共享 DeBERTa Ranker backbone/head 来自阶段一的全局 Ranker checkpoint。
每个用户只更新独立 Linear Head；backbone 和全局 head 均保持冻结。

适配标签来自历史 Leave-One-Out Gold，但 Gold 只用于离线计算候选分数，
不会写入 Ranker 的 q+c 输入。当前 test Query 只用于最终预测与报告。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from common import user_head  # noqa: E402
from pipeline_common import (  # noqa: E402
    load_config,
    load_project_stage,
    read_jsonl,
    resolve_path,
    stage_path,
    write_json,
    write_jsonl,
)
from idpo_common import idpo_path, seed_with_score  # noqa: E402


def _candidate_with_score(
    candidate_id: str,
    kind: str,
    text: str,
    metrics: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate_id),
        "type": str(kind),
        "text": str(text).strip(),
        "scores": {
            "rouge_1": float(metrics.get("rouge_1", 0.0)),
            "rouge_l": float(metrics.get("rouge_l", 0.0)),
        },
        **extra,
    }


def _build_adaptation_rows(config: dict[str, Any], split: str, destination: Path) -> list[dict[str, Any]]:
    """Merge Seed parents and Gold-scored IDPO responses per LOO Query."""
    round_index = int(config["idpo"]["round"])
    seeds = read_jsonl(stage_path(config, f"adaptation_{split}", "seeds"))
    user_limit = int(config.get("idpo", {}).get("user_limit", 0))
    if user_limit > 0:
        selected_users = sorted(
            {str(row.get("user_id", row.get("parent_sample_id", ""))) for row in seeds}
        )[:user_limit]
        seeds = [
            row for row in seeds
            if str(row.get("user_id", row.get("parent_sample_id", ""))) in selected_users
        ]
    preferences = read_jsonl(
        idpo_path(config, round_index, f"{split}_preferences.jsonl")
    )
    if user_limit > 0:
        selected_ids = {str(row["id"]) for row in seeds}
        preferences = [
            row for row in preferences if str(row.get("pseudo_query_id", "")) in selected_ids
        ]
    seed_by_id = {str(row["id"]): row for row in seeds}
    by_query: dict[str, dict[str, Any]] = {}
    for row in seeds:
        target = str(row.get("target", "")).strip()
        if not target:
            raise ValueError(f"LOO Seed query={row.get('id')} 缺少隐藏 Gold 标签")
        by_query[str(row["id"])] = {
            **row,
            "parent_sample_id": str(row.get("user_id", row.get("parent_sample_id", ""))),
            "user_id": str(row.get("user_id", row.get("parent_sample_id", ""))),
            "candidates": [
                seed_with_score(item, target) for item in row.get("candidates", [])
            ],
            "mutations": [],
        }
    response_count = 0
    for rollout in preferences:
        query_id = str(rollout["pseudo_query_id"])
        if query_id not in by_query:
            raise KeyError(f"IDPO preference references missing Seed query={query_id}")
        reward_by_id = {
            str(item["response_id"]): item for item in rollout.get("response_rewards", [])
        }
        target_row = by_query[query_id]
        for response in rollout.get("responses", []):
            reward = reward_by_id.get(str(response["response_id"]))
            if reward is None:
                continue
            trace = response.get("trace") or {}
            target_row["mutations"].append(
                _candidate_with_score(
                    f"{rollout['rollout_id']}:{response['response_id']}",
                    "idpo_mutation",
                    str(response["output"]),
                    reward.get("metrics", {}),
                    parent_id=str(rollout["parent_a"]["candidate_id"]),
                    operation_type=str(rollout.get("operation_type", "mutation")),
                    operation_trace=trace,
                )
            )
            response_count += 1
    rows = [row for row in by_query.values() if row["mutations"]]
    if not rows:
        raise ValueError("没有包含有效 IDPO rollout 候选的 Ranker 适配数据")
    write_jsonl(destination, rows)
    write_json(
        destination.with_name("ranker_adaptation_manifest.json"),
        {
            "protocol": "idpo_loo_gold_ranker_adaptation_v2",
            "queries": len(rows),
            "users": len({str(row["user_id"]) for row in rows}),
            "scored_seed_parents": sum(len(row["candidates"]) for row in rows),
            "scored_idpo_responses": response_count,
            "gold_used_for_labels_only": True,
            "current_query_used": False,
        },
    )
    return rows


def _build_current_groups(config: dict[str, Any], split: str, destination: Path) -> list[dict[str, Any]]:
    """Create target-blind q+c groups for per-user head prediction."""
    round_index = int(config["idpo"]["round"])
    source = read_jsonl(
        idpo_path(config, round_index, f"{split}_current_editor_scored.jsonl")
    )
    user_limit = int(config.get("idpo", {}).get("user_limit", 0))
    if user_limit > 0:
        selected_users = sorted({str(row.get("user_id", "")) for row in source})[:user_limit]
        source = [row for row in source if str(row.get("user_id", "")) in selected_users]
    ranker_data = load_project_stage("code/07_build_ranker_data.py", "idpo_ranker_views")
    groups = []
    for row in source:
        group = ranker_data._group_view(row)
        group["sample_id"] = str(row["id"])
        groups.append(group)
    if not groups:
        raise ValueError("没有当前 Query 候选，无法评估 Ranker Head")
    write_jsonl(destination, groups)
    return source


def _compatibility_config(config: dict[str, Any], adaptation_source: Path, current_groups: Path, output_dir: Path) -> dict[str, Any]:
    ranker = {
        **config["scorer"],
        "shuffle_factor_directions": False,
        "factor_dropout_probability": 0.0,
    }
    if str(ranker.get("input_mode", "task_only")) != "task_only":
        raise ValueError("IDPO per-user Ranker 必须使用 q+c，scorer.input_mode=task_only")
    settings = {
        **config.get("user_adaptation", {}),
        "output_dir": str(output_dir),
        "global_model_dir": str(resolve_path(config["paths"]["scorer_output_dir"])),
        "adaptation_source": str(adaptation_source),
        "validation_candidates": str(current_groups),
        "profiles_per_user": "all",
        "validation_user_field": "user_id",
    }
    return {
        "project": {"seed": int(config["project"]["seed"])},
        "metric": dict(config["metric"]),
        "ranker": ranker,
        "user_adaptation": settings,
    }


def _evaluate_predictions(
    config: dict[str, Any], split: str, source: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    from common.metrics import score

    source_by_id = {str(row["id"]): row for row in source}
    user_rows = defaultdict(list)
    for prediction in predictions:
        source_row = source_by_id[str(prediction["sample_id"])]
        target = str(source_row["target"])
        ranked = list(prediction["ranked_candidates"])
        global_ranked = sorted(ranked, key=lambda item: (-float(item["initial_score"]), str(item["candidate_id"])))
        user_text = str(prediction["prediction"])
        global_text = str(global_ranked[0]["text"])
        user_metrics = score(user_text, target)
        global_metrics = score(global_text, target)
        oracle = max(score(str(item["text"]), target)["rouge_l"] for item in ranked)
        user_rows[str(source_row.get("user_id", ""))].append(
            {
                "sample_id": str(prediction["sample_id"]),
                "user_id": str(source_row.get("user_id", "")),
                "user_rouge_1": float(user_metrics["rouge_1"]),
                "user_rouge_l": float(user_metrics["rouge_l"]),
                "global_rouge_1": float(global_metrics["rouge_1"]),
                "global_rouge_l": float(global_metrics["rouge_l"]),
                "oracle_rouge_l": float(oracle),
                "user_hit_at_1": abs(float(user_metrics["rouge_l"]) - oracle) <= 1.0e-12,
                "global_hit_at_1": abs(float(global_metrics["rouge_l"]) - oracle) <= 1.0e-12,
            }
        )
    rows = [item for values in user_rows.values() for item in values]
    if not rows:
        raise ValueError("Ranker prediction 为空")
    summary = {
        "protocol": "idpo_per_user_ranker_head_v1",
        "split": split,
        "queries": len(rows),
        "users": len(user_rows),
        "global": {
            "rouge_1": sum(item["global_rouge_1"] for item in rows) / len(rows),
            "rouge_l": sum(item["global_rouge_l"] for item in rows) / len(rows),
            "hit_at_1": sum(item["global_hit_at_1"] for item in rows) / len(rows),
        },
        "per_user_head": {
            "rouge_1": sum(item["user_rouge_1"] for item in rows) / len(rows),
            "rouge_l": sum(item["user_rouge_l"] for item in rows) / len(rows),
            "hit_at_1": sum(item["user_hit_at_1"] for item in rows) / len(rows),
        },
        "oracle_rouge_l": sum(item["oracle_rouge_l"] for item in rows) / len(rows),
        "per_query": rows,
    }
    return summary


def run(config: dict[str, Any], split: str = "test") -> dict[str, Any]:
    round_index = int(config["idpo"]["round"])
    output_dir = idpo_path(config, round_index, "user_ranker")
    adaptation_source = output_dir / f"{split}_adaptation_rows.jsonl"
    current_groups = output_dir / f"{split}_current_groups.jsonl"
    adaptation_rows = _build_adaptation_rows(config, split, adaptation_source)
    current_source = _build_current_groups(config, split, current_groups)
    global_model = resolve_path(config["paths"]["scorer_output_dir"])
    if not (global_model / "ranker_head.pt").exists():
        raise FileNotFoundError(
            f"缺少共享 Ranker checkpoint: {global_model / 'ranker_head.pt'}; "
            "请先完成阶段一共享 Ranker 训练"
        )
    compatibility = _compatibility_config(
        config, adaptation_source, current_groups, output_dir
    )
    adaptation_report = user_head.adapt(compatibility)
    predictions = user_head.predict(compatibility)
    summary = _evaluate_predictions(config, split, current_source, predictions)
    summary["adaptation"] = adaptation_report
    write_json(output_dir / f"{split}_ranker_evaluation.json", summary)
    print(
        f"IDPO per-user Ranker -> {output_dir}; "
        f"queries={summary['queries']} users={summary['users']} "
        f"user_rouge_l={summary['per_user_head']['rouge_l']:.6f} "
        f"global_rouge_l={summary['global']['rouge_l']:.6f}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="22 - IDPO 后训练 per-user Ranker Head")
    parser.add_argument(
        "--config", default=str(HERE / "config_simple_trace_top8_idpo_first50.yaml")
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    run(load_config(args.config), args.split)


if __name__ == "__main__":
    main()
