"""平均不同历史划分训练出的 Linear Head，降低 per-user 排序方差。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from pipeline_common import (  # noqa: E402
    load_config,
    load_project_stage,
    resolve_path,
    write_json,
    write_jsonl,
)


def _mean_state(paths: list[Path]) -> dict[str, torch.Tensor]:
    states = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise ValueError("Ensemble Head state keys do not match")
    return {
        key: torch.stack([state[key].float() for state in states]).mean(dim=0)
        for key in sorted(keys)
    }


def run(config: dict[str, Any]) -> Path:
    settings = config["user_head_ensemble"]
    matrix = load_project_stage(
        "code/24_compare_user_head_objectives.py",
        "user_head_objective_matrix_for_ensemble",
    )
    cache_path = resolve_path(settings["feature_cache"])
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    current = payload["current"]
    initial_state = payload["initial_state"]
    hidden_size = int(payload["hidden_size"])
    users = sorted(current)
    source_dirs = [resolve_path(value) for value in settings["source_dirs"]]
    variants = {
        "top_pairs_soup": ["top_pairs"],
        "listwise_kl_soup": ["listwise_kl"],
        "top_pairs_listwise_soup": ["top_pairs", "listwise_kl"],
    }
    output_dir = resolve_path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    predictions = []
    for variant, objectives in variants.items():
        states = {}
        source_count = 0
        for user_id in users:
            paths = [
                root / objective / "user_heads" / f"user_{user_id}.pt"
                for root in source_dirs
                for objective in objectives
            ]
            missing = [path for path in paths if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Missing ensemble heads: {missing}")
            states[user_id] = _mean_state(paths)
            source_count = len(paths)
        alphas = {user_id: 1.0 for user_id in users}
        report, rows = matrix._evaluate_route(
            variant,
            "head_soup",
            current,
            states,
            initial_state,
            hidden_size,
            alphas,
        )
        report["heads_averaged_per_user"] = source_count
        report["source_objectives"] = objectives
        reports[variant] = report
        predictions.extend(rows)
        print(
            f"head soup={variant} rouge_l={report['query_macro']['rouge_l']:.6f} "
            f"hit1={report['hit_at_1']:.4f} regret={report['mean_regret']:.6f}",
            flush=True,
        )
    summary = {
        "protocol": "multi_split_per_user_linear_head_parameter_soup_v1",
        "source_dirs": [str(path) for path in source_dirs],
        "current_query_gold_used_for_ensemble": False,
        "results": reports,
    }
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    return output_dir / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble multi-split per-user Heads")
    parser.add_argument("--config", default=str(HERE / "config_user_head_ensemble.yaml"))
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
