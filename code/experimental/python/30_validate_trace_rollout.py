"""阶段 30：在正式 IDPO 前验证共享 Editor 的新用户 Trace 覆盖率。

每位用户只取一个 LOO Query，先请求完整条件偏好 Trace；结构/证据门控失败时
沿用正式路线回退 output-only。报告用于阻止“名义上 Trace-aware、实际上全量
回退”的昂贵实验。
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from idpo_common import idpo_path  # noqa: E402
from pipeline_common import load_config, read_jsonl, stage_path, write_json  # noqa: E402


def _rollout_module():
    spec = importlib.util.spec_from_file_location(
        "trace_rollout_pilot_stage", HERE / "17_idpo_rollout.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run(config: dict[str, Any], users: int, samples: int, minimum_rate: float) -> dict[str, Any]:
    import torch

    local = copy.deepcopy(config)
    settings = local["idpo"]
    settings["rollout_samples"] = int(samples)
    settings["minimum_valid_responses"] = max(2, int(samples) // 2)
    settings["rollout_batch_size"] = min(
        int(settings.get("rollout_batch_size", samples)), int(samples)
    )
    source = read_jsonl(stage_path(local, "adaptation_test", "seeds"))
    selected = []
    seen = set()
    for row in source:
        user_id = str(row["user_id"])
        if user_id in seen:
            continue
        seen.add(user_id)
        selected.append(row)
        if len(selected) >= int(users):
            break
    if len(selected) < int(users):
        raise ValueError(f"Trace pilot仅找到{len(selected)}位用户，要求{users}")

    module = _rollout_module()
    specs = [module._build_operation_specs(row, local)[0] for row in selected]
    editor = module._editor_for_user(local, str(selected[0]["user_id"]))
    try:
        results = module._run_operation_batch(specs, editor, local)
    finally:
        del editor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    trace = [row for row in results if bool(row.get("trace_aware"))]
    fallback = [row for row in results if bool(row.get("trace_fallback"))]
    valid = [row for row in results if bool(row.get("minimum_responses_met"))]
    report = {
        "protocol": "new_user_conditional_trace_rollout_gate_v1",
        "users": len(selected),
        "queries": len(results),
        "samples_per_query": int(samples),
        "trace_queries": len(trace),
        "fallback_queries": len(fallback),
        "valid_queries": len(valid),
        "trace_query_rate": len(trace) / len(results),
        "minimum_trace_query_rate": float(minimum_rate),
        "gate_passed": len(trace) / len(results) >= float(minimum_rate),
        "examples": [
            {
                "user_id": row["user_id"],
                "pseudo_query_id": row["pseudo_query_id"],
                "mode": "trace" if row.get("trace_aware") else "output_only_fallback",
                "valid_responses": len(row["responses"]),
                "response": row["responses"][0]["trace"] if row["responses"] else None,
            }
            for row in results
        ],
    }
    destination = idpo_path(local, int(settings["round"]), "trace_rollout_pilot.json")
    write_json(destination, report)
    print(f"Trace rollout pilot -> {destination}; report={json.dumps(report, ensure_ascii=False)}")
    if not report["gate_passed"]:
        raise RuntimeError(
            f"Trace覆盖率{report['trace_query_rate']:.1%}低于门槛{minimum_rate:.1%}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="30 - 新用户Trace rollout覆盖率门控")
    parser.add_argument(
        "--config",
        default=str(HERE / "config_conditional_trace_idpo_first50_warmup.yaml"),
    )
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--minimum-trace-rate", type=float, default=0.10)
    args = parser.parse_args()
    run(load_config(args.config), args.users, args.samples, args.minimum_trace_rate)


if __name__ == "__main__":
    main()
