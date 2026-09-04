#!/usr/bin/env python3
"""Run query-only Base/SFT controls after the queued no-seed suite.

This lightweight controller deliberately does not consume a GPU while waiting:

1. Once the merged no-history test Parent Pool has 608 rows, evaluate its
   greedy slot as the raw Llama2 query-only Base.
2. Once the main suite is complete, enqueue three matched controls without
   jumping ahead of the main experiments: query-only Direct SFT, frozen Base
   with Top-1, and Direct SFT with Top-1.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "code" / "26_9_3" / "scripts"
PYTHON = "/home/liux/miniconda3/envs/hydra/bin/python"
CONFIG = ROOT / "config_global_llama2_7b_visgpt_prime_matched.yaml"


def load_queue_module():
    path = SCRIPTS / "run_three_gpu_queue.py"
    spec = importlib.util.spec_from_file_location("mevo_three_gpu_queue", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return -1
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def wait_for(description: str, predicate, poll_seconds: int) -> None:
    while not predicate():
        print(f"FOLLOWUP_WAIT {description}", flush=True)
        time.sleep(poll_seconds)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    suffix = "_noseed_three_gpu_queue"
    run_name = args.run_name if args.run_name.endswith(suffix) else args.run_name + suffix
    data_root = ROOT / "dataset" / "parent_pools" / run_name
    result_root = ROOT / "result" / run_name
    log_root = ROOT / "logs" / run_name
    nohist_test = data_root / "no_history_pool" / "test_parent_pool.jsonl"
    zero_dir = result_root / "base_query_only"
    zero_report = zero_dir / "global_test_report.json"

    wait_for(
        "merged no-history test pool 608/608",
        lambda: jsonl_rows(nohist_test) == 608,
        args.poll_seconds,
    )
    if not zero_report.is_file():
        subprocess.run(
            [
                PYTHON,
                "-B",
                str(SCRIPTS / "evaluate_base_from_parent_pool.py"),
                "--pool",
                str(nohist_test),
                "--output-dir",
                str(zero_dir),
                "--input-mode",
                "query_only",
            ],
            cwd=ROOT,
            check=True,
        )
    print(f"FOLLOWUP_ZERO_SHOT_DONE report={zero_report}", flush=True)

    # Do not insert the additional SFT ahead of the main suite.
    wait_for(
        "main suite manifest",
        lambda: (result_root / "suite_manifest.json").is_file(),
        args.poll_seconds,
    )
    queue = load_queue_module()
    jobs = []
    direct_dir = result_root / "direct_no_history_sft"
    direct_report = direct_dir / "evaluation" / "global_test_report.json"
    if not direct_report.is_file():
        name = f"{run_name}_direct_no_history_sft"
        jobs.append(
            (
                name,
                queue.launch_local(
                    name,
                    0,
                    [
                        PYTHON,
                        "-B",
                        "code/26_9_3/run_sft_input_ablation.py",
                        "--experiment",
                        "direct_no_history_sft",
                        "--stage",
                        "all",
                        "--config",
                        str(CONFIG),
                        "--run-name",
                        name,
                        "--data-dir",
                        str(result_root / "direct_no_history_sft_data"),
                        "--run-dir",
                        str(direct_dir),
                        "--max-steps",
                        "430",
                    ],
                    log_root,
                ),
            )
        )

    remote_result = f"{queue.REMOTE_ROOT}/result/{run_name}"
    remote_log = f"{queue.REMOTE_ROOT}/logs"
    top1_base_dir = result_root / "base_top1"
    top1_base_report = top1_base_dir / "evaluation" / "global_test_report.json"
    if not top1_base_report.is_file():
        name = f"{run_name}_base_top1"
        jobs.append(
            (
                name,
                queue.launch_remote(
                    name,
                    0,
                    [
                        queue.REMOTE_PYTHON,
                        "-B",
                        "code/26_9_3/run_sft_input_ablation.py",
                        "--experiment",
                        "base_top1",
                        "--stage",
                        "all",
                        "--config",
                        queue.CONFIG.name,
                        "--run-name",
                        name,
                        "--data-dir",
                        f"{remote_result}/base_top1_data",
                        "--run-dir",
                        f"{remote_result}/base_top1",
                    ],
                    remote_log,
                ),
            )
        )

    top1_sft_dir = result_root / "direct_top1_sft"
    top1_sft_report = top1_sft_dir / "evaluation" / "global_test_report.json"
    if not top1_sft_report.is_file():
        name = f"{run_name}_direct_top1_sft"
        jobs.append(
            (
                name,
                queue.launch_remote(
                    name,
                    1,
                    [
                        queue.REMOTE_PYTHON,
                        "-B",
                        "code/26_9_3/run_sft_input_ablation.py",
                        "--experiment",
                        "direct_top1_sft",
                        "--stage",
                        "all",
                        "--config",
                        queue.CONFIG.name,
                        "--run-name",
                        name,
                        "--data-dir",
                        f"{remote_result}/direct_top1_sft_data",
                        "--run-dir",
                        f"{remote_result}/direct_top1_sft",
                        "--max-steps",
                        "430",
                    ],
                    remote_log,
                ),
            )
        )
    if jobs:
        queue.wait_jobs(jobs)
    for directory in ("base_top1", "direct_top1_sft"):
        subprocess.run(
            [
                "rsync",
                "-ah",
                f"{queue.REMOTE}:{remote_result}/{directory}/",
                str(result_root / directory) + "/",
            ],
            check=True,
        )

    comparison = {
        "protocol": "matched_history_k_raw_base_vs_direct_sft_v1",
        "run_name": run_name,
        "k0_raw_base": read_json(zero_report),
        "k0_direct_sft": read_json(direct_report),
        "k1_raw_base": read_json(top1_base_report),
        "k1_direct_sft": read_json(top1_sft_report),
        "k8_raw_base": read_json(result_root / "base/global_test_report.json"),
        "k8_direct_sft": read_json(
            result_root / "direct_sft/evaluation/global_test_report.json"
        ),
    }
    destination = result_root / "matched_history_causal_comparison.json"
    destination.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"QUERY_ONLY_CAUSAL_DONE report={destination}", flush=True)


if __name__ == "__main__":
    main()
