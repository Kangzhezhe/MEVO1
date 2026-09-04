#!/usr/bin/env python3
"""无 Seed SFT/Crossover 三 GPU 队列调度器。

资源：local GPU0、RTX3090 GPU0、RTX3090 GPU1。
阶段：
  1. 远端两卡生成 Main Pool，local GPU 同时训练 Direct SFT；Direct 完成后
     local GPU 接手 Main Pool 第三分片。
  2. 合并 Main Pool、无 GPU 生成 Base 报告，并用三卡生成 No-history Pool。
  3. Editor/Multitask/Crossover 三任务先占满三卡；任意一个结束后，自动启动
     No-history Crossover。所有任务用 marker 记录退出码，可重新运行续跑。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# .../MEVO_global_cot/code/26_9_3/scripts/this_file.py -> project root.
ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "code" / "26_9_3" / "scripts"
PYTHON = os.environ.get("PYTHON", "/home/liux/miniconda3/envs/hydra/bin/python")
REMOTE = os.environ.get("REMOTE", "RTX3090")
REMOTE_ROOT = os.environ.get(
    "REMOTE_ROOT", "/home_new/gp4_liux/kk/MEVO_global_cot"
)
REMOTE_PYTHON = os.environ.get(
    "REMOTE_PYTHON", "/home_new/gp4_liux/kk/envs/mevo-direct/bin/python"
)
CONFIG = ROOT / "config_global_llama2_7b_visgpt_prime_matched.yaml"
REMOTE_CONFIG = Path(REMOTE_ROOT) / CONFIG.name
TRAIN_TOTAL, TEST_TOTAL = 3643, 608
TRAIN_RANGES = [(0, 1215), (1215, 2430), (2430, 3643)]
TEST_RANGES = [(0, 203), (203, 406), (406, 608)]


def run(command: list[str], *, check: bool = True) -> None:
    print("RUN", " ".join(shlex.quote(x) for x in command), flush=True)
    subprocess.run(command, check=check)


def marker_path(log_root: Path, name: str) -> Path:
    return log_root / f"{name}.exit"


def launch_local(name: str, gpu: int, command: list[str], log_root: Path) -> Path:
    marker = marker_path(log_root, name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)
    marker.with_suffix(".pending").touch()
    command_text = " ".join(shlex.quote(item) for item in command)
    marker_text = shlex.quote(str(marker))
    log_text = shlex.quote(str(log_root / f"{name}.log"))
    shell = (
        f"set +e; cd {shlex.quote(str(ROOT))}; "
        f"CUDA_VISIBLE_DEVICES={gpu} {command_text} >> {log_text} 2>&1; "
        f"status=$?; echo $status > {marker_text}; exit $status"
    )
    run(["tmux", "new-session", "-d", "-s", name, "bash", "-lc", shell])
    print(f"LAUNCH local name={name} gpu={gpu}", flush=True)
    return marker


def launch_remote(name: str, gpu: int, command: list[str], remote_log_root: str) -> str:
    marker = f"{remote_log_root}/{name}.exit"
    command_text = " ".join(shlex.quote(item) for item in command)
    shell = (
        f"set +e; cd {shlex.quote(REMOTE_ROOT)}; "
        f"CUDA_VISIBLE_DEVICES={gpu} {command_text} >> {shlex.quote(remote_log_root)}/{name}.log 2>&1; "
        f"status=$?; echo $status > {shlex.quote(marker)}; exit $status"
    )
    run(["ssh", REMOTE, f": > {shlex.quote(marker + '.pending')}; screen -dmS {shlex.quote(name)} bash -lc {shlex.quote(shell)}"])
    print(f"LAUNCH remote name={name} gpu={gpu}", flush=True)
    return marker


def local_status(marker: Path) -> str:
    return marker.read_text().strip() if marker.exists() else ""


def remote_status(marker: str) -> str:
    result = subprocess.run(
        ["ssh", REMOTE, f"cat {shlex.quote(marker)} 2>/dev/null || true"],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def wait_jobs(jobs: list[tuple[str, str | Path]]) -> None:
    pending = list(jobs)
    while pending:
        remaining: list[tuple[str, str | Path]] = []
        for name, marker in pending:
            status = local_status(marker) if isinstance(marker, Path) else remote_status(marker)
            if not status:
                remaining.append((name, marker))
                continue
            if status != "0":
                raise RuntimeError(f"任务失败：{name}, exit={status}")
            print(f"DONE {name}", flush=True)
        pending = remaining
        if pending:
            time.sleep(15)


def wait_first_success(
    jobs: list[tuple[str, str, int, str | Path]],
) -> tuple[str, str, int, str | Path]:
    """Return the first completed job and retain the caller's ownership mapping."""
    while True:
        for name, location, gpu, marker in jobs:
            status = local_status(marker) if isinstance(marker, Path) else remote_status(marker)
            if not status:
                continue
            if status != "0":
                raise RuntimeError(f"任务失败：{name}, exit={status}")
            print(f"DONE {name}; released={location}:gpu{gpu}", flush=True)
            return name, location, gpu, marker
        time.sleep(15)


def sync_remote() -> None:
    run(["ssh", REMOTE, "mkdir", "-p", f"{REMOTE_ROOT}/code/26_9_3/scripts", f"{REMOTE_ROOT}/code/common", f"{REMOTE_ROOT}/dataset/candidate_sets/global_llama2_7b_visgpt_train", f"{REMOTE_ROOT}/dataset/candidate_sets/global_llama2_7b_visgpt_test100_full", f"{REMOTE_ROOT}/dataset/parent_pools", f"{REMOTE_ROOT}/result", f"{REMOTE_ROOT}/logs"])
    run(["rsync", "-ah", "--delete", str(ROOT / "code" / "26_9_3") + "/", f"{REMOTE}:{REMOTE_ROOT}/code/26_9_3/"])
    run(["rsync", "-ah", str(ROOT / "code" / "common") + "/", f"{REMOTE}:{REMOTE_ROOT}/code/common/"])
    run(["rsync", "-ah", str(ROOT / "code" / "06_train_editor_lora.py"), str(ROOT / "code" / "pipeline_common.py"), f"{REMOTE}:{REMOTE_ROOT}/code/"])
    run(["rsync", "-ah", str(CONFIG), str(ROOT / "config_global.yaml"), f"{REMOTE}:{REMOTE_ROOT}/"])
    for split in ("train", "test100_full"):
        local = ROOT / "dataset" / "candidate_sets" / f"global_llama2_7b_visgpt_{split}" / "02_retrieved.jsonl"
        remote_dir = f"{REMOTE}:{REMOTE_ROOT}/dataset/candidate_sets/global_llama2_7b_visgpt_{split}/"
        run(["rsync", "-ah", str(local), remote_dir])


def pool_command(
    python: str,
    config: str,
    root: str,
    shard: int,
    ablation: str,
    train_range: tuple[int, int],
    test_range: tuple[int, int],
) -> list[str]:
    return [
        python,
        "-B",
        "code/26_9_3/build_shared_parent_pool.py",
        "--config",
        config,
        "--split",
        "all",
        "--ablation",
        ablation,
        "--data-dir",
        root,
        "--shard-name",
        f"shard_{shard}",
        "--train-start",
        str(train_range[0]),
        "--train-end",
        str(train_range[1]),
        "--test-start",
        str(test_range[0]),
        "--test-end",
        str(test_range[1]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="三 GPU 无 Seed 实验队列")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stamp = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{stamp}_noseed_three_gpu_queue"
    data_root = ROOT / "dataset" / "parent_pools" / run_name
    pool_root = data_root / "main_pool"
    nohist_root = data_root / "no_history_pool"
    result_root = ROOT / "result" / run_name
    log_root = ROOT / "logs" / run_name
    remote_pool_root = f"{REMOTE_ROOT}/dataset/parent_pools/{run_name}/main_pool"
    remote_nohist_root = f"{REMOTE_ROOT}/dataset/parent_pools/{run_name}/no_history_pool"
    for path in (pool_root, nohist_root, result_root, log_root):
        path.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps({"run_name": run_name, "pool_root": str(pool_root), "result_root": str(result_root), "remote": REMOTE}, ensure_ascii=False, indent=2))
        return
    run(["tmux", "kill-session", "-t", "crossover_main_2693"], check=False)
    run(["ssh", REMOTE, "hostname"])
    sync_remote()

    # 阶段1：两张远端卡建 Pool，local 卡同时做 Direct SFT。任何一张先空
    # 出来的卡接手第三分片，避免它因固定 GPU 绑定而闲置。
    pool_jobs: list[tuple[str, str | Path]] = []
    for shard in (0, 1):
        name = f"{run_name}_main_pool_{shard}"
        marker = launch_remote(name, shard, pool_command(REMOTE_PYTHON, CONFIG.name, remote_pool_root, shard, "main", TRAIN_RANGES[shard], TEST_RANGES[shard]), f"{REMOTE_ROOT}/logs")
        pool_jobs.append((name, marker))
    direct_name = f"{run_name}_direct_sft"
    direct_marker = launch_local(direct_name, 0, [PYTHON, "-B", "code/26_9_3/run_sft_input_ablation.py", "--experiment", "direct_sft", "--stage", "all", "--config", str(CONFIG), "--run-name", direct_name, "--data-dir", str(result_root / "direct_sft_data"), "--run-dir", str(result_root / "direct_sft"), "--max-steps", "430"], log_root)
    third_name = f"{run_name}_main_pool_2"
    _, released_location, released_gpu, _ = wait_first_success(
        [
            (f"{run_name}_main_pool_0", "remote", 0, pool_jobs[0][1]),
            (f"{run_name}_main_pool_1", "remote", 1, pool_jobs[1][1]),
            (direct_name, "local", 0, direct_marker),
        ]
    )
    if released_location == "local":
        third_marker: str | Path = launch_local(
            third_name,
            released_gpu,
            pool_command(PYTHON, str(CONFIG), str(pool_root), 2, "main", TRAIN_RANGES[2], TEST_RANGES[2]),
            log_root,
        )
    else:
        third_marker = launch_remote(
            third_name,
            released_gpu,
            pool_command(REMOTE_PYTHON, CONFIG.name, remote_pool_root, 2, "main", TRAIN_RANGES[2], TEST_RANGES[2]),
            f"{REMOTE_ROOT}/logs",
        )
    wait_jobs(pool_jobs + [(direct_name, direct_marker), (third_name, third_marker)])
    if isinstance(third_marker, str):
        (pool_root / "shard_2").mkdir(parents=True, exist_ok=True)
        run(["rsync", "-ah", f"{REMOTE}:{remote_pool_root}/shard_2/", str(pool_root / "shard_2") + "/"])
    for shard in (0, 1):
        (pool_root / f"shard_{shard}").mkdir(parents=True, exist_ok=True)
        run(["rsync", "-ah", f"{REMOTE}:{remote_pool_root}/shard_{shard}/", str(pool_root / f"shard_{shard}") + "/"])
    for split, expected in (("train", TRAIN_TOTAL), ("test", TEST_TOTAL)):
        run([PYTHON, "-B", str(SCRIPT_DIR / "merge_parent_pool_shards.py"), "--shard-root", str(pool_root), "--output-dir", str(pool_root), "--split", split, "--expected", str(expected), "--expected-shards", "3"])
    run([PYTHON, "-B", str(SCRIPT_DIR / "evaluate_base_from_parent_pool.py"), "--pool", str(pool_root / "test_parent_pool.jsonl"), "--output-dir", str(result_root / "base")])
    run(["rsync", "-ah", "--delete", str(pool_root) + "/", f"{REMOTE}:{remote_pool_root}/"])

    # 阶段2：三卡并行生成 No-history Pool。
    nohist_jobs: list[tuple[str, str | Path]] = []
    local_name = f"{run_name}_nohist_pool_2"
    nohist_jobs.append((local_name, launch_local(local_name, 0, pool_command(PYTHON, str(CONFIG), str(nohist_root), 2, "no_history", TRAIN_RANGES[2], TEST_RANGES[2]), log_root)))
    for shard in (0, 1):
        name = f"{run_name}_nohist_pool_{shard}"
        nohist_jobs.append((name, launch_remote(name, shard, pool_command(REMOTE_PYTHON, CONFIG.name, remote_nohist_root, shard, "no_history", TRAIN_RANGES[shard], TEST_RANGES[shard]), f"{REMOTE_ROOT}/logs")))
    wait_jobs(nohist_jobs)
    for shard in (0, 1):
        (nohist_root / f"shard_{shard}").mkdir(parents=True, exist_ok=True)
        run(["rsync", "-ah", f"{REMOTE}:{remote_nohist_root}/shard_{shard}/", str(nohist_root / f"shard_{shard}") + "/"])
    for split, expected in (("train", TRAIN_TOTAL), ("test", TEST_TOTAL)):
        run([PYTHON, "-B", str(SCRIPT_DIR / "merge_parent_pool_shards.py"), "--shard-root", str(nohist_root), "--output-dir", str(nohist_root), "--split", split, "--expected", str(expected), "--expected-shards", "3"])
    run(["rsync", "-ah", "--delete", str(nohist_root) + "/", f"{REMOTE}:{remote_nohist_root}/"])

    # 阶段3：三个主任务先占三张卡。
    editor_name = f"{run_name}_editor_sft"
    editor_marker = launch_local(editor_name, 0, [PYTHON, "-B", "code/26_9_3/run_sft_input_ablation.py", "--experiment", "editor_sft", "--stage", "all", "--config", str(CONFIG), "--run-name", editor_name, "--data-dir", str(result_root / "editor_sft_data"), "--run-dir", str(result_root / "editor_sft"), "--shared-parent-pool-dir", str(pool_root), "--max-steps", "430"], log_root)
    multi_name = f"{run_name}_multitask_sft"
    multi_marker = launch_remote(multi_name, 0, [REMOTE_PYTHON, "-B", "code/26_9_3/run_sft_input_ablation.py", "--experiment", "multitask_sft", "--stage", "all", "--config", CONFIG.name, "--run-name", multi_name, "--data-dir", f"{REMOTE_ROOT}/result/{run_name}/multitask_sft_data", "--run-dir", f"{REMOTE_ROOT}/result/{run_name}/multitask_sft", "--shared-parent-pool-dir", remote_pool_root, "--teacher-mode", "api", "--max-steps", "430"], f"{REMOTE_ROOT}/logs")
    cross_name = f"{run_name}_crossover_sft"
    cross_marker = launch_remote(cross_name, 1, [REMOTE_PYTHON, "-B", "code/26_9_3/run_gold_multi_parent_crossover.py", "--stage", "all", "--config", CONFIG.name, "--ablation", "main", "--run-name", cross_name, "--data-dir", f"{REMOTE_ROOT}/result/{run_name}/crossover_sft_data", "--run-dir", f"{REMOTE_ROOT}/result/{run_name}/crossover_sft", "--shared-parent-pool-dir", remote_pool_root, "--max-steps", "430"], f"{REMOTE_ROOT}/logs")
    running = [(editor_name, editor_marker), (multi_name, multi_marker), (cross_name, cross_marker)]
    owner = None
    while owner is None:
        for name, marker in running:
            status = local_status(marker) if isinstance(marker, Path) else remote_status(marker)
            if status:
                if status != "0":
                    raise RuntimeError(f"任务失败：{name}, exit={status}")
                owner = "local" if isinstance(marker, Path) else ("remote0" if name == multi_name else "remote1")
                print(f"QUEUE_RELEASED owner={owner} job={name}", flush=True)
                break
        if owner is None:
            time.sleep(15)
    no_name = f"{run_name}_nohist_crossover"
    if owner == "local":
        no_marker = launch_local(no_name, 0, [PYTHON, "-B", "code/26_9_3/run_gold_multi_parent_crossover.py", "--stage", "all", "--config", str(CONFIG), "--ablation", "no_history", "--run-name", no_name, "--data-dir", str(result_root / "nohist_crossover_data"), "--run-dir", str(result_root / "nohist_crossover"), "--shared-parent-pool-dir", str(nohist_root), "--max-steps", "430"], log_root)
    else:
        gpu = 0 if owner == "remote0" else 1
        no_marker = launch_remote(no_name, gpu, [REMOTE_PYTHON, "-B", "code/26_9_3/run_gold_multi_parent_crossover.py", "--stage", "all", "--config", CONFIG.name, "--ablation", "no_history", "--run-name", no_name, "--data-dir", f"{REMOTE_ROOT}/result/{run_name}/nohist_crossover_data", "--run-dir", f"{REMOTE_ROOT}/result/{run_name}/nohist_crossover", "--shared-parent-pool-dir", remote_nohist_root, "--max-steps", "430"], f"{REMOTE_ROOT}/logs")
    wait_jobs(running + [(no_name, no_marker)])
    for directory in ("multitask_sft", "crossover_sft", "nohist_crossover"):
        run(["rsync", "-ah", f"{REMOTE}:{REMOTE_ROOT}/result/{run_name}/{directory}/", str(result_root / directory) + "/"], check=False)
    manifest = {"run_name": run_name, "protocol": "noseed_three_gpu_queued_suite_v1", "main_parent_pool": str(pool_root), "no_history_parent_pool": str(nohist_root), "fixed_parent_slots": 5, "expected_users": 100, "expected_queries": 608, "teacher_used_for_parent": False}
    (result_root / "suite_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"QUEUE_DONE result_root={result_root}", flush=True)


if __name__ == "__main__":
    main()
