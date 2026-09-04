#!/usr/bin/env python3
"""无 Seed 的 SFT 输入消融完整流程。

本脚本统一实现四个实验：

``base``
    冻结 Llama2-7B：Query + Top-8 History -> Title。
``direct_sft``
    Query + Top-8 History -> Gold。
``editor_sft``
    Base 先由 Query + Top-8 History 直接生成一个 Parent；随后
    Query + Top-8 History + Parent -> Gold。
``multitask_sft``
    与 editor_sft 使用同一个无 Seed Parent，并联合训练 Title 主任务和
    Rationale 辅助任务。Teacher 仅离线生成 rationale，不参与 Parent 生成。

所有输入都来自 ``02_retrieved.jsonl``，不读取 ``03_seeds.jsonl``，也不使用
``candidates[0]`` 或 ``PARENT TITLE`` 作为 Parent 生成条件。代码中的随机数
random_seed 只用于复现，不是 Seed Parent。
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CODE = HERE.parent
PROJECT = CODE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(PROJECT))

from common.concurrency import BoundedJobError, run_bounded  # noqa: E402
from pipeline_common import (  # noqa: E402
    deterministic_split,
    load_config,
    read_jsonl,
    resolve_path,
    teacher_client,
    visible_history,
    write_json,
    write_jsonl,
)
from run_gold_multi_parent_crossover import (  # noqa: E402
    AdapterGenerator,
    BaseParentSampler,
    base_prompt,
    clean_title,
    configure,
    evaluate_rows,
    import_module,
    load_source_rows,
    verify_formal_test,
)


def editor_prompt(
    row: dict[str, Any], parent: str, maximum_history: int, task: str = "title"
) -> str:
    """构造无 Seed Parent 的 Editor Title/Rationale 输入。"""

    payload = {
        "current_input": str(row.get("source_text", "")),
        "retrieved_history": visible_history(row, maximum_history),
        "parent": clean_title(parent),
    }
    if task == "title":
        instruction = (
            "You are a personalized academic-title editor. Improve PARENT using factual "
            "information from CURRENT_INPUT. Use RETRIEVED_HISTORY only for applicable "
            "recurring user preferences. Return exactly one line containing only the final "
            "title; do not output JSON, labels, or explanations."
        )
        suffix = "FINAL TITLE"
    elif task == "rationale":
        instruction = (
            "Explain in one or two concise sentences how PARENT should be edited for "
            "CURRENT_INPUT. Mention an applicable history pattern only when the visible "
            "examples support it. Do not claim access to a gold or reference answer. "
            "Return only the concise editing rationale."
        )
        suffix = "EDITING RATIONALE"
    else:
        raise ValueError(f"未知 task={task}")
    return (
        instruction
        + "\n\nPAYLOAD:\n"
        + json.dumps(payload, ensure_ascii=False)
        + f"\n\n{suffix}:\n"
    )


def teacher_rationale_prompt(
    row: dict[str, Any], parent: str, maximum_history: int
) -> str:
    """Teacher 可见训练 Gold，只生成辅助解释，不生成 Parent。"""

    payload = {
        "current_input": str(row.get("source_text", "")),
        "retrieved_history": visible_history(row, maximum_history),
        "parent": clean_title(parent),
        "gold": clean_title(row.get("target", "")),
    }
    return (
        "You are an offline annotation teacher. Explain the most important edit that "
        "transforms PARENT toward GOLD while remaining factually supported by CURRENT_INPUT. "
        "Use history only if it gives a genuinely applicable recurring preference. Write one "
        "or two concise sentences. Do not copy the complete GOLD title into the rationale. "
        "Return exactly one JSON object: {\"rationale\":\"...\"}.\n\nPAYLOAD:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def build_one_parent_pool(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    destination: Path,
    limit: int,
) -> list[dict[str, Any]]:
    """直接从 Query/History 生成一个 greedy Parent，无任何初始标题。"""

    if limit > 0:
        rows = rows[:limit]
    protocol = "base_direct_query_history_greedy_parent_v1"
    previous = read_jsonl(destination) if destination.exists() else []
    cache = {str(item.get("id", "")): item for item in previous}
    output: list[dict[str, Any]] = []
    sampler: BaseParentSampler | None = None
    maximum_history = int(config["sft_data"]["maximum_history_records"])
    random_seed = int(config["training"]["seed"])
    checkpoint_every = int(config["crossover_sft"]["checkpoint_every"])
    try:
        for index, row in enumerate(rows, 1):
            sample_id = str(row.get("id", ""))
            cached = cache.get(sample_id)
            if cached and cached.get("parent_protocol") == protocol:
                output.append(cached)
                print(
                    f"parent {index}/{len(rows)} sample={sample_id} source=cache",
                    flush=True,
                )
                continue
            if sampler is None:
                sampler = BaseParentSampler(config)
            prompt = base_prompt(row, maximum_history, include_history=True)
            generated = sampler.generate(
                prompt,
                1,
                sample=False,
                temperature=float(config["crossover_sft"]["temperature"]),
                top_p=float(config["crossover_sft"]["top_p"]),
                seed=random_seed + index,
            )
            parent = clean_title(generated[0]["text"] if generated else "")
            if not parent:
                raise ValueError(f"sample={sample_id} Base direct Parent 为空")
            item = dict(row)
            item.update(
                {
                    "id": sample_id,
                    "parent": parent,
                    "parent_generation_prompt": prompt,
                    "parent_protocol": protocol,
                    "seed_parent_used": False,
                }
            )
            output.append(item)
            print(
                f"parent {index}/{len(rows)} sample={sample_id} chars={len(parent)}",
                flush=True,
            )
            if checkpoint_every > 0 and index % checkpoint_every == 0:
                write_jsonl(destination, output)
        write_jsonl(destination, output)
        return output
    finally:
        if sampler is not None:
            sampler.close()


def annotate_rationales(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    destination: Path,
    teacher_mode: str,
) -> dict[str, str]:
    """为 Multitask 辅助任务生成简短 rationale，并支持断点续跑。"""

    cached_rows = read_jsonl(destination) if destination.exists() else []
    results = {
        str(item["id"]): str(item.get("rationale", "")).strip()
        for item in cached_rows
        if str(item.get("rationale", "")).strip()
    }
    jobs = [row for row in rows if str(row.get("id", "")) not in results]
    maximum_history = int(config["sft_data"]["maximum_history_records"])
    if teacher_mode == "heuristic":
        for row in jobs:
            results[str(row["id"])] = (
                "Revise the parent to preserve the current input's central method and "
                "research objective while improving clarity and concision."
            )
    elif teacher_mode == "api":
        client = teacher_client(config)
        retries = int(config.get("rationale_sft", {}).get("schema_retries", 2))

        def worker(row: dict[str, Any]) -> tuple[str, str]:
            prompt = teacher_rationale_prompt(row, row["parent"], maximum_history)
            last_error: Exception | None = None
            for attempt in range(retries + 1):
                task = f"noseed_multitask_rationale_{row['id']}_{attempt}"
                try:
                    payload, _ = client.json(task, prompt, {"sample_id": str(row["id"])})
                    if not isinstance(payload, dict):
                        raise ValueError("Teacher 返回值不是 object")
                    rationale = str(payload.get("rationale", "")).strip()
                    if not 10 <= len(rationale) <= 500:
                        raise ValueError("rationale 长度无效")
                    return str(row["id"]), rationale
                except Exception as error:
                    last_error = error
                    client.invalidate(task, prompt)
            raise RuntimeError(f"sample={row['id']} rationale 失败：{last_error}")

        def on_result(
            row: dict[str, Any], result: tuple[str, str], completed: int
        ) -> None:
            sample_id, rationale = result
            results[sample_id] = rationale
            if completed % 20 == 0:
                write_jsonl(
                    destination,
                    [{"id": key, "rationale": value} for key, value in results.items()],
                )
            print(
                f"rationale {completed}/{len(jobs)} sample={sample_id}", flush=True
            )

        try:
            run_bounded(
                jobs,
                worker,
                on_result,
                max_workers=int(config.get("rationale_sft", {}).get("concurrency", 2)),
                thread_name_prefix="noseed-rationale",
            )
        except BoundedJobError as error:
            raise RuntimeError(str(error)) from error.error
    else:
        raise ValueError(f"未知 teacher_mode={teacher_mode}")
    write_jsonl(
        destination,
        [{"id": key, "rationale": value} for key, value in results.items()],
    )
    return results


def compile_examples(
    experiment: str,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    rationales: dict[str, str] | None,
    smoke: bool,
) -> list[dict[str, Any]]:
    """将三种 SFT 输入编译为通用 LoRA Trainer 接受的格式。"""

    fraction = float(config["sft_data"]["validation_fraction"])
    maximum_history = int(config["sft_data"]["maximum_history_records"])
    rationale_weight = float(config.get("rationale_sft", {}).get("rationale_loss_weight", 0.1))
    output: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("id", ""))
        gold = clean_title(row.get("target", ""))
        if not sample_id or not gold:
            continue
        split = deterministic_split(sample_id, fraction)
        if experiment == "direct_sft":
            prompt = base_prompt(row, maximum_history, include_history=True)
            parent = None
        else:
            parent = clean_title(row.get("parent", ""))
            if not parent:
                raise ValueError(f"sample={sample_id} 缺少无 Seed Base Parent")
            prompt = editor_prompt(row, parent, maximum_history, "title")
        output.append(
            {
                "example_id": f"{sample_id}:{experiment}:title",
                "sample_id": sample_id,
                "user_id": str(row.get("user_id", "")),
                "task": "title",
                "experiment": experiment,
                "parent": parent,
                "prompt": prompt,
                "target": gold,
                "output": gold,
                "trace_text": "",
                "output_text": gold,
                "sample_weight": 1.0,
                "split": split,
                "student_prompt_sees_gold": False,
                "seed_parent_used": False,
            }
        )
        if experiment == "multitask_sft" and rationales:
            rationale = str(rationales.get(sample_id, "")).strip()
            if rationale:
                output.append(
                    {
                        "example_id": f"{sample_id}:{experiment}:rationale",
                        "sample_id": sample_id,
                        "user_id": str(row.get("user_id", "")),
                        "task": "rationale",
                        "experiment": experiment,
                        "parent": parent,
                        "prompt": editor_prompt(row, parent, maximum_history, "rationale"),
                        "target": rationale,
                        "output": rationale,
                        "trace_text": "",
                        "output_text": rationale,
                        "sample_weight": rationale_weight,
                        "split": split,
                        "student_prompt_sees_gold": False,
                        "seed_parent_used": False,
                    }
                )
    if smoke and output and not any(item["split"] == "validation" for item in output):
        validation_id = output[-1]["sample_id"]
        for item in output:
            if item["sample_id"] == validation_id:
                item["split"] = "validation"
    return output


def load_shared_parent_rows(
    shared_parent_pool_dir: str | Path, split: str
) -> list[dict[str, Any]]:
    """读取独立共享 Pool，并只暴露其中第一个 greedy Parent。"""

    directory = resolve_path(shared_parent_pool_dir)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if bool(manifest.get("seed_parent_used", False)):
            raise ValueError(f"共享 Parent Pool 含 Seed Parent，拒绝复用：{directory}")
        if manifest.get("history_used") is False:
            raise ValueError(
                f"当前实验需要带 History 的 shared Pool，但传入的是 no_history：{directory}"
            )
    path = directory / f"{split}_parent_pool.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"共享 Parent Pool 不存在：{path}")
    rows = read_jsonl(path)
    output: list[dict[str, Any]] = []
    for row in rows:
        pool = list(row.get("parent_pool", []))
        if not pool:
            raise ValueError(f"共享 Parent Pool 的 sample={row.get('id')} 为空")
        item = dict(row)
        item["parent"] = clean_title(pool[0].get("text", ""))
        item["parent_source"] = "shared_parent_pool_greedy"
        item["seed_parent_used"] = False
        output.append(item)
    return output


def save_dataset(
    experiment: str, data_dir: Path, examples: list[dict[str, Any]]
) -> dict[str, Any]:
    train = [item for item in examples if item["split"] == "train"]
    validation = [item for item in examples if item["split"] == "validation"]
    if not train or not validation:
        raise ValueError(
            f"train/validation 不能为空：train={len(train)} validation={len(validation)}"
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(data_dir / "all_sft.jsonl", examples)
    write_jsonl(data_dir / "train_sft.jsonl", train)
    write_jsonl(data_dir / "validation_sft.jsonl", validation)
    title_examples = sum(item["task"] == "title" for item in examples)
    rationale_examples = sum(item["task"] == "rationale" for item in examples)
    report = {
        "protocol": f"noseed_{experiment}_v1",
        "experiment": experiment,
        "source_stage": "02_retrieved",
        "seed_parent_used": False,
        "student_prompt_sees_gold": False,
        "gold_used_as_title_output": True,
        "title_examples": title_examples,
        "rationale_examples": rationale_examples,
        "train_examples": len(train),
        "validation_examples": len(validation),
    }
    write_json(data_dir / "manifest.json", report)
    print(f"SFT dataset -> {data_dir}; report={report}", flush=True)
    return report


def train_adapter(
    config: dict[str, Any], data_dir: Path, run_dir: Path, max_steps: int | None
) -> dict[str, Any]:
    local = copy.deepcopy(config)
    local.setdefault("paths", {})["sft_dir"] = str(data_dir)
    local["paths"]["editor_output_dir"] = str(run_dir / "editor")
    trainer = import_module("06_train_editor_lora.py", "noseed_sft_trainer")
    return trainer.train(local, max_steps_override=max_steps)


def generate_predictions(
    experiment: str,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    run_dir: Path,
    adapter: Path | None,
    parent_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    maximum_history = int(config["sft_data"]["maximum_history_records"])
    prompts: list[str] = []
    parents: dict[str, str] = {}
    if parent_rows is not None:
        parents = {str(item["id"]): clean_title(item["parent"]) for item in parent_rows}
    for row in rows:
        if experiment in {"base", "direct_sft"}:
            prompts.append(base_prompt(row, maximum_history, include_history=True))
        else:
            parent = parents.get(str(row["id"]), "")
            if not parent:
                raise ValueError(f"测试 sample={row['id']} 缺少无 Seed Parent")
            prompts.append(editor_prompt(row, parent, maximum_history, "title"))

    if adapter is None:
        sampler = BaseParentSampler(config)
        try:
            predictions = [
                clean_title(
                    sampler.generate(
                        prompt,
                        1,
                        sample=False,
                        temperature=float(config["crossover_sft"]["temperature"]),
                        top_p=float(config["crossover_sft"]["top_p"]),
                        seed=int(config["training"]["seed"]) + index,
                    )[0]["text"]
                )
                for index, prompt in enumerate(prompts, 1)
            ]
        finally:
            sampler.close()
    else:
        generator = AdapterGenerator(config, adapter)
        try:
            predictions = generator.generate(
                prompts, int(config["evaluation"]["prediction_batch_size"])
            )
        finally:
            generator.close()
    prediction_rows = []
    for row, prediction in zip(rows, predictions):
        prediction_rows.append(
            {
                "id": str(row["id"]),
                "user_id": str(row.get("user_id", "")),
                "source_text": str(row.get("source_text", "")),
                "target": clean_title(row.get("target", "")),
                "parent": parents.get(str(row["id"])),
                "prediction": prediction,
                "error": None if prediction else "empty_prediction",
                "seed_parent_used": False,
            }
        )
    output_dir = run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "test_predictions.jsonl", prediction_rows)
    report = evaluate_rows(prediction_rows)
    report.update(
        {
            "experiment": experiment,
            "seed_parent_used": False,
            "input": (
                "query+top8_history"
                if experiment in {"base", "direct_sft"}
                else "query+top8_history+base_direct_parent"
            ),
        }
    )
    write_json(output_dir / "global_test_report.json", report)
    print(f"Evaluation -> {output_dir}; report={report}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="无 Seed SFT 输入消融")
    parser.add_argument(
        "--experiment",
        choices=("base", "direct_sft", "editor_sft", "multitask_sft"),
        required=True,
    )
    parser.add_argument(
        "--stage", choices=("parents", "data", "train", "eval", "all"), default="all"
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT / "config_global_llama2_7b_visgpt_prime_matched.yaml"),
    )
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--teacher-mode", choices=("api", "heuristic"), default="api")
    parser.add_argument(
        "--shared-parent-pool-dir",
        default="",
        help="独立 build_shared_parent_pool.py 的目录；Editor/Multitask 复用 greedy Parent",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = configure(load_config(args.config))
    config.setdefault("rationale_sft", {}).update(
        {"concurrency": 2, "schema_retries": 2, "rationale_loss_weight": 0.1}
    )
    stamp = args.run_name or (
        datetime.now().strftime("%Y%m%d_%H%M%S") + f"_noseed_{args.experiment}"
    )
    data_dir = resolve_path(
        args.data_dir or f"/data/liux/MEVO_global_cot/dataset/editor_sets/{stamp}"
    )
    run_dir = resolve_path(
        args.run_dir or f"/data/liux/MEVO_global_cot/result/{stamp}"
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"NOSEED_SFT_PATHS experiment={args.experiment} data={data_dir} run={run_dir}",
        flush=True,
    )
    result: dict[str, Any] = {}
    needs_parent = args.experiment in {"editor_sft", "multitask_sft"}
    parent_file = data_dir / "01_train_base_direct_parents.jsonl"

    if args.experiment == "base":
        if args.stage not in {"eval", "all"}:
            raise ValueError("base 实验只支持 --stage eval/all")
    else:
        train_rows = load_source_rows(config, "", "train")
        if args.limit > 0:
            train_rows = train_rows[: args.limit]
        if args.stage in {"parents", "all"} and needs_parent:
            if args.shared_parent_pool_dir:
                shared = load_shared_parent_rows(args.shared_parent_pool_dir, "train")
                if args.limit > 0:
                    shared = shared[: args.limit]
                write_jsonl(parent_file, shared)
                result["parents"] = {"rows": len(shared), "source": "shared_parent_pool"}
            else:
                result["parents"] = {
                    "rows": len(
                        build_one_parent_pool(train_rows, config, parent_file, args.limit)
                    ),
                    "source": "local_single_parent_pool",
                }
        if args.stage in {"data", "all"}:
            source_rows = train_rows
            if needs_parent:
                if not parent_file.exists():
                    if args.shared_parent_pool_dir:
                        shared = load_shared_parent_rows(args.shared_parent_pool_dir, "train")
                        if args.limit > 0:
                            shared = shared[: args.limit]
                        write_jsonl(parent_file, shared)
                    else:
                        raise FileNotFoundError(f"先运行 --stage parents：{parent_file}")
                source_rows = read_jsonl(parent_file)
            rationales = None
            if args.experiment == "multitask_sft":
                rationales = annotate_rationales(
                    source_rows,
                    config,
                    data_dir / "02_teacher_rationales.jsonl",
                    args.teacher_mode,
                )
            examples = compile_examples(
                args.experiment, source_rows, config, rationales, args.smoke
            )
            result["data"] = save_dataset(args.experiment, data_dir, examples)
        if args.stage in {"train", "all"}:
            result["train"] = train_adapter(
                config,
                data_dir,
                run_dir,
                args.max_steps if args.max_steps > 0 else None,
            )

    if args.stage in {"eval", "all"}:
        test_rows = load_source_rows(config, "", "test")
        if args.test_limit > 0:
            test_rows = test_rows[: args.test_limit]
        verify_formal_test(test_rows, config, args.test_limit, args.smoke)
        test_parent_rows = None
        if needs_parent:
            if args.shared_parent_pool_dir:
                test_parent_rows = load_shared_parent_rows(args.shared_parent_pool_dir, "test")
            else:
                test_parent_rows = build_one_parent_pool(
                    test_rows,
                    config,
                    data_dir / "test_base_direct_parents.jsonl",
                    0,
                )
        adapter = None
        if args.experiment != "base":
            adapter = run_dir / "editor" / "final_adapter"
            if not adapter.exists():
                raise FileNotFoundError(f"缺少 Adapter：{adapter}")
        result["eval"] = generate_predictions(
            args.experiment,
            test_rows,
            config,
            run_dir,
            adapter,
            test_parent_rows,
        )
    write_json(run_dir / "pipeline_report.json", result)
    print(f"NOSEED_SFT_DONE={result}", flush=True)


if __name__ == "__main__":
    main()
