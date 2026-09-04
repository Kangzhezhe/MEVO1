#!/usr/bin/env python3
"""无 Teacher 的多 Parent Crossover（候选聚合）SFT。

正式协议：

1. Base Llama2-7B 为每个 Query 生成 1 个 greedy Parent 和 4 个采样 Parent；
2. 只清理空值、重复、乱码或明显不是标题的输出，不进行语义筛选；
3. Student 输入 Query、Top-8 History 和候选集合，直接监督为 Gold 标题；
4. 训练候选随机排序，并额外构造一个 candidate-dropout 样本；
5. 测试仍生成同分布的 5 个 Parent，并在标准 100 用户/608 Query 上评估。

Teacher 不在数据构建、训练或推理链路中出现。旧的二元 Pair/Teacher gate
脚本仅保留为历史消融，不能与本协议混用。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CODE = HERE.parent
PROJECT = CODE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(PROJECT))

from pipeline_common import (  # noqa: E402
    deterministic_split,
    load_config,
    read_jsonl,
    resolve_path,
    visible_history,
    write_json,
    write_jsonl,
)
from run_crossover_sft import (  # noqa: E402
    AdapterGenerator,
    BaseParentSampler,
    _base_prompt,
    _load_module,
    clean_title,
    evaluate_prediction_rows,
    load_source_rows,
    normalized,
)


DEFAULT_PARENT_RECORDS = (
    "/data/liux/MEVO_global_cot/dataset/editor_sets/"
    "20260902_094902_direct_parent_gold_full/all_sft.jsonl"
)

INVALID_PARENT = re.compile(
    r"^(?:\[?paper title\]?|user title example|example input|abstract|title|output)\s*:?$",
    re.IGNORECASE,
)
INVALID_PARENT_LEAK = re.compile(
    r"(?:write one concise|example input\s*:|example output|user title example|"
    r"parent title\s*:|abstract\s*:|output only|```)",
    re.IGNORECASE,
)


def invalid_parent_title(title: str) -> bool:
    """仅识别明显格式/Prompt 污染；不判断标题语义是否优质。"""

    return bool(
        INVALID_PARENT.match(title)
        or INVALID_PARENT_LEAK.search(title)
        or title.lstrip().startswith(("•", "- ", "* "))
    )


def _stable_rng(sample_id: str, variant: str, seed: int) -> random.Random:
    """为候选排序与 dropout 提供跨进程可复现的随机数。"""

    digest = hashlib.sha256(f"{seed}:{sample_id}:{variant}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def usable_parents(row: dict[str, Any], maximum: int = 5) -> list[dict[str, Any]]:
    """执行最低限度的数据卫生检查并返回最多 ``maximum`` 个 Parent。"""

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(row.get("parent_pool", [])):
        title = clean_title(raw.get("text", ""))
        key = normalized(title)
        words = title.split()
        if (
            not key
            or key in seen
            or invalid_parent_title(title)
            or not 2 <= len(words) <= 40
        ):
            continue
        seen.add(key)
        output.append(
            {
                "candidate_id": str(
                    raw.get("candidate_id", f"{row.get('id')}:parent_{index}")
                ),
                "text": title,
                "source": str(raw.get("source", "base_model")),
            }
        )
        if len(output) >= maximum:
            break
    return output


def _append_parent(
    pool: list[dict[str, Any]], value: dict[str, Any], source: str
) -> None:
    """只做格式卫生与精确去重，不判断语义质量或与 Gold 的关系。"""

    title = clean_title(value.get("text", ""))
    key = normalized(title)
    if (
        not key
        or invalid_parent_title(title)
        or not 2 <= len(title.split()) <= 40
        or any(normalized(item["text"]) == key for item in pool)
    ):
        return
    raw_score = value.get("mean_logprob")
    mean_logprob = (
        float(raw_score)
        if raw_score is not None and math.isfinite(float(raw_score))
        else None
    )
    pool.append(
        {
            "candidate_id": "",
            "text": title,
            "source": source,
            # 非有限 log-prob 只表示旧 Transformers 无法可靠统计该序列，
            # 不能据此删除一个格式正常的候选。
            "mean_logprob": mean_logprob,
        }
    )


def build_simple_parent_pools(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    destination: Path,
    *,
    pool_source: str,
    sample_parents: int,
    limit: int,
) -> list[dict[str, Any]]:
    """构造 1 greedy + N sampling Pool，不做 best-of 或语义筛选。"""

    if limit > 0:
        rows = rows[:limit]
    cached_rows = read_jsonl(destination) if destination.exists() else []
    cached = {str(item.get("id", "")): item for item in cached_rows}
    protocol = (
        f"base_model_greedy1_sample{sample_parents}_teacher_free_v4"
        if pool_source == "base_model"
        else "existing_candidates_teacher_free_smoke_v4"
    )
    settings = config.get("crossover_sft", {})
    temperature = float(settings.get("temperature", 1.1))
    top_p = float(settings.get("top_p", 0.95))
    checkpoint_every = int(settings.get("checkpoint_every", 20))
    maximum_history = int(config.get("sft_data", {}).get("maximum_history_records", 8))
    seed = int(config.get("training", {}).get("seed", 42))
    sampler: BaseParentSampler | None = None
    output: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(rows, 1):
            sample_id = str(row.get("id", row.get("sample_id", "")))
            previous = cached.get(sample_id)
            if previous and previous.get("parent_pool_protocol") == protocol:
                output.append(previous)
                print(
                    f"parent pool {index}/{len(rows)} sample={sample_id} source=cache "
                    f"parents={len(previous.get('parent_pool', []))}",
                    flush=True,
                )
                continue
            pool: list[dict[str, Any]] = []
            if pool_source == "existing_candidates":
                _append_parent(
                    pool,
                    {"text": row.get("parent", row.get("prediction", ""))},
                    "cached_parent",
                )
                for candidate in row.get("candidates", [])[:sample_parents]:
                    _append_parent(pool, candidate, "existing_candidate_smoke")
            elif pool_source == "base_model":
                if sampler is None:
                    sampler = BaseParentSampler(config)
                prompt = _base_prompt(row, maximum_history)
                greedy = sampler.generate(
                    prompt,
                    1,
                    sample=False,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed + index,
                )
                for value in greedy:
                    _append_parent(pool, value, "base_greedy")
                # 这里只调用一次固定 N 路采样。没有超额生成、打分排名或选 B；
                # 精确重复被删除后，Student 接收的候选数允许少于 1+N。
                sampled = sampler.generate(
                    prompt,
                    sample_parents,
                    sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed * 100003 + index,
                )
                for value in sampled:
                    _append_parent(pool, value, "base_sample")
            else:
                raise ValueError(f"未知 pool_source={pool_source}")
            for parent_index, parent in enumerate(pool):
                parent["candidate_id"] = f"{sample_id}:base_parent_{parent_index}"
            item = dict(row)
            item["id"] = sample_id
            item["parent_pool"] = pool
            item["parent_pool_protocol"] = protocol
            item["requested_parent_count"] = 1 + sample_parents
            output.append(item)
            print(
                f"parent pool {index}/{len(rows)} sample={sample_id} "
                f"source={pool_source} requested={1 + sample_parents} parents={len(pool)}",
                flush=True,
            )
            if checkpoint_every > 0 and index % checkpoint_every == 0:
                write_jsonl(destination, output)
        write_jsonl(destination, output)
        return output
    finally:
        if sampler is not None:
            sampler.close()


def multi_parent_prompt(
    row: dict[str, Any], parents: list[dict[str, Any]], maximum_history: int
) -> str:
    """构造纯文本输出协议；候选集合没有固定的 A/B 主次关系。"""

    payload = {
        "current_input": str(row.get("source_text", "")),
        "retrieved_history": visible_history(row, maximum_history),
        "parent_candidates": [
            {"candidate_id": item["candidate_id"], "text": item["text"]}
            for item in parents
        ],
    }
    return (
        "You are a personalized response aggregation editor. Produce the best final "
        "academic paper title supported by CURRENT_INPUT. Compare all PARENT_CANDIDATES: "
        "you may preserve a strong candidate, repair one candidate, or combine compatible "
        "useful content from several candidates. Use RETRIEVED_HISTORY only as guidance "
        "for applicable recurring user preferences. Candidate order does not indicate "
        "quality. Do not invent facts. Return exactly one line containing only the final "
        "title; do not output JSON, Markdown, labels, or explanations.\n\nPAYLOAD:\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n\nFINAL TITLE:\n"
    )


def _ordered_subset(
    parents: list[dict[str, Any]], sample_id: str, variant: str, seed: int
) -> list[dict[str, Any]]:
    values = list(parents)
    _stable_rng(sample_id, variant, seed).shuffle(values)
    return values


def build_examples(
    pool_rows: list[dict[str, Any]], config: dict[str, Any], smoke: bool
) -> list[dict[str, Any]]:
    """每个训练 Query 构造完整候选和一个 dropout 候选集合。"""

    settings = config.get("crossover_sft", {})
    maximum_history = int(config.get("sft_data", {}).get("maximum_history_records", 8))
    fraction = float(config.get("sft_data", {}).get("validation_fraction", 0.05))
    seed = int(config.get("training", {}).get("seed", 42))
    maximum_parents = int(settings.get("total_parent_count", 5))
    dropout_copies = max(0, int(settings.get("candidate_dropout_copies", 1)))
    examples: list[dict[str, Any]] = []

    for row in pool_rows:
        sample_id = str(row.get("id", row.get("sample_id", "")))
        gold = clean_title(row.get("target", ""))
        parents = usable_parents(row, maximum_parents)
        if not sample_id or not gold or len(parents) < 2:
            continue
        split = deterministic_split(sample_id, fraction)
        variants: list[tuple[str, list[dict[str, Any]]]] = [
            ("full", _ordered_subset(parents, sample_id, "full", seed))
        ]
        # Validation 只保留完整候选集合；dropout 是训练增强，不应改变验证分布。
        if split == "train" and len(parents) >= 3:
            for copy_index in range(dropout_copies):
                variant = f"dropout_{copy_index}"
                rng = _stable_rng(sample_id, variant, seed)
                keep_count = rng.randint(2, len(parents) - 1)
                selected = rng.sample(parents, keep_count)
                rng.shuffle(selected)
                variants.append((variant, selected))

        for variant, selected in variants:
            examples.append(
                {
                    "example_id": f"{sample_id}:crossover:{variant}",
                    "sample_id": sample_id,
                    "user_id": str(row.get("user_id", "")),
                    "task": "multi_parent_crossover_title",
                    "operation_type": "crossover_aggregation",
                    "variant": variant,
                    "parent_count": len(selected),
                    "parent_ids": [item["candidate_id"] for item in selected],
                    "parents": selected,
                    "prompt": multi_parent_prompt(row, selected, maximum_history),
                    "target": gold,
                    "output": gold,
                    "trace_text": "",
                    "output_text": gold,
                    "sample_weight": 1.0,
                    "split": split,
                    "student_prompt_sees_gold": False,
                    "teacher_used": False,
                }
            )

    if smoke and examples and not any(item["split"] == "validation" for item in examples):
        # smoke 小集合可能没有命中 hash 验证桶；移动整个 Query，避免同 Query 泄漏。
        validation_id = examples[-1]["sample_id"]
        for item in examples:
            if item["sample_id"] == validation_id:
                item["split"] = "validation"
    return examples


def write_dataset(
    output_dir: Path,
    pool_rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    *,
    pool_source: str,
    smoke: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = [item for item in examples if item["split"] == "train"]
    validation = [item for item in examples if item["split"] == "validation"]
    if not train or not validation:
        raise ValueError(
            f"Crossover SFT train/validation 不能为空：train={len(train)} "
            f"validation={len(validation)}"
        )
    write_jsonl(output_dir / "01_parent_pool.jsonl", pool_rows)
    write_jsonl(output_dir / "02_aggregation_examples.jsonl", examples)
    write_jsonl(output_dir / "all_sft.jsonl", examples)
    write_jsonl(output_dir / "train_sft.jsonl", train)
    write_jsonl(output_dir / "validation_sft.jsonl", validation)

    full = [item for item in examples if item["variant"] == "full"]
    report = {
        "protocol": "teacher_free_multi_parent_crossover_sft_v4",
        "source_queries": len(pool_rows),
        "queries_with_at_least_two_parents": len(full),
        "query_coverage": len(full) / max(len(pool_rows), 1),
        "configured_parents": "1 greedy + 4 sampling",
        "full_parent_count_distribution": {
            str(count): sum(item["parent_count"] == count for item in full)
            for count in range(2, 6)
        },
        "examples": len(examples),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "candidate_dropout_examples": sum(
            item["variant"].startswith("dropout") for item in examples
        ),
        "pool_source": pool_source,
        "teacher_used": False,
        "semantic_gate_used": False,
        "gold_used_only_as_sft_target": True,
        "student_prompt_sees_gold": False,
        "single_task_plain_title": True,
        "formal_result_eligible": pool_source == "base_model" and not smoke,
    }
    write_json(output_dir / "manifest.json", report)
    print(f"Multi-parent Crossover dataset -> {output_dir}; report={report}", flush=True)
    return report


def train_adapter(
    config: dict[str, Any], data_dir: Path, editor_dir: Path, max_steps: int | None
) -> dict[str, Any]:
    local = copy.deepcopy(config)
    local.setdefault("paths", {})["sft_dir"] = str(data_dir)
    local["paths"]["editor_output_dir"] = str(editor_dir)
    local.setdefault("sft_data", {})["supervision_mode"] = "plain_output_only"
    local.setdefault("training", {})["trace_loss_weight"] = 0.0
    local["training"]["output_loss_weight"] = 1.0
    local["training"]["initial_adapter_path"] = ""
    trainer = _load_module("06_train_editor_lora.py", "multi_parent_crossover_trainer")
    return trainer.train(local, max_steps_override=max_steps)


def evaluate(
    config: dict[str, Any], pool_rows: list[dict[str, Any]], adapter: Path, output_dir: Path
) -> dict[str, Any]:
    settings = config.get("crossover_sft", {})
    maximum_history = int(config.get("sft_data", {}).get("maximum_history_records", 8))
    maximum_parents = int(settings.get("total_parent_count", 5))
    seed = int(config.get("training", {}).get("seed", 42))
    specs: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    prompts: list[str] = []
    for row in pool_rows:
        sample_id = str(row.get("id", ""))
        parents = usable_parents(row, maximum_parents)
        if len(parents) < 2:
            continue
        parents = _ordered_subset(parents, sample_id, "test_full", seed)
        specs.append((row, parents))
        prompts.append(multi_parent_prompt(row, parents, maximum_history))

    generator = AdapterGenerator(config, adapter)
    try:
        predictions = generator.generate(
            prompts, int(config.get("evaluation", {}).get("prediction_batch_size", 1))
        )
    finally:
        generator.close()
    by_id = {
        str(row.get("id", "")): (parents, prediction)
        for (row, parents), prediction in zip(specs, predictions)
    }
    rows: list[dict[str, Any]] = []
    for source in pool_rows:
        sample_id = str(source.get("id", ""))
        generated = by_id.get(sample_id)
        if generated is None:
            parents = usable_parents(source, maximum_parents)
            prediction = parents[0]["text"] if parents else ""
            error = "insufficient_parents_fallback_greedy"
        else:
            parents, prediction = generated
            error = None if prediction else "empty_prediction"
        rows.append(
            {
                "id": sample_id,
                "user_id": str(source.get("user_id", "")),
                "source_text": str(source.get("source_text", "")),
                "target": clean_title(source.get("target", "")),
                "parents": parents,
                "parent_count": len(parents),
                "prediction": prediction,
                "error": error,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "test_predictions.jsonl", rows)
    report = evaluate_prediction_rows(rows)
    report.update(
        {
            "protocol": "multi_parent_crossover_single_output_v4",
            "aggregation_queries": len(specs),
            "aggregation_coverage": len(specs) / max(len(pool_rows), 1),
            "insufficient_parent_fallbacks": len(pool_rows) - len(specs),
            "teacher_used": False,
        }
    )
    write_json(output_dir / "global_test_report.json", report)
    print(f"Multi-parent Crossover evaluation -> {output_dir}; report={report}", flush=True)
    return report


def _pool(
    args: argparse.Namespace, config: dict[str, Any], split: str, destination: Path
) -> list[dict[str, Any]]:
    source = args.parent_records if split == "train" else args.test_parent_records
    rows = load_source_rows(config, source, split)
    limit = args.limit if split == "train" else args.test_limit
    return build_simple_parent_pools(
        rows,
        config,
        destination,
        pool_source=args.pool_source,
        sample_parents=args.sample_parents,
        limit=limit,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="无 Teacher 的多 Parent Crossover SFT")
    parser.add_argument("--config", default=str(HERE / "config_crossover_sft.yaml"))
    parser.add_argument(
        "--stage", choices=("pool", "build", "train", "eval", "all"), default="all"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--parent-records", default=DEFAULT_PARENT_RECORDS)
    parser.add_argument("--test-parent-records", default="")
    parser.add_argument(
        "--pool-source", choices=("base_model", "existing_candidates"), default="base_model"
    )
    parser.add_argument("--sample-parents", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    settings = config.setdefault("crossover_sft", {})
    if args.sample_parents <= 0:
        args.sample_parents = int(settings.get("sample_parent_count", 4))
    if args.sample_parents != 4 and not args.smoke:
        raise ValueError("正式协议固定为 1 greedy + 4 sampling；sample-parents 必须为 4")
    data_dir = resolve_path(args.data_dir)
    run_dir = resolve_path(args.run_dir)
    result: dict[str, Any] = {}

    if args.stage in {"pool", "all"}:
        pool_rows = _pool(args, config, "train", data_dir / "01_parent_pool.jsonl")
        result["pool"] = {
            "queries": len(pool_rows),
            "output": str(data_dir / "01_parent_pool.jsonl"),
        }
    if args.stage in {"build", "all"}:
        pool_path = data_dir / "01_parent_pool.jsonl"
        if not pool_path.exists():
            raise FileNotFoundError(f"请先运行 --stage pool：{pool_path}")
        pool_rows = read_jsonl(pool_path)
        if args.limit > 0:
            pool_rows = pool_rows[: args.limit]
        examples = build_examples(pool_rows, config, args.smoke)
        result["build"] = write_dataset(
            data_dir, pool_rows, examples, pool_source=args.pool_source, smoke=args.smoke
        )
    if args.stage in {"train", "all"}:
        result["train"] = train_adapter(
            config,
            data_dir,
            run_dir / "editor",
            args.max_steps if args.max_steps > 0 else None,
        )
    if args.stage in {"eval", "all"}:
        test_pool = _pool(
            args, config, "test", data_dir / "test_parent_pool.jsonl"
        )
        adapter = run_dir / "editor" / "final_adapter"
        if not adapter.exists():
            raise FileNotFoundError(f"缺少 Crossover Adapter：{adapter}")
        result["eval"] = evaluate(config, test_pool, adapter, run_dir / "crossover")
    print(f"MULTI_PARENT_CROSSOVER_DONE={result}", flush=True)


if __name__ == "__main__":
    main()
