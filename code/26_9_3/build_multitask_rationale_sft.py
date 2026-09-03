#!/usr/bin/env python3
"""构造 Distilling Step-by-Step 风格的 Title/Rationale 多任务 SFT 数据。

该脚本建立在真实 Base Parent 之上：

    Query + History + Base Parent
             ↓
        ┌────┴────┐
        │         │
     [TITLE]   [RATIONALE]
       ↓           ↓
     Gold      短编辑解释

Teacher 只在离线标注 rationale 时看到 Gold；Student 的两个 Prompt 都不包含 Gold。
Title 和 Rationale 是两个独立样本，而不是 ``rationale + title`` 的一个长输出。
因此可以直接复用现有 ``06_train_editor_lora.py``：每行的 ``output_text`` 是该
任务自己的目标，``sample_weight`` 控制 rationale 辅助损失的相对权重。

示例：

    python code/26_9_1/build_multitask_rationale_sft.py \
      --config config_global_llama2_7b_visgpt_prime_matched.yaml \
      --parent-records /data/liux/MEVO_global_cot/dataset/editor_sets/\
        base_parent_teacher_test8_pilot/01_base_parent_records.jsonl \
      --limit 8 --output /data/liux/MEVO_global_cot/dataset/editor_sets/\
        multitask_rationale_test8_pilot

脚本只负责数据构造，不启动模型 SFT。生成的 ``train_sft.jsonl`` 和
``validation_sft.jsonl`` 可以被 ``06_train_editor_lora.py`` 读取。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "code"))

from common.concurrency import BoundedJobError, run_bounded  # noqa: E402
from pipeline_common import (  # noqa: E402
    build_base_text_prompt,
    deterministic_split,
    load_config,
    read_jsonl,
    resolve_path,
    stage_path,
    teacher_client,
    visible_history,
    write_json,
    write_jsonl,
)


# Teacher 偶尔会在解释中写 “gold standard”。这属于元话语，但不等于泄漏
# Gold 内容；规范化时将其替换为 neutral 文本。真正的评估指标元信息仍拒绝。
META_RATIONALE = re.compile(
    r"\b(?:gold standard|ground[ -]?truth|reference answer|target title|gold)\b",
    re.IGNORECASE,
)
FORBIDDEN_RATIONALE = re.compile(r"\b(?:rouge|metric|evaluation score)\b", re.IGNORECASE)


def clean_title(value: Any) -> str:
    """保持与 Direct Parent→Gold 脚本相同的标题清理协议。"""

    text = str(value or "").strip()
    text = re.sub(r"^```(?:text|title)?\s*|\s*```$", "", text, flags=re.I).strip()
    text = re.sub(r"^(?:title|output)\s*:\s*", "", text, flags=re.I).strip()
    for line in text.splitlines():
        line = line.strip().strip('"').strip()
        if line:
            return line[:300]
    return ""


def compact_history(row: dict[str, Any], maximum: int = 8) -> list[dict[str, str]]:
    """只保留 Teacher/Student 可见的 Top-k 历史字段。"""

    return [
        {
            "id": str(item.get("id", "")),
            "input": str(item.get("input", ""))[:500],
            "output": str(item.get("output", ""))[:300],
        }
        for item in visible_history(row, maximum)
    ]


def rationale_prompt(row: dict[str, Any], parent: str, maximum_history: int) -> str:
    """Student 的 RATIONALE Prompt；绝不包含 Gold。"""

    payload = {
        "current_input": str(row.get("source_text", "")),
        "retrieved_history": compact_history(row, maximum_history),
        "parent": parent,
    }
    return (
        "[RATIONALE]\n"
        "Explain the editing decision for the parent title in one or two concise sentences. "
        "State what the parent misses or overemphasizes, whether a visible history example "
        "supports a relevant user-specific editing behavior, and the resulting edit direction. "
        "If history does not support a reliable preference, explicitly say that no reliable "
        "user-specific signal is available. Use only visible history IDs. Do not output the final "
        "title, JSON, Markdown, a reference answer, or evaluation language.\n\n"
        "PAYLOAD:\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n\nRATIONALE:\n"
    )


def teacher_prompt(row: dict[str, Any], parent: str, maximum_history: int) -> str:
    """Gold-aware Teacher Prompt；Gold 只用于离线生成 rationale。"""

    payload = {
        "current_input": str(row.get("source_text", "")),
        "retrieved_history": compact_history(row, maximum_history),
        "parent": parent,
        "gold": clean_title(row.get("target", "")),
    }
    return (
        "You are an offline annotation teacher for a personalized text editor. "
        "Use GOLD only to identify the necessary correction from PARENT. Produce one concise "
        "natural-language edit rationale, not a hidden chain of thought and not the final title. "
        "The rationale must explain the concrete Parent issue, optionally cite a visible history "
        "ID when it supports a recurring user-specific behavior, and state the edit direction. "
        "Do not force a history signal when none is supported. Do not copy GOLD verbatim. "
        "Return exactly one JSON object with this schema: "
        '{"rationale":"one or two short sentences",'
        '"evidence_ids":["visible_id"],'
        '"quality":"history_supported|task_only"}'
        "\n\nPAYLOAD:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def normalize_teacher(
    value: Any, row: dict[str, Any], parent: str, maximum_history: int
) -> dict[str, Any]:
    """严格规范 Teacher 返回，但不要求每条样本都有历史证据。"""

    if not isinstance(value, dict):
        raise ValueError("rationale Teacher 返回的不是 JSON object")
    rationale = str(value.get("rationale", "")).strip()
    rationale = re.sub(r"\s+", " ", rationale)
    if not rationale:
        raise ValueError("rationale 为空")
    if len(rationale) > 600:
        raise ValueError("rationale 超过 600 字符")
    if FORBIDDEN_RATIONALE.search(rationale):
        raise ValueError("rationale 包含 Gold/评估元信息")
    rationale = META_RATIONALE.sub("the desired title", rationale)
    rationale = re.sub(r"\bthe the desired\b", "the desired", rationale, flags=re.I)
    gold = clean_title(row.get("target", ""))
    # 禁止完整复制 Gold；允许 rationale 提及摘要中与编辑有关的少数术语。
    def norm(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.casefold())

    if norm(gold) and norm(gold) in norm(rationale):
        raise ValueError("rationale 泄漏了完整 Gold 标题")
    visible_ids = {
        str(item.get("id", "")) for item in visible_history(row, maximum_history)
    }
    evidence = value.get("evidence_ids", [])
    if not isinstance(evidence, list):
        evidence = []
    evidence = list(dict.fromkeys(str(item) for item in evidence))
    unknown = set(evidence) - visible_ids
    if unknown:
        raise ValueError(f"rationale 引用了不可见历史: {sorted(unknown)}")
    # 只保留最多两条证据，避免 Teacher 将整个 Top-8 都复制进监督目标。
    evidence = evidence[:2]
    quality = str(value.get("quality", "task_only")).strip().lower()
    if quality not in {"history_supported", "task_only"}:
        quality = "history_supported" if evidence else "task_only"
    if quality == "history_supported" and not evidence:
        raise ValueError("history_supported 必须有 evidence_ids")
    if quality == "task_only" and evidence:
        # 保守地保留证据，并将标签归为 history_supported，避免元数据与文本矛盾。
        quality = "history_supported"
    return {
        "rationale": rationale,
        "evidence_ids": evidence,
        "quality": quality,
        "parent": parent,
        "gold": gold,
    }


def load_rows(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, Any]]:
    if args.parent_records:
        rows = read_jsonl(resolve_path(args.parent_records))
        # 兼容旧版 Direct 构建产物：all_sft.jsonl 只有 sample_id/parent，
        # 而 Teacher rationale 还需要 source_text/profile。用同一 split 的
        # retrieve 文件补回原始字段，避免重新运行昂贵的 Base Parent 生成。
        if rows and not rows[0].get("source_text"):
            retrieved = read_jsonl(stage_path(config, args.split, "retrieve"))
            by_id = {str(item.get("id", "")): item for item in retrieved}
            merged = []
            for item in rows:
                sample_id = str(item.get("id", item.get("sample_id", "")))
                source = by_id.get(sample_id)
                if source is None:
                    raise ValueError(f"无法从 retrieve 补回 sample={sample_id}")
                value = dict(source)
                value["parent"] = item.get("parent", "")
                value["parent_source"] = item.get("parent_source", "base_model")
                merged.append(value)
            rows = merged
    else:
        rows = read_jsonl(stage_path(config, args.split, "retrieve"))
        cached = None
        if args.parent_file:
            cached = {str(item["id"]): item for item in read_jsonl(resolve_path(args.parent_file))}
        # 复用 Parent 生成函数，不执行原脚本的 Teacher annotate 阶段。
        from build_base_parent_teacher_sft import generate_parents

        rows = generate_parents(rows, config, cached)
    if args.limit > 0:
        rows = rows[: args.limit]
    return rows


def annotate_rationales(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    client = teacher_client(config)
    maximum_history = int(config.get("sft_data", {}).get("maximum_history_records", 8))
    jobs = list(enumerate(rows))
    results: list[dict[str, Any] | None] = [None] * len(rows)
    retries = int(config.get("rationale_sft", {}).get("schema_retries", 2))

    def worker(job: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        index, row = job
        parent = clean_title(row.get("parent", ""))
        if not parent:
            return {
                "index": index,
                "sample_id": str(row.get("id", "")),
                "user_id": str(row.get("user_id", "")),
                "status": "failed",
                "error": "Parent 为空",
            }
        prompt = teacher_prompt(row, parent, maximum_history)
        last_error: Exception | None = None
        for attempt in range(max(1, retries + 1)):
            task = f"multitask_rationale_{row.get('id')}_attempt_{attempt}"
            try:
                payload, raw = client.json(task, prompt, {"sample_id": str(row.get("id"))})
                annotation = normalize_teacher(payload, row, parent, maximum_history)
                return {
                    "index": index,
                    "sample_id": str(row.get("id", "")),
                    "user_id": str(row.get("user_id", "")),
                    "annotation": annotation,
                    "teacher_raw_response": raw,
                }
            except Exception as error:  # API、JSON 或业务字段错误均重试
                last_error = error
                client.invalidate(task, prompt)
        # rationale 是辅助任务；单条失败不应丢掉对应的 Title 样本。失败
        # 记录会写入 teacher_rationales.jsonl，并在 manifest 中统计，便于
        # 后续只重试失败子集。
        return {
            "index": index,
            "sample_id": str(row.get("id", "")),
            "user_id": str(row.get("user_id", "")),
            "status": "failed",
            "error": str(last_error),
        }

    concurrency = int(
        config.get("rationale_sft", {}).get(
            "concurrency", config.get("generation", {}).get("concurrency", 1)
        )
    )

    def on_result(job: tuple[int, dict[str, Any]], result: dict[str, Any], completed: int) -> None:
        results[result["index"]] = result
        if result.get("status") == "failed":
            print(
                f"rationale progress {completed}/{len(rows)} sample={result['sample_id']} "
                f"status=failed error={result['error']}",
                flush=True,
            )
        else:
            ann = result["annotation"]
            print(
                f"rationale progress {completed}/{len(rows)} sample={result['sample_id']} "
                f"quality={ann['quality']} evidence={len(ann['evidence_ids'])}",
                flush=True,
            )

    try:
        run_bounded(
            jobs,
            worker,
            on_result,
            max_workers=max(1, concurrency),
            thread_name_prefix="multitask-rationale",
        )
    except BoundedJobError as error:
        raise RuntimeError(str(error)) from error.error
    return [item for item in results if item is not None]


def build_multitask_rows(
    rows: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {
        str(item["sample_id"]): item
        for item in annotations
        if item.get("status", "ok") != "failed"
    }
    maximum_history = int(config.get("sft_data", {}).get("maximum_history_records", 8))
    split_fraction = float(config.get("sft_data", {}).get("validation_fraction", 0.05))
    rationale_weight = float(config.get("rationale_sft", {}).get("rationale_loss_weight", 0.1))
    if rationale_weight <= 0:
        raise ValueError("rationale_loss_weight 必须大于 0")
    examples: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("id", ""))
        ann = by_id.get(sample_id)
        parent = clean_title(row.get("parent", ""))
        gold = clean_title(row.get("target", ""))
        parent_candidate = {
            "candidate_id": f"{sample_id}:base_model",
            "text": parent,
        }
        split = deterministic_split(sample_id, split_fraction)
        base_metadata = {
            "sample_id": sample_id,
            "user_id": str(row.get("user_id", "")),
            "source_text": str(row.get("source_text", "")),
            "profile": list(row.get("profile", [])),
            "parent": parent,
            "parent_source": str(row.get("parent_source", "base_model")),
            "student_prompt_sees_gold": False,
            # Title 样本完全不需要 Teacher；RATIONALE 样本在下面单独标记
            # teacher_sees_gold=True，避免把离线标注误写成主任务泄漏。
            "teacher_sees_gold": False,
            "split": split,
        }
        examples.append(
            {
                **base_metadata,
                "example_id": f"{sample_id}:base_parent:title",
                "task": "title",
                "prompt": "[TITLE]\n" + build_base_text_prompt(row, parent_candidate, maximum_history),
                "target": gold,
                "output": gold,
                "trace_text": "",
                "output_text": gold,
                "sample_weight": 1.0,
                "teacher_used": False,
            }
        )
        if ann is not None:
            examples.append(
                {
                    **base_metadata,
                    "teacher_sees_gold": True,
                    "example_id": f"{sample_id}:base_parent:rationale",
                    "task": "rationale",
                    "prompt": rationale_prompt(row, parent, maximum_history),
                    "target": str(ann["annotation"]["rationale"]),
                    "output": str(ann["annotation"]["rationale"]),
                    "trace_text": "",
                    "output_text": str(ann["annotation"]["rationale"]),
                    "sample_weight": rationale_weight,
                    "rationale_quality": ann["annotation"]["quality"],
                    "rationale_evidence_ids": ann["annotation"]["evidence_ids"],
                    "teacher_used": True,
                }
            )
    return examples, annotations


def write_rationale_analysis(output: Path, annotations: list[dict[str, Any]]) -> None:
    """写出可人工复核的 rationale 质量摘要。"""

    successful = [x for x in annotations if x.get("status", "ok") != "failed"]
    failed = [x for x in annotations if x.get("status") == "failed"]
    history = [x for x in successful if x["annotation"]["quality"] == "history_supported"]
    lines = [
        "# Rationale 辅助任务质量分析",
        "",
        f"- 总 Query：{len(annotations)}",
        f"- 成功 rationale：{len(successful)}",
        f"- 失败 rationale（对应 Title 仍保留）：{len(failed)}",
        f"- history_supported：{len(history)}",
        f"- task_only：{len(successful) - len(history)}",
        "",
        "## 成功样本",
        "",
        "| Sample | 类型 | 证据数 | Rationale |",
        "|---|---|---:|---|",
    ]
    for item in successful:
        ann = item["annotation"]
        rationale = ann["rationale"].replace("|", "\\|")
        lines.append(
            f"| {item['sample_id']} | {ann['quality']} | {len(ann['evidence_ids'])} | {rationale} |"
        )
    lines.extend(["", "## 失败样本", "", "| Sample | 原因 |", "|---|---|"])
    for item in failed:
        lines.append(f"| {item['sample_id']} | {item.get('error', '')} |")
    (output / "rationale_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="构造 Title/Rationale 多任务 SFT 数据")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--parent-records", default="", help="已有真实 Base Parent JSONL（也兼容旧版 all_sft.jsonl）"
    )
    parser.add_argument("--parent-file", default="", help="已有 Base predictions JSONL")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    rows = load_rows(args, config)
    if not rows:
        raise ValueError("没有可用的输入行")
    print(f"multi-task rationale annotation rows={len(rows)}", flush=True)
    annotations = annotate_rationales(rows, config)
    examples, annotations = build_multitask_rows(rows, annotations, config)
    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "teacher_rationales.jsonl", annotations)
    write_jsonl(output / "all_sft.jsonl", examples)
    write_jsonl(output / "train_sft.jsonl", [x for x in examples if x["split"] == "train"])
    write_jsonl(output / "validation_sft.jsonl", [x for x in examples if x["split"] == "validation"])
    write_rationale_analysis(output, annotations)
    rationale_annotations = [x["annotation"] for x in annotations if x.get("status", "ok") != "failed"]
    failed_annotations = [x for x in annotations if x.get("status") == "failed"]
    history_supported = sum(x["quality"] == "history_supported" for x in rationale_annotations)
    report = {
        "protocol": "distilling_step_by_step_multitask_title_rationale_v1",
        "split": args.split,
        "source_queries": len(rows),
        "title_examples": sum(x["task"] == "title" for x in examples),
        "rationale_examples": sum(x["task"] == "rationale" for x in examples),
        "examples": len(examples),
        "train_examples": sum(x["split"] == "train" for x in examples),
        "validation_examples": sum(x["split"] == "validation" for x in examples),
        "one_parent_per_query": True,
        "teacher_used_for_rationale_only": True,
        "teacher_sees_gold": True,
        "student_prompt_sees_gold": False,
        "title_inference_requires_rationale": False,
        "history_supported_rationales": history_supported,
        "task_only_rationales": len(rationale_annotations) - history_supported,
        "failed_rationale_annotations": len(failed_annotations),
        "rationale_loss_weight": float(config.get("rationale_sft", {}).get("rationale_loss_weight", 0.1)),
        "output_dir": str(output),
    }
    write_json(output / "manifest.json", report)
    print(f"Multi-task Title/Rationale SFT -> {output}; report={report}", flush=True)


if __name__ == "__main__":
    main()
