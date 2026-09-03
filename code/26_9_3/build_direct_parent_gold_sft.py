#!/usr/bin/env python3
"""构造纯文本 Parent→Gold Editor SFT 样本。

本脚本只复用 ``build_base_parent_teacher_sft.py`` 的真实 Base Parent 生成逻辑，
不调用 Teacher、不生成 Trace，也不把 Gold 写入 Prompt。每个 Query 默认生成一个
SFT 样本：

    Query + Top-8 History + Base Parent  ->  Gold 标题

脚本同时生成离线质量分析，便于在正式 SFT 前逐条检查 Parent 与 Gold 的距离。
质量分析可以使用 Gold，但分析字段不会写入 Student Prompt 或训练目标。

推荐用已有 Parent 缓存做 pilot：

    python code/26_9_1/build_direct_parent_gold_sft.py \
      --config config_direct_gold_sft_base_protocol.yaml \
      --split test --limit 8 \
      --parent-records /data/liux/MEVO_global_cot/dataset/editor_sets/\
        base_parent_teacher_test8_pilot/01_base_parent_records.jsonl \
      --output /data/liux/MEVO_global_cot/dataset/editor_sets/direct_parent_gold_test8_pilot

如果没有 ``--parent-records``，脚本会调用同目录下的
``build_base_parent_teacher_sft.generate_parents``；此时只生成 Parent，不会调用其
Teacher ``annotate`` 阶段。
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "code"))

from pipeline_common import (  # noqa: E402
    build_base_text_prompt,
    deterministic_split,
    load_config,
    read_jsonl,
    resolve_path,
    stage_path,
    visible_history,
    write_json,
    write_jsonl,
)


def clean_title(text: Any) -> str:
    """将 Parent/Gold 归一为单行文本；不做语义改写。"""

    value = str(text or "").strip()
    value = re.sub(r"^```(?:text|title)?\s*|\s*```$", "", value, flags=re.I).strip()
    value = re.sub(r"^(?:title|output)\s*:\s*", "", value, flags=re.I).strip()
    for line in value.splitlines():
        line = line.strip().strip('"').strip()
        if line:
            return line[:300]
    return ""


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).casefold())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).casefold())


def quality_analysis(parent: str, gold: str) -> dict[str, Any]:
    """离线诊断 Parent 质量；这些字段不进入 SFT 监督。"""

    parent_tokens = tokenize(parent)
    gold_tokens = tokenize(gold)
    pset, gset = set(parent_tokens), set(gold_tokens)
    union = pset | gset
    jaccard = len(pset & gset) / len(union) if union else 0.0
    sequence = difflib.SequenceMatcher(None, parent.casefold(), gold.casefold()).ratio()
    exact = bool(parent and gold and normalized(parent) == normalized(gold))
    if not parent:
        category = "invalid_empty"
    elif exact:
        category = "exact_or_punctuation"
    elif sequence >= 0.70:
        category = "near_repair"
    elif sequence >= 0.45:
        category = "moderate_rewrite"
    else:
        category = "large_rewrite"
    return {
        "parent_nonempty": bool(parent),
        "gold_nonempty": bool(gold),
        "normalized_exact": exact,
        "char_similarity": round(sequence, 4),
        "token_jaccard": round(jaccard, 4),
        "parent_chars": len(parent),
        "gold_chars": len(gold),
        "category": category,
    }


def load_parent_rows(
    args: argparse.Namespace, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """读取已有 Parent，或仅调用原脚本的 Parent 生成函数。"""

    if args.parent_records:
        rows = read_jsonl(resolve_path(args.parent_records))
        if args.limit > 0:
            rows = rows[: args.limit]
        return rows

    # Base 编辑协议需要一个 target-blind seed 作为初始 Parent。retrieve 阶段
    # 只有 Query/History，没有 candidates；若误读 retrieve，会让全部 Parent
    # 为空。必须读取已经生成 target-blind candidates 的 seeds 阶段。
    rows = read_jsonl(stage_path(config, args.split, "seeds"))
    if args.limit > 0:
        rows = rows[: args.limit]
    cached = None
    if args.parent_file:
        cached = {str(item["id"]): item for item in read_jsonl(resolve_path(args.parent_file))}

    # 只导入并调用 generate_parents；不会执行原脚本 main，也不会调用 annotate。
    from build_base_parent_teacher_sft import generate_parents  # noqa: WPS433

    print(f"复用 Base Parent 生成逻辑，rows={len(rows)}; teacher_annotation=disabled", flush=True)
    return generate_parents(rows, config, cached)


def build_examples(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    maximum_history = int(config.get("sft_data", {}).get("maximum_history_records", 8))
    fraction = float(config.get("sft_data", {}).get("validation_fraction", 0.05))
    examples: list[dict[str, Any]] = []
    analyses: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        parent = clean_title(row.get("parent", ""))
        gold = clean_title(row.get("target", ""))
        parent_candidate = {
            "candidate_id": f"{row.get('id')}:base_model",
            "text": parent,
        }
        # 纯文本协议：不能使用原脚本的 structured trace prompt。
        prompt = build_base_text_prompt(row, parent_candidate, maximum_history)
        quality = quality_analysis(parent, gold)
        if not parent:
            raise ValueError(
                f"sample={row.get('id')} 的 Base Parent 为空；禁止把空 Parent 写入正式 SFT"
            )
        analysis = {
            "sample_id": str(row.get("id", "")),
            "user_id": str(row.get("user_id", "")),
            "parent": parent,
            "gold": gold,
            **quality,
        }
        analyses.append(analysis)
        if not gold:
            raise ValueError(f"sample={row.get('id')} Gold 为空")
        example = {
            "example_id": f"{row.get('id')}:base_parent:direct_gold",
            "sample_id": str(row.get("id", "")),
            "user_id": str(row.get("user_id", "")),
            # 保留原始字段，便于后续构造 RATIONALE 辅助任务；这些字段不
            # 会拼入 prompt 以外的训练目标，也不会包含额外 Gold 泄漏。
            "source_text": str(row.get("source_text", "")),
            "profile": list(row.get("profile", [])),
            "target": gold,
            "operation_type": "mutation",
            "parent_a_id": parent_candidate["candidate_id"],
            "parent_b_id": None,
            "parent": parent,
            "parent_source": str(row.get("parent_source", "base_model")),
            "output": gold,
            "prompt": prompt,
            "trace_text": "",
            "output_text": gold,
            "sample_weight": 1.0,
            "split": deterministic_split(str(row.get("id", "")), fraction),
            "student_prompt_sees_gold": False,
            "teacher_used": False,
        }
        examples.append(example)
        print(
            f"direct SFT progress {index}/{len(rows)} sample={row.get('id')} "
            f"category={quality['category']} sim={quality['char_similarity']:.3f}",
            flush=True,
        )
    return examples, analyses


def write_analysis_markdown(output: Path, analyses: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for item in analyses:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    lines = [
        "# Direct Parent→Gold SFT Pilot 逐条质量分析",
        "",
        "该分析仅用于离线检查，Gold 不会写入 Student Prompt。",
        "",
        "## 汇总",
        "",
        f"- 样本数：{len(analyses)}",
        f"- 类别统计：`{json.dumps(counts, ensure_ascii=False)}`",
        f"- 非空 Parent：{sum(int(x['parent_nonempty']) for x in analyses)}/{len(analyses)}",
        f"- 归一化后与 Gold 相同：{sum(int(x['normalized_exact']) for x in analyses)}/{len(analyses)}",
        "",
        "## 逐条结果",
        "",
        "| Sample | 类别 | 字符相似度 | Token Jaccard | Parent | Gold |",
        "|---|---|---:|---:|---|---|",
    ]
    for item in analyses:
        parent = item["parent"].replace("|", "\\|")
        gold = item["gold"].replace("|", "\\|")
        lines.append(
            f"| {item['sample_id']} | {item['category']} | {item['char_similarity']:.3f} | "
            f"{item['token_jaccard']:.3f} | {parent} | {gold} |"
        )
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="构造纯文本 Parent→Gold SFT 样本")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--parent-records", default="", help="已有 01_base_parent_records.jsonl")
    parser.add_argument("--parent-file", default="", help="已有 Base predictions JSONL；跳过本地模型生成")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    rows = load_parent_rows(args, config)
    if not rows:
        raise ValueError("没有可用的 Parent 行")
    examples, analyses = build_examples(rows, config)
    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # 供后续 RATIONALE 辅助任务复用真实 Parent；该文件只保存原始行，
    # 不包含 Teacher 标注，也不改变 Direct SFT 的训练目标。
    write_jsonl(output / "01_base_parent_records.jsonl", rows)
    write_jsonl(output / "all_sft.jsonl", examples)
    write_jsonl(output / "train_sft.jsonl", [x for x in examples if x["split"] == "train"])
    write_jsonl(output / "validation_sft.jsonl", [x for x in examples if x["split"] == "validation"])
    write_jsonl(output / "quality_analysis.jsonl", analyses)
    write_analysis_markdown(output, analyses)
    categories: dict[str, int] = {}
    for item in analyses:
        categories[item["category"]] = categories.get(item["category"], 0) + 1
    report = {
        "protocol": "direct_parent_gold_plain_text_sft_pilot_v1",
        "split": args.split,
        "source_queries": len(rows),
        "examples": len(examples),
        "train_examples": sum(x["split"] == "train" for x in examples),
        "validation_examples": sum(x["split"] == "validation" for x in examples),
        "one_parent_per_query": True,
        "parent_source": "existing_parent_records" if args.parent_records else "base_model_generation",
        "teacher_used": False,
        "student_prompt_sees_gold": False,
        "supervision_mode": "plain_output_only",
        "trace_loss_weight": 0.0,
        "output_loss_weight": 1.0,
        "quality_categories": categories,
        "output_dir": str(output),
    }
    write_json(output / "manifest.json", report)
    print(f"Direct Parent→Gold SFT -> {output}; report={report}", flush=True)


if __name__ == "__main__":
    main()
