#!/usr/bin/env python3
"""基于真实 Base Parent 构造 Gold-aware Editor SFT 监督。

本脚本把一个小规模数据构造流程放在同一个文件中：

1. 使用与 Base 推理完全相同的 Prompt，让 Llama2 生成真实 Parent；
2. Teacher 只在离线标注阶段查看 Query、History、Parent 和 Gold；
3. Teacher 判断 Parent 是 valid/repairable/unusable，并给出从 Parent 到 Gold
   的最小编辑解释；
4. 生成不含 Gold 的 Student Prompt，以及包含紧凑 Trace + Gold 的 SFT 样本。

Gold 不会写入 ``prompt``，只会写入监督字段 ``output_text``。默认丢弃
unusable Parent，避免把占位标题或完全无关标题当作正常编辑样本。

示例（先生成 8 个样本，不训练 GPU）：

    python code/26_9_1/build_base_parent_teacher_sft.py \
      --config config_global_llama2_7b_visgpt_prime_matched.yaml \
      --split train --limit 8 \
      --output dataset/editor_sets/base_parent_teacher_pilot
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

from pipeline_common import (  # noqa: E402
    build_base_text_prompt,
    load_config,
    read_jsonl,
    resolve_path,
    stage_path,
    teacher_client,
    visible_history,
    write_json,
    write_jsonl,
)


def clean_parent(text: str) -> str:
    """按 Base 评估协议取第一行标题，避免把解释当作 Parent。"""

    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:text|title)?\s*|\s*```$", "", value, flags=re.I).strip()
    value = re.sub(r"^(?:title|output)\s*:\s*", "", value, flags=re.I).strip()
    for line in value.splitlines():
        line = line.strip().strip('"').strip()
        if line:
            return line[:300]
    return ""


def normalized_text(text: str) -> str:
    """用于判断 keep 是否真的无需改变（忽略大小写、空格和标点）。"""

    return re.sub(r"[^a-z0-9]+", "", str(text).casefold())


def build_trace_student_prompt(row: dict[str, Any], parent: str, maximum_history: int = 8) -> str:
    """构造与 Trace+Output 监督目标匹配的 Student Prompt。

    Base 生成 Parent 使用纯文本标题协议；SFT Student 则必须被明确告知
    需要生成 JSON，否则会出现“Prompt 要求纯文本、标签却是 JSON”的协议冲突。
    Gold 只在离线标签中出现，不进入此 Prompt。
    """

    payload = {
        "current_input": str(row.get("source_text", "")),
        "retrieved_history": visible_history(row, maximum_history),
        "parent": parent,
    }
    schema = (
        '{"parent_quality":"valid|repairable|unusable",'
        '"decision":"keep|revise",'
        '"parent_issue":"short text",'
        '"history_signal":{"evidence_ids":[],"observation":"short text"},'
        '"edit_actions":[{"operation":"insert|delete|replace|reorder|compress|preserve",'
        '"source":"short span","target":"short span","reason":"short text"}],'
        '"output":"final title"}'
    )
    return (
        "You are a personalized academic-title editor. Use only CURRENT_INPUT, "
        "RETRIEVED_HISTORY, and PARENT. Decide whether the parent should be kept or "
        "revised, then describe only the minimum observable edits. Use history as optional "
        "evidence and cite only visible history IDs. Never mention a reference answer, gold, "
        "or evaluation metric. Return exactly one JSON object matching the schema.\n\n"
        f"REQUIRED_SCHEMA:\n{schema}\n\nPAYLOAD:\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n\nOUTPUT:\n"
    )


def generate_parents(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    cached_predictions: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """用冻结 Base 逐条生成 Parent；输入协议与 Base 评估完全一致。"""

    # 已有同一 Base/同一 Prompt 的缓存时直接复用，便于在 GPU 被其他任务
    # 占用时先完成 Teacher 标注 pilot；这仍然是真实 Base 分布，而不是伪造标题。
    if cached_predictions is not None:
        output = []
        for index, row in enumerate(rows, 1):
            cached = cached_predictions.get(str(row.get("id")), {})
            item = dict(row)
            candidates = list(row.get("candidates", []))
            cached_parent = clean_parent(
                cached.get("prediction", cached.get("parent", ""))
            )
            generation_prompt = (
                build_base_text_prompt(
                    row,
                    candidates[0],
                    int(config.get("sft_data", {}).get("maximum_history_records", 8)),
                )
                if candidates
                else build_base_text_prompt(
                    row,
                    {
                        "candidate_id": f"{row.get('id')}:base_model",
                        "text": cached_parent,
                    },
                    int(config.get("sft_data", {}).get("maximum_history_records", 8)),
                )
            )
            item["parent"] = cached_parent
            item["base_prompt"] = build_trace_student_prompt(
                row,
                item["parent"],
                int(config.get("sft_data", {}).get("maximum_history_records", 8)),
            )
            item["parent_generation_prompt"] = generation_prompt
            item["parent_raw_response"] = str(cached.get("raw_response", item["parent"]))
            item["parent_source"] = "cached_base_model"
            output.append(item)
            print(
                f"parent cache progress {index}/{len(rows)} sample={row.get('id')} "
                f"valid={bool(item['parent'])}",
                flush=True,
            )
        return output

    # 只有真正调用本地 Base 时才需要 hydra 环境中的 torch/transformers；
    # 使用 --parent-file 的 Teacher pilot 可以在轻量数据环境中运行。
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = resolve_path(config["model"]["path"])
    if not model_path.exists():
        raise FileNotFoundError(f"基础模型不存在: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    ).to(device).eval()
    settings = config.get("inference", {})
    max_length = int(config.get("training", {}).get("max_length", 2048))
    max_new_tokens = int(settings.get("max_new_tokens", 64))
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        candidates = list(row.get("candidates", []))
        # 没有 seed 时不能伪造 Parent；这类样本后续会被报告为 unusable。
        if candidates:
            parent = ""
            generation_prompt = build_base_text_prompt(
                row, candidates[0], int(config.get("sft_data", {}).get("maximum_history_records", 8))
            )
            encoded = tokenizer(
                generation_prompt, return_tensors="pt", truncation=True, max_length=max_length
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            raw = tokenizer.decode(
                generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
            )
            parent = clean_parent(raw)
        else:
            prompt, raw, parent = "", "", ""
        item = dict(row)
        item["base_prompt"] = build_trace_student_prompt(
            row,
            parent,
            int(config.get("sft_data", {}).get("maximum_history_records", 8)),
        )
        item["parent_generation_prompt"] = generation_prompt if candidates else ""
        item["parent"] = parent
        item["parent_raw_response"] = raw
        item["parent_source"] = "base_model"
        output.append(item)
        print(
            f"parent progress {index}/{len(rows)} sample={row.get('id')} "
            f"valid={bool(parent)}",
            flush=True,
        )
    # 释放 Base，Teacher 阶段只使用 API，不与本地模型争用显存。
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def teacher_prompt(row: dict[str, Any]) -> str:
    """Teacher 的 Gold-aware 标注提示；Gold 不会进入 Student Prompt。"""

    history = visible_history(row, 8)
    schema = (
        '{"parent_quality":"valid|repairable|unusable",'
        '"decision":"keep|revise",'
        '"parent_issue":"short text",'
        '"history_signal":{"evidence_ids":[],"observation":"short text"},'
        '"edit_actions":[{"operation":"insert|delete|replace|reorder|compress|preserve",'
        '"source":"short span","target":"short span","reason":"short text"}],'
        '"output":"exact Gold title"}'
    )
    payload = {
        "current_input": str(row.get("source_text", "")),
        "retrieved_history": history,
        "parent": str(row.get("parent", "")),
        "gold": str(row.get("target", "")),
    }
    return (
        "You are an offline annotation teacher for a personalized text editor. "
        "Gold is available only for annotation. Judge the given Parent; do not invent a "
        "different Parent. Mark it valid when it is already semantically correct, "
        "repairable when it is related but misses or misstates a fixable detail, and "
        "unusable when it is a placeholder, unrelated, or the input lacks enough content. "
        "For valid/repairable cases, give the minimum observable edits from Parent to Gold. "
        "Use history only as optional evidence of a user-specific behavior; an empty signal "
        "is allowed. Do not write a long rationale. Return JSON only.\n\n"
        f"REQUIRED_SCHEMA:\n{schema}\n\nPAYLOAD:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def normalize_annotation(value: Any, row: dict[str, Any]) -> dict[str, Any]:
    """宽容处理 Teacher 字段，但不放宽质量类别和输出一致性。"""

    if not isinstance(value, dict):
        raise ValueError("Teacher 标注不是 JSON object")
    quality = str(value.get("parent_quality", "")).strip().lower()
    if quality not in {"valid", "repairable", "unusable"}:
        raise ValueError(f"parent_quality 无效: {quality!r}")
    decision = str(value.get("decision", "")).strip().lower()
    if decision not in {"keep", "revise"}:
        decision = "keep" if quality == "valid" else "revise"
    # Teacher 有时把“语义上可接受”误标为 keep，但它仍然不同于 Gold。
    # SFT 的 keep 必须表示 Parent 与 Gold 相同（允许标点/大小写差异），
    # 因此这里将矛盾标签自动降级为 repairable + revise。
    if decision == "keep" and normalized_text(row.get("parent", "")) != normalized_text(
        row.get("target", "")
    ):
        decision = "revise"
        if quality == "valid":
            quality = "repairable"
    signal = value.get("history_signal")
    if not isinstance(signal, dict):
        signal = {"evidence_ids": [], "observation": ""}
    evidence = signal.get("evidence_ids", [])
    if not isinstance(evidence, list):
        evidence = []
    visible_ids = {str(item.get("id", "")) for item in visible_history(row, 8)}
    evidence = [str(item) for item in evidence if str(item) in visible_ids]
    observation = str(signal.get("observation", "")).strip()[:240]
    actions = value.get("edit_actions", [])
    if not isinstance(actions, list):
        actions = []
    clean_actions = []
    allowed = {"insert", "delete", "replace", "reorder", "compress", "preserve"}
    for action in actions[:6]:
        if not isinstance(action, dict):
            continue
        operation = str(action.get("operation", "")).strip().lower()
        if operation not in allowed:
            continue
        clean_actions.append(
            {
                "operation": operation,
                "source": str(action.get("source", "")).strip()[:120],
                "target": str(action.get("target", "")).strip()[:120],
                "reason": str(action.get("reason", "")).strip()[:240],
            }
        )
    # 最终监督严格由数据集 Gold 写入，避免 Teacher 输出错 Gold 破坏标签。
    return {
        "parent_quality": quality,
        "decision": decision,
        "parent_issue": str(value.get("parent_issue", "")).strip()[:240],
        "history_signal": {"evidence_ids": evidence, "observation": observation},
        "edit_actions": clean_actions,
        "output": str(row.get("target", "")).strip(),
    }


def annotate(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    client = teacher_client(config)
    annotations: list[dict[str, Any]] = []
    sft: list[dict[str, Any]] = []
    include_unusable = bool(config.get("parent_teacher_sft", {}).get("include_unusable", False))
    for index, row in enumerate(rows, 1):
        raw_payload = None
        raw_response = ""
        error: Exception | None = None
        for attempt in range(3):
            try:
                raw_payload, raw_response = client.json(
                    f"base_parent_teacher_sft_{row.get('id')}_attempt_{attempt}",
                    teacher_prompt(row),
                    {"sample_id": str(row.get("id"))},
                )
                annotation = normalize_annotation(raw_payload, row)
                error = None
                break
            except Exception as exc:  # Teacher 服务瞬时错误或 schema 错误时重试
                error = exc
                client.invalidate(
                    f"base_parent_teacher_sft_{row.get('id')}_attempt_{attempt}",
                    teacher_prompt(row),
                )
        if error is not None:
            raise RuntimeError(f"Teacher 标注失败 sample={row.get('id')}: {error}") from error
        annotation_row = {
            "sample_id": str(row.get("id")),
            "user_id": str(row.get("user_id", "")),
            "parent": str(row.get("parent", "")),
            "gold": str(row.get("target", "")),
            "annotation": annotation,
            "teacher_raw_response": raw_response,
        }
        annotations.append(annotation_row)
        if include_unusable or annotation["parent_quality"] != "unusable":
            trace = {
                "parent_quality": annotation["parent_quality"],
                "decision": annotation["decision"],
                "parent_issue": annotation["parent_issue"],
                "history_signal": annotation["history_signal"],
                "edit_actions": annotation["edit_actions"],
            }
            trace_json = json.dumps(trace, ensure_ascii=False, separators=(",", ":"))
            # 与项目现有 TraceOutputDataset 兼容：Trace 是 JSON object 的前缀，
            # output_text 补齐最后的字符串和右括号，完整目标只有一个 JSON 对象。
            trace_text = trace_json[:-1] + ',"output":'
            output_text = json.dumps(annotation["output"], ensure_ascii=False) + "}"
            sft.append(
                {
                    "example_id": f"{row.get('id')}:base_parent",
                    "sample_id": str(row.get("id")),
                    "user_id": str(row.get("user_id", "")),
                    "prompt": str(row.get("base_prompt", "")),
                    "parent": str(row.get("parent", "")),
                    "parent_source": "base_model",
                    "trace": trace,
                    "trace_text": trace_text,
                    "output": str(row.get("target", "")).strip(),
                    "output_text": output_text,
                    "student_prompt_sees_gold": False,
                    "teacher_sees_gold": True,
                }
            )
        print(
            f"teacher progress {index}/{len(rows)} sample={row.get('id')} "
            f"quality={annotation['parent_quality']} sft={len(sft)}",
            flush=True,
        )
    return annotations, sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Base Parent + Teacher Gold-aware SFT pilot")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument(
        "--parent-file",
        default="",
        help="可选的同一 Base Prompt 预测缓存 JSONL；提供后跳过本地模型加载",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    rows = read_jsonl(stage_path(config, args.split, "retrieve"))
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("输入数据为空；请先运行 01_prepare.py 和 02_retrieve.py")
    cached_predictions = None
    if args.parent_file:
        cached_predictions = {
            str(item["id"]): item
            for item in read_jsonl(resolve_path(args.parent_file))
        }
        print(
            f"using cached Base predictions={len(cached_predictions)}; "
            "skip local Base model generation",
            flush=True,
        )
    print(f"base Parent generation rows={len(rows)} split={args.split}", flush=True)
    parent_rows = generate_parents(rows, config, cached_predictions)
    annotations, sft = annotate(parent_rows, config)
    output = resolve_path(args.output)
    write_jsonl(output / "01_base_parent_records.jsonl", parent_rows)
    write_jsonl(output / "02_teacher_annotations.jsonl", annotations)
    write_jsonl(output / "03_sft_examples.jsonl", sft)
    quality_counts: dict[str, int] = {}
    for row in annotations:
        quality = row["annotation"]["parent_quality"]
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    report = {
        "protocol": "base_model_parent_teacher_gold_aware_sft_pilot_v1",
        "split": args.split,
        "input_rows": len(rows),
        "parent_source": (
            "cached Llama2-7B frozen base model"
            if cached_predictions is not None
            else "Llama2-7B frozen base model"
        ),
        "teacher_sees_gold": True,
        "student_prompt_sees_gold": False,
        "sft_examples": len(sft),
        "quality_counts": quality_counts,
        "unusable_dropped": (
            0
            if bool(config.get("parent_teacher_sft", {}).get("include_unusable", False))
            else sum(
                1
                for row in annotations
                if row["annotation"]["parent_quality"] == "unusable"
            )
        ),
        "output_dir": str(output),
    }
    write_json(output / "report.json", report)
    print(f"Base Parent + Teacher SFT pilot -> {output}; report={report}", flush=True)


if __name__ == "__main__":
    main()
