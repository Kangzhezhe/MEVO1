"""阶段 07：用训练后的 LoRA Editor 生成 Mutation/Crossover 候选池。

模型只读取当前输入、父候选和检索历史，不读取 Gold，也不读取任何预先构建的
用户 Factor。Gold 只在十个候选全部生成后用于离线评分。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline_common import (  # noqa: E402
    build_editor_prompt,
    candidate,
    choose_crossover_pairs,
    load_config,
    normalized_text,
    read_jsonl,
    resolve_path,
    score_candidate_pool,
    stage_path,
    validate_crossover,
    validate_mutation,
    visible_history,
    write_json,
    write_jsonl,
)


def extract_json(text: str, output_only: bool = False) -> dict[str, Any]:
    """容忍 Markdown fence、前后说明和结尾多余逗号。"""

    value = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, re.I | re.S)
    if fenced:
        value = fenced.group(1).strip()
    candidates = [value]
    start, end = value.find("{"), value.rfind("}")
    if 0 <= start <= end:
        candidates.append(value[start : end + 1])
    error: Exception | None = None
    for item in candidates:
        try:
            parsed = json.loads(re.sub(r",\s*([}\]])", r"\1", item))
            if not isinstance(parsed, dict):
                raise ValueError("Editor 输出必须是 JSON object")
            return parsed
        except (json.JSONDecodeError, ValueError) as current:
            error = current
    # S0 的监督本质是最终标题，不应因模型漏掉 JSON 包装而整条候选失效。
    # 仅 output-only 开启此回退；Gold-aware Trace 仍要求完整结构化响应。
    if output_only:
        raw = value.strip().strip('"').strip()
        if raw.startswith("```") and raw.endswith("```"):
            raw = raw[3:-3].strip().removeprefix("json").strip()
        if raw and "\n" not in raw and len(raw) <= 300:
            return {"output": raw}
    raise ValueError(f"无法解析 Editor JSON: {error}")


class LocalEditor:
    """一次加载 Qwen 基座和 LoRA，按配置批量生成编辑响应。"""

    def __init__(self, config: dict):
        # 数据处理虚拟环境不一定安装 GPU 依赖，因此仅在真实推理时导入。
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        model_path = resolve_path(config["model"]["path"])
        configured_adapter = config.get("model", {}).get("adapter_path")
        adapter_path = (
            resolve_path(configured_adapter)
            if configured_adapter
            else resolve_path(config["paths"]["editor_output_dir"]) / "final_adapter"
        )
        if not adapter_path.exists():
            raise FileNotFoundError(f"Editor adapter 不存在: {adapter_path}")
        fraction = float(config["training"].get("cuda_memory_fraction", 0.0))
        if torch.cuda.is_available() and fraction > 0:
            torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path, use_fast=True, local_files_only=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only 模型批量生成必须左侧 padding，保证最后一个输入 token 对齐。
        self.tokenizer.padding_side = "left"
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path).to(self.device).eval()
        self.config = config
        self.output_only = str(
            config.get("sft_data", {}).get("supervision_mode", "gold_aware_trace")
        ) in {
            "output_only",
            "atomic_trace",
            "conditional_preference_trace",
            "simple_conditional_trace",
        }

    def generate_many(
        self, prompts: list[str]
    ) -> list[tuple[dict[str, Any] | None, str, Exception | None]]:
        torch = self.torch
        settings = self.config["inference"]
        batch_size = max(1, int(settings.get("batch_size", 1)))
        results: list[tuple[dict[str, Any] | None, str, Exception | None]] = []
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            # 必须与 SFT Dataset 一致地同时保留 Prompt 开头的任务说明和末尾的
            # Parent/OUTPUT 标记。Tokenizer 默认右截断会切掉末尾，模型随后只会
            # 续写中间的 History JSON，导致大量无关输出和无效 JSON。
            maximum = int(self.config["training"]["max_length"])
            batch_ids = [
                truncate_prompt_ids(
                    self.tokenizer.encode(prompt, add_special_tokens=True), maximum
                )
                for prompt in batch_prompts
            ]
            encoded = self.tokenizer.pad(
                {"input_ids": batch_ids},
                return_tensors="pt",
                padding=True,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            input_width = int(encoded["input_ids"].shape[1])
            try:
                with torch.inference_mode():
                    generated = self.model.generate(
                        **encoded,
                        max_new_tokens=int(settings["max_new_tokens"]),
                        do_sample=bool(settings.get("do_sample", False)),
                        temperature=float(settings.get("temperature", 1.0)),
                        top_p=float(settings.get("top_p", 1.0)),
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                raw_values = self.tokenizer.batch_decode(
                    generated[:, input_width:], skip_special_tokens=True
                )
                for raw_value in raw_values:
                    raw = raw_value.strip()
                    try:
                        results.append((extract_json(raw, self.output_only), raw, None))
                    except Exception as error:
                        results.append((None, raw, error))
            except Exception as error:
                # 保留 batch 中每个逻辑请求的位置，随后逐条执行 keep 回退与记录。
                results.extend((None, "", error) for _ in batch_prompts)
        return results


def truncate_prompt_ids(input_ids: list[int], maximum: int) -> list[int]:
    """保留 Prompt 头尾；与训练数据的截断契约一致。"""

    if maximum <= 1:
        raise ValueError("maximum 必须大于 1")
    if len(input_ids) <= maximum:
        return input_ids
    head = max(1, maximum // 2)
    return input_ids[:head] + input_ids[-(maximum - head) :]


class MockLocalEditor:
    """仅用于 smoke：按 Prompt 内父候选返回合法 keep，不模拟质量收益。"""

    def _generate(self, prompt: str) -> tuple[dict[str, Any], str]:
        payload = json.loads(
            prompt.split("PAYLOAD:\n", 1)[1].rsplit("\n\nOUTPUT:\n", 1)[0]
        )
        history = payload.get("retrieved_history", [])
        signal = _fallback_signal(history)
        parent_a = payload["parent_a"]
        if payload["operation_type"] == "mutation":
            value = {
                "decision": "keep",
                "task_correction": "No task-content correction is needed in this structural smoke test.",
                "profile_signal": signal,
                "edit_action": "Keep the parent in this structural smoke test.",
                "output": str(parent_a["text"]),
            }
        else:
            value = {
                "decision": "keep_a",
                "profile_signal": signal,
                "merge_action": "Keep parent A in this structural smoke test.",
                "output": str(parent_a["text"]),
            }
        return value, json.dumps(value, ensure_ascii=False)

    def generate_many(
        self, prompts: list[str]
    ) -> list[tuple[dict[str, Any] | None, str, Exception | None]]:
        return [(*self._generate(prompt), None) for prompt in prompts]


def _fallback_signal(history: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "evidence_ids": [history[0]["id"]] if history else [],
        "observation": "The visible history does not support a reliable change.",
    }


def _mutation_fallback(parent: dict, history: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "parent_id": str(parent["candidate_id"]),
        "decision": "keep",
        "profile_signal": _fallback_signal(history),
        "edit_action": "Keep the current parent because no reliable edit was produced.",
        "output": str(parent["text"]),
    }


def _output_only_mutation(
    payload: dict[str, Any], parent: dict, history: list[dict[str, str]]
) -> dict[str, Any]:
    """把 S0 的单字段输出转换成候选池统一诊断结构。"""

    output = candidate("s0-output", "mutation", payload.get("output", ""))["text"]
    keep = normalized_text(output) == normalized_text(parent["text"])
    return {
        "decision": "keep" if keep else "revise",
        "profile_signal": _fallback_signal(history),
        "edit_action": "S0 output-only supervision; no explicit edit trace was generated.",
        "output": output,
    }


def _crossover_fallback(
    left: dict, right: dict, history: list[dict[str, str]]
) -> dict[str, Any]:
    return {
        "parent_a_id": str(left["candidate_id"]),
        "parent_b_id": str(right["candidate_id"]),
        "decision": "keep_a",
        "profile_signal": _fallback_signal(history),
        "merge_action": "Keep parent A because no reliable merge was produced.",
        "output": str(left["text"]),
    }


def _output_only_crossover(
    payload: dict[str, Any], left: dict, right: dict, history: list[dict[str, str]]
) -> dict[str, Any]:
    output = candidate("s0-output", "crossover", payload.get("output", ""))["text"]
    if normalized_text(output) == normalized_text(left["text"]):
        decision = "keep_a"
    elif normalized_text(output) == normalized_text(right["text"]):
        decision = "keep_b"
    else:
        decision = "merge"
    return {
        "decision": decision,
        "profile_signal": _fallback_signal(history),
        "merge_action": "S0 output-only supervision; no explicit merge trace was generated.",
        "output": output,
    }


def generate_one(row: dict[str, Any], config: dict, editor: Any) -> dict[str, Any]:
    settings = config["evolution"]
    history = visible_history(row, int(settings["maximum_history_records"]))
    supervision_mode = str(
        config.get("sft_data", {}).get("supervision_mode", "gold_aware_trace")
    )
    simple_settings = config.get("simple_conditional_trace", {})
    prompt_limits = (
        {
            "history_input_max_chars": int(simple_settings.get("history_input_max_chars", 500)),
            "history_output_max_chars": int(simple_settings.get("history_output_max_chars", 300)),
        }
        if supervision_mode == "simple_conditional_trace"
        else {}
    )
    evidence = {item["id"] for item in history}
    failures = []
    mutation_prompts = [
        build_editor_prompt(
            row,
            "mutation",
            parent,
            None,
            int(settings["maximum_history_records"]),
            supervision_mode=supervision_mode,
            **prompt_limits,
        )
        for parent in row["candidates"]
    ]
    mutation_results = editor.generate_many(mutation_prompts)
    if len(mutation_results) != len(row["candidates"]):
        raise AssertionError("Mutation batch 返回数量错误")
    mutations = []
    for parent, (payload, raw, generation_error) in zip(
        row["candidates"], mutation_results
    ):
        failed = generation_error is not None
        try:
            if generation_error is not None:
                raise generation_error
            if payload is None:
                raise ValueError("Mutation batch 缺少 JSON payload")
            if supervision_mode in {
                "output_only", "atomic_trace", "conditional_preference_trace",
                "simple_conditional_trace",
            }:
                value = _output_only_mutation(payload, parent, history)
            else:
                payload["parent_id"] = str(parent["candidate_id"])
                value = validate_mutation(payload, parent, evidence)
        except Exception as error:  # 单条格式错误不应丢弃整个用户。
            if not bool(config["inference"].get("fallback_on_invalid_json", True)):
                raise
            failed = True
            failures.append(
                {"operation": "mutation", "parent_id": parent["candidate_id"],
                 "error": str(error), "raw_response": raw}
            )
            value = validate_mutation(_mutation_fallback(parent, history), parent, evidence)
        mutations.append(
            candidate(
                f"{parent['candidate_id']}_editor_mut",
                "mutation",
                value.pop("output"),
                parent_id=str(parent["candidate_id"]),
                editor_valid=not failed,
                **value,
            )
        )

    pairs = choose_crossover_pairs(
        list(row["candidates"]) + mutations, int(settings["crossovers_per_query"])
    )
    crossover_prompts = [
        build_editor_prompt(
            row,
            "crossover",
            left,
            right,
            int(settings["maximum_history_records"]),
            supervision_mode=supervision_mode,
            **prompt_limits,
        )
        for left, right in pairs
    ]
    crossover_results = editor.generate_many(crossover_prompts)
    if len(crossover_results) != len(pairs):
        raise AssertionError("Crossover batch 返回数量错误")
    crossovers = []
    for index, ((left, right), (payload, raw, generation_error)) in enumerate(
        zip(pairs, crossover_results)
    ):
        failed = generation_error is not None
        try:
            if generation_error is not None:
                raise generation_error
            if payload is None:
                raise ValueError("Crossover batch 缺少 JSON payload")
            if supervision_mode in {
                "output_only", "atomic_trace", "conditional_preference_trace",
                "simple_conditional_trace",
            }:
                value = _output_only_crossover(payload, left, right, history)
            else:
                payload["parent_a_id"] = str(left["candidate_id"])
                payload["parent_b_id"] = str(right["candidate_id"])
                value = validate_crossover(payload, left, right, evidence)
        except Exception as error:
            if not bool(config["inference"].get("fallback_on_invalid_json", True)):
                raise
            failed = True
            failures.append(
                {"operation": "crossover", "parent_a_id": left["candidate_id"],
                 "parent_b_id": right["candidate_id"], "error": str(error),
                 "raw_response": raw}
            )
            value = validate_crossover(
                _crossover_fallback(left, right, history), left, right, evidence
            )
        crossovers.append(
            candidate(
                f"{row['id']}_editor_cross_{index}",
                "crossover",
                value.pop("output"),
                parent_a_id=str(left["candidate_id"]),
                parent_b_id=str(right["candidate_id"]),
                editor_valid=not failed,
                **value,
            )
        )

    output = {**row, "mutations": mutations + crossovers}
    output.pop("factors", None)
    budget = len(output["candidates"]) + len(output["mutations"])
    if budget != int(settings["candidate_budget"]):
        raise AssertionError(f"sample={row['id']} 候选预算错误: {budget}")
    score_candidate_pool(
        output,
        str(config["metric"]["primary"]),
        float(config["metric"]["preference_margin"]),
    )
    output["editor_metadata"] = {
        "candidate_budget": budget,
        "mutation_count": len(mutations),
        "crossover_count": len(crossovers),
        "invalid_generations": len(failures),
        "failures": failures,
        "gold_visible_during_generation": False,
        "explicit_user_factors": False,
        "supervision_mode": supervision_mode,
        "inference_batch_size": int(config["inference"].get("batch_size", 1)),
        "generation_batches": (
            (len(row["candidates"]) + int(config["inference"].get("batch_size", 1)) - 1)
            // int(config["inference"].get("batch_size", 1))
            + (len(pairs) + int(config["inference"].get("batch_size", 1)) - 1)
            // int(config["inference"].get("batch_size", 1))
        ),
    }
    return output


def generate_many_rows(
    rows: list[dict[str, Any]], config: dict[str, Any], editor: Any
) -> list[dict[str, Any]]:
    """Generate several rows in cross-Query mutation/crossover batches.

    Rows are grouped by the caller so they share one Editor/Adapter.  The
    mutation batch must finish before crossover prompts can be built, but all
    rows participate in each of those two model calls.  This preserves the
    original per-row candidate budget and fallback behavior.
    """
    if not rows:
        return []
    settings = config["evolution"]
    supervision_mode = str(
        config.get("sft_data", {}).get("supervision_mode", "gold_aware_trace")
    )
    simple_settings = config.get("simple_conditional_trace", {})
    prompt_limits = (
        {
            "history_input_max_chars": int(simple_settings.get("history_input_max_chars", 500)),
            "history_output_max_chars": int(simple_settings.get("history_output_max_chars", 300)),
        }
        if supervision_mode == "simple_conditional_trace"
        else {}
    )
    prepared = []
    mutation_prompts = []
    for row in rows:
        history = visible_history(row, int(settings["maximum_history_records"]))
        evidence = {item["id"] for item in history}
        prompts = [
            build_editor_prompt(
                row, "mutation", parent, None,
                int(settings["maximum_history_records"]),
                supervision_mode=supervision_mode,
                **prompt_limits,
            )
            for parent in row["candidates"]
        ]
        prepared.append({"row": row, "history": history, "evidence": evidence, "prompts": prompts, "failures": []})
        mutation_prompts.extend(prompts)
    mutation_results = editor.generate_many(mutation_prompts)
    if len(mutation_results) != len(mutation_prompts):
        raise AssertionError("跨 Query Mutation batch 返回数量错误")
    offset = 0
    for item in prepared:
        row = item["row"]
        count = len(row["candidates"])
        mutations = []
        for parent, (payload, raw, generation_error) in zip(
            row["candidates"], mutation_results[offset : offset + count]
        ):
            failed = generation_error is not None
            try:
                if generation_error is not None:
                    raise generation_error
                if payload is None:
                    raise ValueError("Mutation batch 缺少 JSON payload")
                if supervision_mode in {
                    "output_only", "atomic_trace", "conditional_preference_trace",
                    "simple_conditional_trace",
                }:
                    value = _output_only_mutation(payload, parent, item["history"])
                else:
                    payload["parent_id"] = str(parent["candidate_id"])
                    value = validate_mutation(payload, parent, item["evidence"])
            except Exception as error:
                if not bool(config["inference"].get("fallback_on_invalid_json", True)):
                    raise
                failed = True
                item["failures"].append(
                    {"operation": "mutation", "parent_id": parent["candidate_id"],
                     "error": str(error), "raw_response": raw}
                )
                value = validate_mutation(
                    _mutation_fallback(parent, item["history"]), parent, item["evidence"]
                )
            mutations.append(
                candidate(
                    f"{parent['candidate_id']}_editor_mut", "mutation",
                    value.pop("output"), parent_id=str(parent["candidate_id"]),
                    editor_valid=not failed, **value,
                )
            )
        item["mutations"] = mutations
        offset += count

    crossover_prompts = []
    for item in prepared:
        row = item["row"]
        pairs = choose_crossover_pairs(
            list(row["candidates"]) + item["mutations"],
            int(settings["crossovers_per_query"]),
        )
        item["pairs"] = pairs
        prompts = [
            build_editor_prompt(
                row, "crossover", left, right,
                int(settings["maximum_history_records"]),
                supervision_mode=supervision_mode,
                **prompt_limits,
            )
            for left, right in pairs
        ]
        item["crossover_prompts"] = prompts
        crossover_prompts.extend(prompts)
    crossover_results = editor.generate_many(crossover_prompts)
    if len(crossover_results) != len(crossover_prompts):
        raise AssertionError("跨 Query Crossover batch 返回数量错误")
    offset = 0
    outputs = []
    for item in prepared:
        row = item["row"]
        pairs = item["pairs"]
        crossovers = []
        for index, ((left, right), (payload, raw, generation_error)) in enumerate(
            zip(pairs, crossover_results[offset : offset + len(pairs)])
        ):
            failed = generation_error is not None
            try:
                if generation_error is not None:
                    raise generation_error
                if payload is None:
                    raise ValueError("Crossover batch 缺少 JSON payload")
                if supervision_mode in {
                    "output_only", "atomic_trace", "conditional_preference_trace",
                    "simple_conditional_trace",
                }:
                    value = _output_only_crossover(payload, left, right, item["history"])
                else:
                    payload["parent_a_id"] = str(left["candidate_id"])
                    payload["parent_b_id"] = str(right["candidate_id"])
                    value = validate_crossover(payload, left, right, item["evidence"])
            except Exception as error:
                if not bool(config["inference"].get("fallback_on_invalid_json", True)):
                    raise
                failed = True
                item["failures"].append(
                    {"operation": "crossover", "parent_a_id": left["candidate_id"],
                     "parent_b_id": right["candidate_id"], "error": str(error),
                     "raw_response": raw}
                )
                value = validate_crossover(
                    _crossover_fallback(left, right, item["history"]),
                    left, right, item["evidence"],
                )
            crossovers.append(
                candidate(
                    f"{row['id']}_editor_cross_{index}", "crossover",
                    value.pop("output"), parent_a_id=str(left["candidate_id"]),
                    parent_b_id=str(right["candidate_id"]), editor_valid=not failed,
                    **value,
                )
            )
        output = {**row, "mutations": item["mutations"] + crossovers}
        output.pop("factors", None)
        budget = len(output["candidates"]) + len(output["mutations"])
        if budget != int(settings["candidate_budget"]):
            raise AssertionError(f"sample={row['id']} 候选预算错误: {budget}")
        score_candidate_pool(
            output, str(config["metric"]["primary"]),
            float(config["metric"]["preference_margin"]),
        )
        output["editor_metadata"] = {
            "candidate_budget": budget,
            "mutation_count": len(item["mutations"]),
            "crossover_count": len(crossovers),
            "invalid_generations": len(item["failures"]),
            "failures": item["failures"],
            "gold_visible_during_generation": False,
            "explicit_user_factors": False,
            "supervision_mode": supervision_mode,
            "cross_query_batch": True,
            "inference_batch_size": int(config["inference"].get("batch_size", 1)),
        }
        outputs.append(output)
        offset += len(pairs)
    return outputs


def generate(config: dict, split: str, limit: int = 0) -> dict[str, Any]:
    source_rows = read_jsonl(stage_path(config, split, "seeds"))
    if limit > 0:
        source_rows = source_rows[:limit]
    destination = stage_path(config, split, "editor")
    resume = bool(config["inference"].get("resume_existing", True))
    existing = read_jsonl(destination) if resume and destination.exists() else []
    source_ids = {str(row["id"]) for row in source_rows}
    by_id = {
        str(row["id"]): row for row in existing
        if str(row["id"]) in source_ids
        if int(row.get("editor_metadata", {}).get("candidate_budget", 0))
        == int(config["evolution"]["candidate_budget"])
    }
    jobs = [row for row in source_rows if str(row["id"]) not in by_id]
    # 数据 smoke 可显式跳过模型；full smoke 即使 Teacher=mock 也加载真实 LoRA。
    editor = (
        MockLocalEditor()
        if bool(config["inference"].get("mock_editor", False))
        else LocalEditor(config)
    )
    checkpoint_every = int(config["inference"].get("checkpoint_every", 10))
    for index, row in enumerate(jobs, 1):
        by_id[str(row["id"])] = generate_one(row, config, editor)
        if index % checkpoint_every == 0:
            write_jsonl(destination, [by_id[str(item["id"])] for item in source_rows if str(item["id"]) in by_id])
        print(f"Editor progress {index}/{len(jobs)} sample={row['id']}", flush=True)
    ordered = [by_id[str(row["id"])] for row in source_rows if str(row["id"]) in by_id]
    write_jsonl(destination, ordered)
    calls = len(ordered) * (
        int(config["evolution"]["mutations_per_query"])
        + int(config["evolution"]["crossovers_per_query"])
    )
    invalid = sum(int(row["editor_metadata"]["invalid_generations"]) for row in ordered)
    batches = sum(int(row["editor_metadata"]["generation_batches"]) for row in ordered)
    report = {
        "split": split,
        "rows": len(ordered),
        "generation_calls": calls,
        "generation_batches": batches,
        "inference_batch_size": int(config["inference"].get("batch_size", 1)),
        "invalid_generations": invalid,
        "json_valid_rate": (calls - invalid) / calls if calls else 0.0,
        "candidate_budget": int(config["evolution"]["candidate_budget"]),
        "explicit_user_factors": False,
    }
    write_json(destination.parent / "07_editor_report.json", report)
    print(f"factor-free Editor pool -> {destination}; report={report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="07 - 本地 LoRA Editor 生成候选池")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "adaptation_validation", "adaptation_test"),
        default="validation",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    generate(load_config(args.config), args.split, args.limit)


if __name__ == "__main__":
    main()
