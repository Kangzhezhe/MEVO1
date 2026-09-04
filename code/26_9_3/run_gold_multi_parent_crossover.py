#!/usr/bin/env python3
"""Gold-supervised 多 Parent Crossover：单文件正式流程。

这不是 AFT 的 Teacher aggregation。训练集的 Gold 只作为 Student 输出监督：

    Query + Top-8 History -> {1 greedy + 4 sampled Parents}
    Query + Top-8 History + 5 Parents -> Gold

Teacher API、语义门控、Gold-aware Parent 筛选和 Parent A/B 二元配对均不在
主链路中。候选只做空值、明显 Prompt 污染和精确重复清理。训练时随机打乱
候选；当前固定宽度协议不使用 candidate-dropout，测试时同样使用全部五个槽位。

默认 ``--stage all`` 顺序完成：

1. 训练集 Parent Pool；
2. Crossover SFT JSONL；
3. Llama2-7B FP16 LoRA；
4. test100 Parent Pool 与 100 用户/608 Query 评估。

正式运行：

    /home/liux/miniconda3/envs/hydra/bin/python -B \
      code/26_9_3/run_gold_multi_parent_crossover.py --stage all

脚本会打印自动生成的 data_dir、run_dir。也可显式传入这两个目录分阶段续跑。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import random
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CODE = HERE.parent
PROJECT = CODE.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(PROJECT))

from common.metrics import corpus_bleu, score  # noqa: E402
from pipeline_common import (  # noqa: E402
    deterministic_split,
    load_config,
    read_jsonl,
    resolve_path,
    stage_path,
    truncate_prompt_ids,
    visible_history,
    write_json,
    write_jsonl,
)


DEFAULT_CONFIG = PROJECT / "config_global_llama2_7b_visgpt_prime_matched.yaml"
DEFAULT_PARENT_RECORDS = ""
INVALID_EXACT = re.compile(
    r"^(?:\[?paper title\]?|user title example|example input|abstract|title|output)\s*:?$",
    re.IGNORECASE,
)
INVALID_LEAK = re.compile(
    r"(?:write one concise|example input\s*:|example output|user title example|"
    r"parent title\s*:|abstract\s*:|output only|```)",
    re.IGNORECASE,
)


def configure(config: dict[str, Any]) -> dict[str, Any]:
    """在基础实验配置上固定本实验协议，避免依赖额外 YAML。"""

    value = copy.deepcopy(config)
    value.setdefault("crossover_sft", {}).update(
        {
            "temperature": 1.1,
            "top_p": 0.95,
            "sample_parent_count": 4,
            "total_parent_count": 5,
            # 固定宽度 Crossover：训练和测试均保留五个 Parent 槽位。
            "candidate_dropout_copies": 0,
            "max_new_tokens": 64,
            "checkpoint_every": 20,
        }
    )
    value.setdefault("sft_data", {}).update(
        {
            "supervision_mode": "plain_output_only",
            "maximum_history_records": 8,
            "validation_fraction": 0.05,
        }
    )
    value.setdefault("training", {}).update(
        {
            "max_length": 4096,
            "trace_loss_weight": 0.0,
            "output_loss_weight": 1.0,
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "epochs": 2,
            "learning_rate": 2e-4,
            "initial_adapter_path": "",
        }
    )
    value.setdefault("evaluation", {}).update(
        {
            "user_limit": 0,
            "expected_users": 100,
            "expected_queries": 608,
            "prediction_batch_size": 1,
            "max_new_tokens": 64,
        }
    )
    configured_model = resolve_path(value["model"]["path"])
    remote_model = PROJECT.parent / "PriME" / "models" / "Llama-2-7b-ms-hf"
    if not configured_model.exists() and remote_model.exists():
        value["model"]["path"] = str(remote_model)
        print(
            f"MODEL_PATH_FALLBACK {configured_model} -> {remote_model}",
            flush=True,
        )
    return value


def clean_title(value: Any) -> str:
    """统一提取首个非空纯文本标题行。"""

    text = str(value or "").strip()
    text = re.sub(r"^```(?:text|title)?\s*|\s*```$", "", text, flags=re.I).strip()
    text = re.sub(r"^(?:title|output)\s*:\s*", "", text, flags=re.I).strip()
    for line in text.splitlines():
        line = line.strip().strip('"').strip()
        if line:
            return line[:300]
    return ""


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def invalid_parent(title: str) -> bool:
    """只识别格式污染，不根据主题、Gold 或指标判断候选质量。"""

    return bool(
        INVALID_EXACT.match(title)
        or INVALID_LEAK.search(title)
        or title.lstrip().startswith(("•", "- ", "* "))
    )


def stable_rng(sample_id: str, variant: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{sample_id}:{variant}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def import_module(filename: str, name: str):
    path = CODE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_prompt(
    row: dict[str, Any], maximum_history: int, *, include_history: bool
) -> str:
    """直接从 Query/History 生成标题，不读取任何 Seed Parent。"""

    history = visible_history(row, maximum_history) if include_history else []
    demonstrations = "\n\n".join(
        f"Example input: {item['input'][:300]}\n"
        f"Example title: {item['output'][:180]}"
        for item in history
    )
    prefix = (
        "Write one concise academic paper title for the final input. Preserve factual "
        "content and output only the title on one line."
    )
    if demonstrations:
        prefix += " Use the preceding user examples as style guidance."
        context = f"\n\nUSER EXAMPLES:\n{demonstrations}"
    else:
        context = ""
    # 当前 Query 放在所有 few-shot 示例之后，使 Base LM 紧接着补全
    # Example title，而不是续写 Query 摘要。
    return (
        prefix
        + context
        + f"\n\nExample input: {row['source_text']}\nExample title:"
    )


class BaseParentSampler:
    """加载 Base Llama2-7B，生成 greedy 和 sampling Parents。"""

    def __init__(self, config: dict[str, Any]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = resolve_path(config["model"]["path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Base 模型不存在：{model_path}")
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, local_files_only=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()
        # 清理模型 generation_config 中与 greedy 无关的旧采样参数警告；
        # sampling 调用会显式覆盖这两个值。
        self.model.generation_config.temperature = 1.0
        self.model.generation_config.top_p = 1.0
        settings = config["crossover_sft"]
        self.max_length = int(config["training"]["max_length"])
        self.max_new_tokens = int(settings["max_new_tokens"])
        self.prompt_max_length = self.max_length - self.max_new_tokens
        if self.prompt_max_length <= 0:
            raise ValueError("max_new_tokens 必须小于 max_length")

    def generate(
        self,
        prompt: str,
        count: int,
        *,
        sample: bool,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> list[dict[str, str]]:
        if count <= 0:
            return []
        if count <= 0:
            return []
        prompt_ids = truncate_prompt_ids(
            self.tokenizer.encode(prompt, add_special_tokens=True),
            self.prompt_max_length,
        )
        encoded = self.tokenizer.pad(
            {"input_ids": [prompt_ids]}, return_tensors="pt", padding=True
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        self.torch.manual_seed(int(seed))
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(int(seed))
        kwargs: dict[str, Any] = {
            **encoded,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": sample,
            "num_return_sequences": count,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "remove_invalid_values": True,
            "renormalize_logits": True,
        }
        if sample:
            kwargs.update(
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=50,
            )
        with self.torch.inference_mode():
            sequences = self.model.generate(**kwargs)
        prompt_length = encoded["input_ids"].shape[1]
        output: list[dict[str, str]] = []
        for sequence in sequences:
            raw = self.tokenizer.decode(
                sequence[prompt_length:], skip_special_tokens=True
            )
            output.append({"text": clean_title(raw), "raw_response": raw})
        return output

    def close(self) -> None:
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class AdapterGenerator:
    """加载训练后的 LoRA，以纯文本协议生成最终标题。"""

    def __init__(self, config: dict[str, Any], adapter: Path):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = resolve_path(config["model"]["path"])
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter, use_fast=True, local_files_only=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(base, adapter).to(self.device).eval()
        self.max_length = int(config["training"]["max_length"])
        self.max_new_tokens = int(config["evaluation"]["max_new_tokens"])
        self.prompt_max_length = self.max_length - self.max_new_tokens

    def generate(self, prompts: list[str], batch_size: int) -> list[str]:
        output: list[str] = []
        for start in range(0, len(prompts), max(1, batch_size)):
            batch = prompts[start : start + max(1, batch_size)]
            batch_ids = [
                truncate_prompt_ids(
                    self.tokenizer.encode(prompt, add_special_tokens=True),
                    self.prompt_max_length,
                )
                for prompt in batch
            ]
            encoded = self.tokenizer.pad(
                {"input_ids": batch_ids}, return_tensors="pt", padding=True
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            prompt_length = encoded["input_ids"].shape[1]
            for sequence in generated:
                raw = self.tokenizer.decode(
                    sequence[prompt_length:], skip_special_tokens=True
                )
                output.append(clean_title(raw))
        return output

    def close(self) -> None:
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def append_parent(
    pool: list[dict[str, Any]], value: dict[str, Any], source: str
) -> None:
    """执行最低限度的格式卫生检查；有意保留精确重复候选。"""

    title = clean_title(value.get("text", ""))
    key = normalized(title)
    if (
        not key
        or invalid_parent(title)
        or not 2 <= len(title.split()) <= 40
    ):
        return
    pool.append({"candidate_id": "", "text": title, "source": source})


def usable_parents(row: dict[str, Any], maximum: int = 5) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for item in row.get("parent_pool", []):
        append_parent(output, item, str(item.get("source", "base_model")))
        if len(output) >= maximum:
            break
    for index, item in enumerate(output):
        if not item["candidate_id"]:
            item["candidate_id"] = f"{row.get('id')}:parent_{index}"
    return output


def query_derived_fallback(row: dict[str, Any]) -> str:
    """极端生成失败时从 Query 截取短语补位；不读取 Seed 或 Gold。"""

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", str(row.get("source_text", "")))
    title = " ".join(words[:12]).strip()
    return title if len(title.split()) >= 2 else "Research Study"


def load_source_rows(
    config: dict[str, Any], source: str, split: str
) -> list[dict[str, Any]]:
    """读取 Query/History；默认直接使用 retrieve 阶段，不依赖 seeds。"""

    if source:
        path = resolve_path(source)
        rows = read_jsonl(path)
        if rows and not rows[0].get("source_text"):
            originals = read_jsonl(stage_path(config, split, "retrieve"))
            by_id = {str(item.get("id", "")): item for item in originals}
            merged: list[dict[str, Any]] = []
            for row in rows:
                sample_id = str(row.get("id", row.get("sample_id", "")))
                original = by_id.get(sample_id)
                if original is None:
                    raise ValueError(f"无法从 {split} retrieved 数据补回 sample={sample_id}")
                item = dict(original)
                merged.append(item)
            rows = merged
    else:
        rows = read_jsonl(stage_path(config, split, "retrieve"))
    if not rows:
        raise ValueError(f"没有输入行：source={source!r} split={split}")
    return rows


def build_parent_pool(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    destination: Path,
    *,
    pool_source: str,
    limit: int,
    ablation: str,
) -> list[dict[str, Any]]:
    """生成或从 checkpoint 续跑 Parent Pool。"""

    if limit > 0:
        rows = rows[:limit]
    settings = config["crossover_sft"]
    sample_count = (
        0 if ablation == "single_parent" else int(settings["sample_parent_count"])
    )
    include_history = ablation != "no_history"
    protocol = (
        f"base_model_direct_fixed{1 + sample_count}_{ablation}_gold_crossover_v4"
        if pool_source == "base_model"
        else "synthetic_query_only_crossover_smoke_v1"
    )
    previous = read_jsonl(destination) if destination.exists() else []
    cache = {str(item.get("id", "")): item for item in previous}
    output: list[dict[str, Any]] = []
    sampler: BaseParentSampler | None = None
    seed = int(config["training"]["seed"])
    maximum_history = int(config["sft_data"]["maximum_history_records"])
    checkpoint_every = int(settings["checkpoint_every"])
    try:
        for index, row in enumerate(rows, 1):
            sample_id = str(row.get("id", row.get("sample_id", "")))
            cached = cache.get(sample_id)
            if cached and cached.get("parent_pool_protocol") == protocol:
                output.append(cached)
                print(
                    f"parent pool {index}/{len(rows)} sample={sample_id} "
                    f"source=cache parents={len(cached.get('parent_pool', []))}",
                    flush=True,
                )
                continue
            pool: list[dict[str, Any]] = []
            if pool_source == "synthetic_smoke":
                base = query_derived_fallback(row)
                for candidate_index in range(1 + sample_count):
                    append_parent(
                        pool,
                        {"text": f"{base} Variant {candidate_index + 1}"},
                        "synthetic_query_only_smoke",
                    )
            elif pool_source == "base_model":
                if sampler is None:
                    sampler = BaseParentSampler(config)
                prompt = base_prompt(
                    row, maximum_history, include_history=include_history
                )
                for value in sampler.generate(
                    prompt,
                    1,
                    sample=False,
                    temperature=float(settings["temperature"]),
                    top_p=float(settings["top_p"]),
                    seed=seed + index,
                ):
                    append_parent(pool, value, "base_greedy")
                # 固定采样 4 次；没有过采样、best-of、奖励排序或语义门控。
                for value in sampler.generate(
                    prompt,
                    sample_count,
                    sample=True,
                    temperature=float(settings["temperature"]),
                    top_p=float(settings["top_p"]),
                    seed=seed * 100003 + index,
                ):
                    append_parent(pool, value, "base_sample")
            else:
                raise ValueError(f"未知 pool_source={pool_source}")
            requested = 1 + sample_count
            if not pool:
                append_parent(
                    pool,
                    {"text": query_derived_fallback(row)},
                    "fallback_query_derived",
                )
            # 空值或明显格式污染不会进入模型；对应槽位复制 greedy/首个有效
            # Parent，以保证固定 K。重复是协议允许的，不代表新增信息。
            while len(pool) < requested:
                pool.append(
                    {
                        "candidate_id": "",
                        "text": pool[0]["text"],
                        "source": "fallback_copy_of_parent_0",
                    }
                )
            pool = pool[:requested]
            for parent_index, parent in enumerate(pool):
                parent["candidate_id"] = f"{sample_id}:base_parent_{parent_index}"
            item = dict(row)
            item["id"] = sample_id
            item["parent_pool"] = pool
            item["parent_pool_protocol"] = protocol
            item["requested_parent_count"] = requested
            item["unique_parent_count"] = len(
                {normalized(parent["text"]) for parent in pool}
            )
            output.append(item)
            print(
                f"parent pool {index}/{len(rows)} sample={sample_id} "
                f"requested={requested} parents={len(pool)} "
                f"unique={item['unique_parent_count']}",
                flush=True,
            )
            if checkpoint_every > 0 and index % checkpoint_every == 0:
                write_jsonl(destination, output)
        write_jsonl(destination, output)
        return output
    finally:
        if sampler is not None:
            sampler.close()


def crossover_prompt(
    row: dict[str, Any],
    parents: list[dict[str, str]],
    maximum_history: int,
    *,
    include_history: bool,
) -> str:
    payload = {
        "current_input": str(row.get("source_text", "")),
        "retrieved_history": (
            visible_history(row, maximum_history) if include_history else []
        ),
        "parent_candidates": [
            {"candidate_id": item["candidate_id"], "text": item["text"]}
            for item in parents
        ],
    }
    return (
        "You are a personalized multi-parent crossover editor. Produce the best final "
        "academic paper title supported by CURRENT_INPUT. Compare all PARENT_CANDIDATES. "
        "You may preserve a strong candidate, repair one candidate, or recombine compatible "
        "useful content from multiple candidates. Use RETRIEVED_HISTORY only for applicable "
        "recurring user preferences. Candidate order does not indicate quality. Do not invent "
        "facts. Return exactly one line containing only the final title; do not output JSON, "
        "Markdown, labels, or explanations.\n\nPAYLOAD:\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n\nFINAL TITLE:\n"
    )


def ordered(
    parents: list[dict[str, str]], sample_id: str, variant: str, seed: int
) -> list[dict[str, str]]:
    output = list(parents)
    stable_rng(sample_id, variant, seed).shuffle(output)
    return output


def compile_sft(
    pool_rows: list[dict[str, Any]],
    config: dict[str, Any],
    smoke: bool,
    ablation: str,
) -> list[dict[str, Any]]:
    settings = config["crossover_sft"]
    seed = int(config["training"]["seed"])
    fraction = float(config["sft_data"]["validation_fraction"])
    maximum_history = int(config["sft_data"]["maximum_history_records"])
    maximum_parents = (
        1 if ablation == "single_parent" else int(settings["total_parent_count"])
    )
    dropout_copies = (
        0 if ablation == "single_parent" else int(settings["candidate_dropout_copies"])
    )
    minimum_parents = 1 if ablation == "single_parent" else 2
    include_history = ablation != "no_history"
    examples: list[dict[str, Any]] = []
    for row in pool_rows:
        sample_id = str(row.get("id", ""))
        gold = clean_title(row.get("target", ""))
        parents = usable_parents(row, maximum_parents)
        if not sample_id or not gold or len(parents) < minimum_parents:
            continue
        split = deterministic_split(sample_id, fraction)
        variants: list[tuple[str, list[dict[str, str]]]] = [
            ("full", ordered(parents, sample_id, "full", seed))
        ]
        if split == "train" and len(parents) >= 3:
            for copy_index in range(dropout_copies):
                name = f"dropout_{copy_index}"
                rng = stable_rng(sample_id, name, seed)
                selected = rng.sample(parents, rng.randint(2, len(parents) - 1))
                rng.shuffle(selected)
                variants.append((name, selected))
        for name, selected in variants:
            examples.append(
                {
                    "example_id": f"{sample_id}:gold_crossover:{name}",
                    "sample_id": sample_id,
                    "user_id": str(row.get("user_id", "")),
                    "task": (
                        "gold_single_parent_editor"
                        if ablation == "single_parent"
                        else "gold_multi_parent_crossover"
                    ),
                    "operation_type": (
                        "single_parent_editor"
                        if ablation == "single_parent"
                        else "crossover"
                    ),
                    "ablation": ablation,
                    "variant": name,
                    "parent_count": len(selected),
                    "unique_parent_count": len(
                        {normalized(item["text"]) for item in selected}
                    ),
                    "parents": selected,
                    "prompt": crossover_prompt(
                        row,
                        selected,
                        maximum_history,
                        include_history=include_history,
                    ),
                    "target": gold,
                    "output": gold,
                    "trace_text": "",
                    "output_text": gold,
                    "sample_weight": 1.0,
                    "split": split,
                    "student_prompt_sees_gold": False,
                    "gold_used_as_output_supervision": True,
                    "teacher_used": False,
                }
            )
    if smoke and examples and not any(item["split"] == "validation" for item in examples):
        validation_id = examples[-1]["sample_id"]
        for item in examples:
            if item["sample_id"] == validation_id:
                item["split"] = "validation"
    return examples


def save_sft(
    data_dir: Path,
    pool_rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    pool_source: str,
    smoke: bool,
    ablation: str,
) -> dict[str, Any]:
    train = [item for item in examples if item["split"] == "train"]
    validation = [item for item in examples if item["split"] == "validation"]
    if not train or not validation:
        raise ValueError(
            f"SFT train/validation 不能为空：train={len(train)} validation={len(validation)}"
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(data_dir / "01_parent_pool.jsonl", pool_rows)
    write_jsonl(data_dir / "02_crossover_examples.jsonl", examples)
    write_jsonl(data_dir / "all_sft.jsonl", examples)
    write_jsonl(data_dir / "train_sft.jsonl", train)
    write_jsonl(data_dir / "validation_sft.jsonl", validation)
    full = [item for item in examples if item["variant"] == "full"]
    report = {
        "protocol": f"gold_supervised_fixed_width_noseed_crossover_{ablation}_v4",
        "ablation": ablation,
        "source_queries": len(pool_rows),
        "usable_queries": len(full),
        "query_coverage": len(full) / max(len(pool_rows), 1),
        "requested_parents": (
            "1 greedy" if ablation == "single_parent" else "1 greedy + 4 sampling"
        ),
        "actual_parent_distribution": {
            str(number): sum(item["parent_count"] == number for item in full)
            for number in range(1, 6)
        },
        "unique_parent_distribution": {
            str(number): sum(item["unique_parent_count"] == number for item in full)
            for number in range(1, 6)
        },
        "examples": len(examples),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "dropout_examples": sum(item["variant"] != "full" for item in examples),
        "pool_source": pool_source,
        "teacher_used": False,
        "semantic_gate_used": False,
        "student_prompt_sees_gold": False,
        "gold_used_only_as_output_supervision": True,
        "history_in_parent_generation": ablation != "no_history",
        "history_in_student_prompt": ablation != "no_history",
        "formal_result_eligible": pool_source == "base_model" and not smoke,
        "seed_parent_used": False,
    }
    write_json(data_dir / "manifest.json", report)
    print(f"Crossover SFT data -> {data_dir}; report={report}", flush=True)
    return report


def train(
    config: dict[str, Any], data_dir: Path, run_dir: Path, max_steps: int | None
) -> dict[str, Any]:
    local = copy.deepcopy(config)
    local.setdefault("paths", {})["sft_dir"] = str(data_dir)
    local["paths"]["editor_output_dir"] = str(run_dir / "editor")
    trainer = import_module("06_train_editor_lora.py", "gold_crossover_trainer")
    return trainer.train(local, max_steps_override=max_steps)


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        score(str(row.get("prediction", "")), str(row.get("target", "")))
        for row in rows
    ]
    bleu = corpus_bleu(
        [str(row.get("prediction", "")) for row in rows],
        [str(row.get("target", "")) for row in rows],
    )
    users = {str(row.get("user_id", "")) for row in rows if row.get("user_id")}
    return {
        "protocol": "gold_multi_parent_crossover_single_output_v1",
        "users": len(users),
        "queries": len(rows),
        "valid_predictions": sum(bool(row.get("prediction")) for row in rows),
        "rouge_1": statistics.mean(item["rouge_1"] for item in metrics),
        "rouge_l": statistics.mean(item["rouge_l"] for item in metrics),
        "sacrebleu": float(bleu["score"]),
    }


def evaluate(
    config: dict[str, Any],
    pool_rows: list[dict[str, Any]],
    adapter: Path,
    run_dir: Path,
    ablation: str,
) -> dict[str, Any]:
    settings = config["crossover_sft"]
    seed = int(config["training"]["seed"])
    maximum_history = int(config["sft_data"]["maximum_history_records"])
    maximum_parents = (
        1 if ablation == "single_parent" else int(settings["total_parent_count"])
    )
    minimum_parents = 1 if ablation == "single_parent" else 2
    include_history = ablation != "no_history"
    specs: list[tuple[dict[str, Any], list[dict[str, str]]]] = []
    prompts: list[str] = []
    for row in pool_rows:
        sample_id = str(row.get("id", ""))
        parents = usable_parents(row, maximum_parents)
        if len(parents) < minimum_parents:
            continue
        parents = ordered(parents, sample_id, "test_full", seed)
        specs.append((row, parents))
        prompts.append(
            crossover_prompt(
                row,
                parents,
                maximum_history,
                include_history=include_history,
            )
        )
    generator = AdapterGenerator(config, adapter)
    try:
        predictions = generator.generate(
            prompts, int(config["evaluation"]["prediction_batch_size"])
        )
    finally:
        generator.close()
    generated = {
        str(row.get("id", "")): (parents, prediction)
        for (row, parents), prediction in zip(specs, predictions)
    }
    rows: list[dict[str, Any]] = []
    for source in pool_rows:
        sample_id = str(source.get("id", ""))
        value = generated.get(sample_id)
        if value is None:
            parents = usable_parents(source, maximum_parents)
            prediction = parents[0]["text"] if parents else ""
            error = "insufficient_parents_fallback_greedy"
        else:
            parents, prediction = value
            error = None if prediction else "empty_prediction"
        rows.append(
            {
                "id": sample_id,
                "user_id": str(source.get("user_id", "")),
                "source_text": str(source.get("source_text", "")),
                "target": clean_title(source.get("target", "")),
                "parents": parents,
                "parent_count": len(parents),
                "unique_parent_count": len(
                    {normalized(parent["text"]) for parent in parents}
                ),
                "prediction": prediction,
                "error": error,
            }
        )
    output_dir = run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "test_predictions.jsonl", rows)
    report = evaluate_rows(rows)
    report.update(
        {
            "aggregation_queries": len(specs),
            "aggregation_coverage": len(specs) / max(len(pool_rows), 1),
            "fallback_queries": len(pool_rows) - len(specs),
            "teacher_used": False,
            "ablation": ablation,
            "history_used": include_history,
        }
    )
    write_json(output_dir / "global_test_report.json", report)
    print(f"Crossover evaluation -> {output_dir}; report={report}", flush=True)
    return report


def verify_formal_test(
    rows: list[dict[str, Any]], config: dict[str, Any], test_limit: int, smoke: bool
) -> None:
    if smoke or test_limit > 0:
        return
    expected_queries = int(config["evaluation"]["expected_queries"])
    expected_users = int(config["evaluation"]["expected_users"])
    users = {str(row.get("user_id", "")) for row in rows if row.get("user_id")}
    if len(rows) != expected_queries or len(users) != expected_users:
        raise ValueError(
            "正式评估口径错误："
            f"actual={len(users)} users/{len(rows)} queries，"
            f"expected={expected_users}/{expected_queries}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gold-supervised 多 Parent Crossover")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--stage", choices=("pool", "build", "train", "eval", "all"), default="all"
    )
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--parent-records", default=DEFAULT_PARENT_RECORDS)
    parser.add_argument("--test-parent-records", default="")
    parser.add_argument(
        "--shared-parent-pool-dir",
        default="",
        help="独立共享 Parent Pool 目录；读取 train_parent_pool/test_parent_pool",
    )
    parser.add_argument(
        "--pool-source", choices=("base_model", "synthetic_smoke"), default="base_model"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--ablation",
        choices=("main", "single_parent", "no_history"),
        default="main",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = configure(load_config(args.config))
    stamp = args.run_name or (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_gold_crossover_{args.ablation}_v2"
    )
    data_dir = resolve_path(
        args.data_dir
        or f"/data/liux/MEVO_global_cot/dataset/editor_sets/{stamp}"
    )
    run_dir = resolve_path(
        args.run_dir or f"/data/liux/MEVO_global_cot/result/{stamp}"
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"GOLD_CROSSOVER_PATHS data_dir={data_dir} run_dir={run_dir} "
        f"ablation={args.ablation} teacher_used=false",
        flush=True,
    )
    result: dict[str, Any] = {}

    if args.stage in {"pool", "all"}:
        if args.shared_parent_pool_dir:
            shared_dir = resolve_path(args.shared_parent_pool_dir)
            shared_manifest = shared_dir / "manifest.json"
            if shared_manifest.exists():
                manifest = json.loads(shared_manifest.read_text(encoding="utf-8"))
                if bool(manifest.get("seed_parent_used", False)):
                    raise ValueError(f"共享 Parent Pool 含 Seed Parent，拒绝复用：{shared_dir}")
                if manifest.get("history_used") is False and args.ablation != "no_history":
                    raise ValueError("main Crossover 不能复用 no_history Parent Pool")
            shared_path = shared_dir / "train_parent_pool.jsonl"
            if not shared_path.exists():
                raise FileNotFoundError(f"共享训练 Parent Pool 不存在：{shared_path}")
            pool_rows = read_jsonl(shared_path)
            if args.limit > 0:
                pool_rows = pool_rows[: args.limit]
            write_jsonl(data_dir / "01_parent_pool.jsonl", pool_rows)
            print(f"reuse shared train Parent Pool={shared_path}", flush=True)
        else:
            parent_source = args.parent_records
            if parent_source and not resolve_path(parent_source).exists():
                print(
                    f"PARENT_RECORDS_FALLBACK missing={parent_source}; use=train retrieved data",
                    flush=True,
                )
                parent_source = ""
            train_rows = load_source_rows(config, parent_source, "train")
            pool_rows = build_parent_pool(
                train_rows,
                config,
                data_dir / "01_parent_pool.jsonl",
                pool_source=args.pool_source,
                limit=args.limit,
                ablation=args.ablation,
            )
        result["pool"] = {"queries": len(pool_rows)}
    if args.stage in {"build", "all"}:
        pool_path = data_dir / "01_parent_pool.jsonl"
        if args.shared_parent_pool_dir and not pool_path.exists():
            shared_path = resolve_path(args.shared_parent_pool_dir) / "train_parent_pool.jsonl"
            if not shared_path.exists():
                raise FileNotFoundError(f"共享训练 Parent Pool 不存在：{shared_path}")
            shared_rows = read_jsonl(shared_path)
            if args.limit > 0:
                shared_rows = shared_rows[: args.limit]
            write_jsonl(pool_path, shared_rows)
            print(f"reuse shared train Parent Pool={shared_path}", flush=True)
        if not pool_path.exists():
            raise FileNotFoundError(f"先运行 --stage pool：{pool_path}")
        pool_rows = read_jsonl(pool_path)
        if args.limit > 0:
            pool_rows = pool_rows[: args.limit]
        result["build"] = save_sft(
            data_dir,
            pool_rows,
            compile_sft(pool_rows, config, args.smoke, args.ablation),
            args.pool_source,
            args.smoke,
            args.ablation,
        )
    if args.stage in {"train", "all"}:
        result["train"] = train(
            config,
            data_dir,
            run_dir,
            args.max_steps if args.max_steps > 0 else None,
        )
    if args.stage in {"eval", "all"}:
        if args.shared_parent_pool_dir:
            shared_dir = resolve_path(args.shared_parent_pool_dir)
            shared_manifest = shared_dir / "manifest.json"
            if shared_manifest.exists():
                manifest = json.loads(shared_manifest.read_text(encoding="utf-8"))
                if bool(manifest.get("seed_parent_used", False)):
                    raise ValueError(f"共享 Parent Pool 含 Seed Parent，拒绝复用：{shared_dir}")
                if manifest.get("history_used") is False and args.ablation != "no_history":
                    raise ValueError("main Crossover 不能复用 no_history Parent Pool")
            shared_path = shared_dir / "test_parent_pool.jsonl"
            if not shared_path.exists():
                raise FileNotFoundError(f"共享测试 Parent Pool 不存在：{shared_path}")
            test_pool = read_jsonl(shared_path)
            if args.test_limit > 0:
                test_pool = test_pool[: args.test_limit]
            test_rows = test_pool
            write_jsonl(data_dir / "test_parent_pool.jsonl", test_pool)
        else:
            test_rows = load_source_rows(config, args.test_parent_records, "test")
            if args.test_limit > 0:
                test_rows = test_rows[: args.test_limit]
            test_pool = build_parent_pool(
                test_rows,
                config,
                data_dir / "test_parent_pool.jsonl",
                pool_source=args.pool_source,
                limit=0,
                ablation=args.ablation,
            )
        verify_formal_test(test_rows, config, args.test_limit, args.smoke)
        adapter = run_dir / "editor" / "final_adapter"
        if not adapter.exists():
            raise FileNotFoundError(f"缺少训练后的 Adapter：{adapter}")
        result["eval"] = evaluate(
            config, test_pool, adapter, run_dir, args.ablation
        )
    write_json(run_dir / "pipeline_report.json", result)
    print(f"GOLD_MULTI_PARENT_CROSSOVER_DONE={result}", flush=True)


if __name__ == "__main__":
    main()
