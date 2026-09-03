#!/usr/bin/env python3
"""构建并训练 Crossover-only Editor SFT。

正式协议
--------
1. 每个 Query 使用与 Base 评估一致的 Llama2-7B Prompt 生成 Parent Pool；
2. Parent A 固定为 greedy Parent；从 best-of-N 采样池中先过滤低质量候选，
   再选择质量接近 A、且提供 Query 支持增量信息的 Parent B；
3. 训练阶段可让 Teacher 看 Gold，只判断 Pair 是否真的具有融合价值；
4. Student Prompt 只包含 Query、Top-k History、Parent A/B，输出为一行 Gold；
5. Prompt token 全部 mask，直接复用根目录 ``06_train_editor_lora.py``。

为了快速验证代码，``--pool-source existing_candidates --teacher-mode heuristic``
可复用 seeds 中的已有候选。该模式只用于 smoke，manifest 会明确标记，不能作为
正式实验结果。正式实验必须使用 ``--pool-source base_model --teacher-mode api``。
"""

from __future__ import annotations

import argparse
import copy
import difflib
import importlib.util
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
CODE = HERE.parent
PROJECT = CODE.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(PROJECT))

from common.concurrency import BoundedJobError, run_bounded  # noqa: E402
from common.metrics import corpus_bleu, score  # noqa: E402
from pipeline_common import (  # noqa: E402
    build_base_text_prompt,
    build_editor_prompt,
    deterministic_split,
    load_config,
    read_jsonl,
    resolve_path,
    stage_path,
    teacher_client,
    truncate_prompt_ids,
    visible_history,
    write_json,
    write_jsonl,
)


DEFAULT_PARENT_RECORDS = (
    "/data/liux/MEVO_global_cot/dataset/editor_sets/"
    "20260902_094902_direct_parent_gold_full/all_sft.jsonl"
)

PARENT_META_TEXT = re.compile(
    r"(?:user title|title examples?|write one concise|output only|parent title|"
    r"required_schema|candidate_id|payload|abstract and user|"
    r"\\end\{|```|\"output\"\s*:|@[a-z]+\s*\{)",
    re.IGNORECASE,
)
PARENT_PROSE_START = re.compile(
    r"^(?:in this paper|this paper|we propose|we present|we show|our (?:method|work))\b",
    re.IGNORECASE,
)
NEGATIVE_CONTRIBUTION = re.compile(
    r"\b(?:no useful|not useful|irrelevant|unrelated|placeholder|redundant|identical|"
    r"adds? no|does not (?:add|contribute)|no additional|discard)\b",
    re.IGNORECASE,
)
MERGE_CONFLICT_ACTION = re.compile(
    r"\b(?:keep (?:only )?parent|discard parent|ignore parent|parent_[ab] alone|"
    r"use only parent|replace (?:the )?(?:generic )?placeholder|no merge)\b",
    re.IGNORECASE,
)
CONTENT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "based", "by", "for", "from",
    "in", "into", "is", "of", "on", "or", "the", "to", "using", "via", "with",
    "approach", "method", "novel", "paper", "study", "new", "results",
}


def content_tokens(value: str) -> set[str]:
    """用于 target-blind 内容覆盖检查的轻量词集合。

    这里刻意不使用 Gold，也不引入 LaMP-5 专属字段。长度为 1--2 的词和常见
    功能词通常不能证明候选带来了有效信息，因此不参与互补性判断。
    """

    return {
        token
        for token in word_tokens(value)
        if len(token) >= 3 and token not in CONTENT_STOPWORDS
    }


def same_content_family(left: str, right: str) -> bool:
    """判断两个内容词是否只是常见词形变化，而非真正的新增信息。"""

    if left == right:
        return True
    if min(len(left), len(right)) < 5:
        return False
    common_prefix = 0
    for a, b in zip(left, right):
        if a != b:
            break
        common_prefix += 1
    if common_prefix >= 6:
        return True
    return difflib.SequenceMatcher(None, left, right).ratio() >= 0.72


def incremental_query_tokens(
    candidate_terms: set[str], anchor_terms: set[str], query_terms: set[str]
) -> set[str]:
    """返回 B 中受 Query 支持、且不是 A 中已有词形变体的内容词。"""

    return {
        token
        for token in candidate_terms & query_terms
        if not any(same_content_family(token, anchor) for anchor in anchor_terms)
    }


def clean_title(value: Any) -> str:
    """按统一纯文本协议提取第一行标题。"""

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


def word_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value).casefold())


def token_jaccard(left: str, right: str) -> float:
    a, b = set(word_tokens(left)), set(word_tokens(right))
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _load_module(filename: str, name: str):
    path = CODE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_prompt(row: dict[str, Any], maximum_history: int) -> str:
    """复用已有 Parent 生成 Prompt，保证训练/测试 Parent 分布一致。"""

    cached = str(row.get("parent_generation_prompt", "")).strip()
    if cached:
        return cached
    candidates = list(row.get("candidates", []))
    if not candidates:
        # test retrieve/seeds 极少数缺 seed 时，用已有 Parent 作为起点；若连 Parent
        # 也没有，调用方会给出明确错误，不能悄悄生成空标题。
        parent = clean_title(row.get("parent", ""))
        if not parent:
            raise ValueError(f"sample={row.get('id')} 缺少生成 Parent 所需 seed")
        candidates = [{"candidate_id": f"{row.get('id')}:cached", "text": parent}]
    return build_base_text_prompt(row, candidates[0], maximum_history)


class BaseParentSampler:
    """惰性加载 Llama2-7B，并为每个 Query 生成可复现 Parent Pool。"""

    def __init__(self, config: dict[str, Any]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = resolve_path(config["model"]["path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Base 模型不存在: {model_path}")
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
        self.max_length = int(config.get("training", {}).get("max_length", 2048))
        settings = config.get("crossover_sft", {})
        self.max_new_tokens = int(
            settings.get("max_new_tokens", config.get("evaluation", {}).get("max_new_tokens", 64))
        )
        if self.max_new_tokens >= self.max_length:
            raise ValueError("Parent 生成 token 数必须小于模型上下文上限")
        self.prompt_max_length = self.max_length - self.max_new_tokens

    def generate(
        self,
        prompt: str,
        count: int,
        *,
        sample: bool,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> list[dict[str, Any]]:
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
        # 当前 hydra 环境的 Transformers 版本不接受 generate(generator=...)。
        # 在单进程 Parent 构建阶段直接固定 CPU/CUDA RNG，效果等价且兼容旧版本。
        self.torch.manual_seed(int(seed))
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(int(seed))
        kwargs: dict[str, Any] = {
            **encoded,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": bool(sample),
            "num_return_sequences": int(count),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
            # Llama2-7B FP16 在极长 Top-8 Prompt 上随机采样时偶发 Inf/NaN
            # logits。Transformers 的标准处理器会在 softmax 前清理并重新归一化。
            "remove_invalid_values": True,
            "renormalize_logits": True,
        }
        if sample:
            kwargs.update(temperature=float(temperature), top_p=float(top_p), top_k=50)
        with self.torch.inference_mode():
            generated = self.model.generate(**kwargs)
        prompt_length = encoded["input_ids"].shape[1]
        sequences = generated.sequences
        transition = self.model.compute_transition_scores(
            sequences, generated.scores, normalize_logits=True
        )
        output: list[dict[str, Any]] = []
        for index, sequence in enumerate(sequences):
            raw = self.tokenizer.decode(
                sequence[prompt_length:], skip_special_tokens=True
            )
            title = clean_title(raw)
            values = transition[index]
            # EOS 后 compute_transition_scores 可能包含填充零值；只对实际非零
            # transition 求均值，作为 target-blind MMR 的弱质量代理。
            active = values[values.ne(0)]
            logprob = float(active.mean().item()) if active.numel() else -100.0
            output.append({"text": title, "raw_response": raw, "mean_logprob": logprob})
        return output

    def close(self) -> None:
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _append_unique(
    pool: list[dict[str, Any]], text: str, source: str, score_value: float | None = None
) -> None:
    title = clean_title(text)
    if score_value is not None and not math.isfinite(float(score_value)):
        return
    if not title or len(word_tokens(title)) < 3 or len(word_tokens(title)) > 30:
        return
    if PARENT_META_TEXT.search(title) or PARENT_PROSE_START.search(title):
        return
    key = normalized(title)
    if not key or any(normalized(item["text"]) == key for item in pool):
        return
    pool.append(
        {
            "candidate_id": "",  # 在 Pool 完成后统一写入稳定 ID
            "text": title,
            "source": source,
            "mean_logprob": score_value,
        }
    )


def build_parent_pools(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    destination: Path,
    *,
    pool_source: str,
    extra_parents: int,
    limit: int,
) -> list[dict[str, Any]]:
    """生成或续跑 Parent Pool；每完成若干 Query 就落盘。"""

    if limit > 0:
        rows = rows[:limit]
    cached_rows = read_jsonl(destination) if destination.exists() else []
    cached = {str(item.get("id", "")): item for item in cached_rows}
    output: list[dict[str, Any]] = []
    sampler: BaseParentSampler | None = None
    settings = config.get("crossover_sft", {})
    temperature = float(settings.get("temperature", 0.8))
    top_p = float(settings.get("top_p", 0.9))
    seed = int(config.get("training", {}).get("seed", 42))
    checkpoint_every = int(settings.get("checkpoint_every", 20))
    maximum_history = int(config.get("sft_data", {}).get("maximum_history_records", 8))
    pool_protocol = (
        "base_model_best_of_n_quality_v3"
        if pool_source == "base_model"
        else "existing_candidates_smoke"
    )
    try:
        for index, row in enumerate(rows, 1):
            sample_id = str(row.get("id", row.get("sample_id", "")))
            if (
                sample_id in cached
                and cached[sample_id].get("parent_pool_protocol") == pool_protocol
                and len(cached[sample_id].get("parent_pool", [])) >= 2
            ):
                output.append(cached[sample_id])
                print(
                    f"parent pool {index}/{len(rows)} sample={sample_id} source=cache "
                    f"parents={len(cached[sample_id]['parent_pool'])}",
                    flush=True,
                )
                continue
            item = dict(row)
            pool: list[dict[str, Any]] = []
            if pool_source == "existing_candidates":
                # 仅供 smoke：这些候选可能来自 Teacher seed，不属于正式 Base 分布。
                _append_unique(
                    pool,
                    row.get("parent", row.get("prediction", "")),
                    str(row.get("parent_source", "cached_greedy_base")),
                )
                for candidate in row.get("candidates", []):
                    _append_unique(pool, candidate.get("text", ""), "existing_candidate_smoke")
            elif pool_source == "base_model":
                if sampler is None:
                    sampler = BaseParentSampler(config)
                prompt = _base_prompt(row, maximum_history)
                # A 必须重新由当前 Base/Prompt greedy 生成，使它和采样候选具有
                # 同口径 mean_logprob。旧记录中的 Parent 虽也来自 Base，但没有
                # 保存 log-prob，不能用于“质量接近 A”的因果筛选。
                greedy = sampler.generate(
                    prompt,
                    1,
                    sample=False,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed + index,
                )
                for value in greedy:
                    _append_unique(
                        pool, value["text"], "base_greedy", value["mean_logprob"]
                    )

                # best-of-N 候选分小 batch 生成，避免 4096 上下文下同时返回十几个
                # 序列造成 KV cache OOM。多生成少量候选用于抵消重复输出。
                sample_batch_size = max(1, int(settings.get("sampling_batch_size", 4)))
                requested = max(1, int(extra_parents))
                attempts = requested + max(2, requested // 3)
                generated_count = 0
                batch_index = 0
                while generated_count < attempts and len(pool) < requested + 1:
                    count = min(sample_batch_size, attempts - generated_count)
                    sampled = sampler.generate(
                        prompt,
                        count,
                        sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        seed=seed * 100003 + index + batch_index * 10000019,
                    )
                    for value in sampled:
                        _append_unique(
                            pool,
                            value["text"],
                            "base_best_of_n_sample",
                            value["mean_logprob"],
                        )
                    generated_count += count
                    batch_index += 1
            else:
                raise ValueError(f"未知 pool_source={pool_source}")
            for pool_index, candidate in enumerate(pool):
                candidate["candidate_id"] = f"{sample_id}:base_parent_{pool_index}"
            item["id"] = sample_id
            item["parent_pool"] = pool
            item["parent_pool_protocol"] = pool_protocol
            output.append(item)
            print(
                f"parent pool {index}/{len(rows)} sample={sample_id} "
                f"source={pool_source} parents={len(pool)}",
                flush=True,
            )
            if checkpoint_every > 0 and index % checkpoint_every == 0:
                write_jsonl(destination, output)
        write_jsonl(destination, output)
        return output
    finally:
        if sampler is not None:
            sampler.close()


def parent_quality_issues(
    item: dict[str, Any], row: dict[str, Any], settings: dict[str, Any], *, require_score: bool
) -> list[str]:
    """执行 target-blind Parent 硬过滤，避免用“多样性”奖励格式垃圾。"""

    title = clean_title(item.get("text", ""))
    words = word_tokens(title)
    issues: list[str] = []
    minimum_words = int(settings.get("minimum_parent_words", 3))
    maximum_words = int(settings.get("maximum_parent_words", 30))
    if not title or not minimum_words <= len(words) <= maximum_words:
        issues.append("invalid_length")
    if PARENT_META_TEXT.search(title):
        issues.append("prompt_or_format_leak")
    if PARENT_PROSE_START.search(title):
        issues.append("abstract_prose")
    query_terms = content_tokens(str(row.get("source_text", "")))
    title_terms = content_tokens(title)
    minimum_overlap = int(settings.get("minimum_query_overlap_tokens", 1))
    if len(query_terms & title_terms) < minimum_overlap:
        issues.append("not_grounded_in_query")
    grounding_ratio = len(query_terms & title_terms) / max(len(title_terms), 1)
    if grounding_ratio < float(settings.get("minimum_query_grounding_ratio", 0.25)):
        issues.append("low_query_grounding_ratio")
    value = item.get("mean_logprob")
    if require_score:
        if value is None or not math.isfinite(float(value)):
            issues.append("missing_or_nonfinite_logprob")
        elif float(value) < float(settings.get("minimum_parent_mean_logprob", -2.5)):
            issues.append("low_logprob")
    return issues


def select_target_blind_pair(
    row: dict[str, Any], settings: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """质量约束下选择 Query 支持、相对 greedy A 有增量信息的 B。

    Parent 构造阶段完全不读取 Gold。若没有真正互补且质量足够接近 A 的候选，
    返回 ``None``，由后续流程降级为 keep-A，而不是制造伪 Crossover。
    """

    settings = settings or {}
    pool = [item for item in row.get("parent_pool", []) if clean_title(item.get("text", ""))]
    if len(pool) < 2:
        return None
    allow_unscored = str(row.get("parent_pool_protocol", "")).startswith(
        "existing_candidates"
    )
    parent_a = pool[0]
    # A 的定义始终是第一个 greedy 输出。若 greedy 本身无效，宁可不构造该
    # Crossover，也不能悄悄把某个采样候选改称 A。
    if parent_quality_issues(
        parent_a, row, settings, require_score=not allow_unscored
    ):
        return None
    alternatives = [
        item
        for item in pool[1:]
        if item is not parent_a
        and not parent_quality_issues(
            item, row, settings, require_score=not allow_unscored
        )
        if normalized(item["text"]) != normalized(parent_a["text"])
        and token_jaccard(item["text"], parent_a["text"]) < 0.92
    ]
    if not alternatives:
        return None

    query_terms = content_tokens(str(row.get("source_text", "")))
    a_terms = content_tokens(parent_a["text"])
    a_value = parent_a.get("mean_logprob")
    a_logprob = (
        float(a_value)
        if a_value is not None and math.isfinite(float(a_value))
        else None
    )
    maximum_gap = float(settings.get("maximum_logprob_gap_from_a", 1.5))
    minimum_incremental = int(settings.get("minimum_incremental_query_tokens", 1))

    eligible: list[tuple[dict[str, Any], set[str], float]] = []
    for item in alternatives:
        value = item.get("mean_logprob")
        item_logprob = (
            float(value)
            if value is not None and math.isfinite(float(value))
            else None
        )
        if a_logprob is not None and item_logprob is not None:
            if item_logprob < a_logprob - maximum_gap:
                continue
        b_terms = content_tokens(item["text"])
        incremental = incremental_query_tokens(b_terms, a_terms, query_terms)
        if len(incremental) < minimum_incremental:
            continue
        grounding = len(b_terms & query_terms) / max(len(b_terms), 1)
        eligible.append((item, incremental, grounding))
    if not eligible:
        return None

    def quality(item: dict[str, Any]) -> float:
        value = item.get("mean_logprob")
        if value is None or not math.isfinite(float(value)):
            return 0.5
        floor = float(settings.get("minimum_parent_mean_logprob", -2.5))
        ceiling = float(settings.get("high_quality_parent_mean_logprob", -0.25))
        if ceiling <= floor:
            raise ValueError("high_quality_parent_mean_logprob 必须大于 minimum_parent_mean_logprob")
        return min(1.0, max(0.0, (float(value) - floor) / (ceiling - floor)))

    def pair_score(item: dict[str, Any], incremental: set[str], grounding: float) -> float:
        diversity = 1.0 - token_jaccard(parent_a["text"], item["text"])
        complement = min(1.0, len(incremental) / 3.0)
        return 0.50 * quality(item) + 0.25 * complement + 0.15 * grounding + 0.10 * diversity

    parent_b, incremental, grounding = max(
        eligible, key=lambda value: pair_score(value[0], value[1], value[2])
    )
    return {
        "parent_a": parent_a,
        "parent_b": parent_b,
        "selection": {
            "protocol": "best_of_n_quality_complement_v3",
            "token_jaccard": round(token_jaccard(parent_a["text"], parent_b["text"]), 6),
            "selection_score": round(pair_score(parent_b, incremental, grounding), 6),
            "parent_b_mean_logprob": parent_b.get("mean_logprob"),
            "parent_a_mean_logprob": parent_a.get("mean_logprob"),
            "logprob_gap_from_a": (
                round(a_logprob - float(parent_b["mean_logprob"]), 6)
                if a_logprob is not None and parent_b.get("mean_logprob") is not None
                else None
            ),
            "incremental_query_tokens": sorted(incremental),
            "incremental_query_token_count": len(incremental),
            "parent_b_query_grounding_ratio": round(grounding, 6),
            "candidate_pool_size": len(pool),
            "eligible_candidate_count": len(eligible),
            "quality_filter_passed": True,
            "gold_used": False,
        },
    }


def teacher_pair_prompt(row: dict[str, Any], pair: dict[str, Any], maximum_history: int) -> str:
    payload = {
        "current_input": str(row.get("source_text", "")),
        "retrieved_history": visible_history(row, maximum_history),
        "parent_a": pair["parent_a"]["text"],
        "parent_b": pair["parent_b"]["text"],
        "gold": clean_title(row.get("target", "")),
    }
    return (
        "You are a strict offline gate for a crossover text editor. GOLD is visible only for "
        "judging the two target-blind parents. Choose merge only when Parent A and Parent B each "
        "supply a distinct, useful, factually compatible contribution supported by CURRENT_INPUT "
        "and needed by GOLD. Redundancy, generic wording, formatting advice, placeholders, or a "
        "worse paraphrase are not contributions. If only A is useful choose keep_a; if only B is "
        "useful choose keep_b; if neither is usable choose reject. A merge_action must genuinely "
        "combine both stated contributions and must never say to keep, discard, ignore, or replace "
        "one parent. Do not invent contributions and do not output a final title. Return exactly "
        "one JSON object: "
        '{"decision":"merge|keep_a|keep_b|reject",'
        '"parent_a_contribution":"...","parent_b_contribution":"...",'
        '"merge_action":"..."}'
        "\n\nPAYLOAD:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def normalize_pair_annotation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Crossover Teacher 返回的不是 JSON object")
    decision = str(value.get("decision", "reject")).strip().lower()
    if decision not in {"merge", "keep_a", "keep_b", "reject"}:
        decision = "reject"
    return {
        "decision": decision,
        "parent_a_contribution": str(value.get("parent_a_contribution", "")).strip()[:300],
        "parent_b_contribution": str(value.get("parent_b_contribution", "")).strip()[:300],
        "merge_action": str(value.get("merge_action", "")).strip()[:500],
    }


def apply_pair_gate(
    row: dict[str, Any], pair: dict[str, Any], annotation: dict[str, Any], settings: dict[str, Any]
) -> dict[str, Any]:
    """用确定性一致性规则复核 Teacher，避免矛盾的 merge 标签进入 SFT。"""

    result = dict(annotation)
    decision = str(result.get("decision", "reject"))
    a_contribution = str(result.get("parent_a_contribution", "")).strip()
    b_contribution = str(result.get("parent_b_contribution", "")).strip()
    action = str(result.get("merge_action", "")).strip()
    errors: list[str] = []
    if decision == "merge":
        if not a_contribution:
            errors.append("missing_parent_a_contribution")
        if not b_contribution:
            errors.append("missing_parent_b_contribution")
        if NEGATIVE_CONTRIBUTION.search(a_contribution):
            errors.append("parent_a_described_as_noncontributing")
        if NEGATIVE_CONTRIBUTION.search(b_contribution):
            errors.append("parent_b_described_as_noncontributing")
        if not action:
            errors.append("missing_merge_action")
        elif MERGE_CONFLICT_ACTION.search(action):
            errors.append("merge_action_contradicts_merge")
        minimum_gold_score = float(settings.get("minimum_parent_gold_rouge_l", 0.05))
        gold = clean_title(row.get("target", ""))
        if score(pair["parent_a"]["text"], gold)["rouge_l"] < minimum_gold_score:
            errors.append("parent_a_has_no_gold_support")
        if score(pair["parent_b"]["text"], gold)["rouge_l"] < minimum_gold_score:
            errors.append("parent_b_has_no_gold_support")
    elif decision == "keep_a" and not a_contribution:
        errors.append("missing_kept_parent_a_contribution")
    elif decision == "keep_b" and not b_contribution:
        errors.append("missing_kept_parent_b_contribution")

    result["gate_version"] = "strict_teacher_program_gate_v2"
    result["gate_errors"] = errors
    result["gate_consistent"] = not errors
    result["training_accepted"] = bool(
        not errors and decision in {"merge", "keep_a", "keep_b"}
    )
    # accepted 保留旧字段语义：表示真正的 merge，而不是任意可训练决策。
    result["accepted"] = bool(not errors and decision == "merge")
    result["usable"] = result["training_accepted"]
    return result


def heuristic_pair_annotation(row: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    """只用于 smoke 的确定性门控，不作为正式 Teacher 判断。"""

    gold = set(word_tokens(clean_title(row.get("target", ""))))
    a = set(word_tokens(pair["parent_a"]["text"]))
    b = set(word_tokens(pair["parent_b"]["text"]))
    unique_a = (a & gold) - b
    unique_b = (b & gold) - a
    merge = bool(unique_a and unique_b)
    if merge:
        decision = "merge"
    else:
        score_a = score(pair["parent_a"]["text"], clean_title(row.get("target", "")))["rouge_l"]
        score_b = score(pair["parent_b"]["text"], clean_title(row.get("target", "")))["rouge_l"]
        decision = "keep_a" if score_a >= score_b else "keep_b"
    annotation = {
        "decision": decision,
        "parent_a_contribution": "heuristic contribution from parent A",
        "parent_b_contribution": "heuristic contribution from parent B",
        "merge_action": "Combine distinct useful content from both parents." if merge else "",
        "heuristic_only": True,
    }
    return apply_pair_gate(row, pair, annotation, {"minimum_parent_gold_rouge_l": 0.0})


def annotate_pairs(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    teacher_mode: str,
    checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    maximum_history = int(config.get("sft_data", {}).get("maximum_history_records", 8))
    settings = config.get("crossover_sft", {})
    selected: list[dict[str, Any]] = []
    for row in rows:
        pair = select_target_blind_pair(row, settings)
        item = dict(row)
        item["selected_pair"] = pair
        selected.append(item)
    if teacher_mode == "none":
        for item in selected:
            item["pair_annotation"] = {
                "usable": item["selected_pair"] is not None,
                "decision": "merge" if item["selected_pair"] else "reject",
                "accepted": item["selected_pair"] is not None,
                "training_accepted": item["selected_pair"] is not None,
                "gate_consistent": True,
                "gate_version": "teacher_skipped",
                "teacher_skipped": True,
            }
        return selected
    if teacher_mode == "heuristic":
        for item in selected:
            pair = item["selected_pair"]
            item["pair_annotation"] = (
                heuristic_pair_annotation(item, pair)
                if pair
                else {
                    "usable": False,
                    "decision": "reject",
                    "accepted": False,
                    "training_accepted": False,
                    "gate_consistent": True,
                    "gate_version": "heuristic_no_pair",
                }
            )
        return selected
    if teacher_mode != "api":
        raise ValueError(f"未知 teacher_mode={teacher_mode}")

    client = teacher_client(config)
    retries = int(config.get("crossover_sft", {}).get("schema_retries", 2))
    cached_rows = read_jsonl(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else []
    cached = {str(item.get("id", "")): item for item in cached_rows}
    for index, item in enumerate(selected):
        previous = cached.get(str(item.get("id", "")))
        if not previous or not item.get("selected_pair"):
            continue
        old_pair = previous.get("selected_pair") or {}
        new_pair = item["selected_pair"]
        old_ids = (
            str((old_pair.get("parent_a") or {}).get("candidate_id", "")),
            str((old_pair.get("parent_b") or {}).get("candidate_id", "")),
        )
        new_ids = (
            str(new_pair["parent_a"].get("candidate_id", "")),
            str(new_pair["parent_b"].get("candidate_id", "")),
        )
        previous_annotation = previous.get("pair_annotation") or {}
        if (
            old_ids == new_ids
            and previous_annotation
            and previous_annotation.get("gate_version") == "strict_teacher_program_gate_v2"
            and not previous_annotation.get("heuristic_only")
            and not previous_annotation.get("teacher_skipped")
        ):
            item["pair_annotation"] = previous_annotation
            if previous.get("teacher_raw_response"):
                item["teacher_raw_response"] = previous["teacher_raw_response"]
    jobs = [
        (index, item)
        for index, item in enumerate(selected)
        if item["selected_pair"] and not item.get("pair_annotation")
    ]
    results: dict[int, dict[str, Any]] = {}

    def worker(job: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        index, item = job
        prompt = teacher_pair_prompt(item, item["selected_pair"], maximum_history)
        last_error: Exception | None = None
        last_annotation: dict[str, Any] | None = None
        for attempt in range(retries + 1):
            task = f"crossover_pair_gate_v2_{item.get('id')}_attempt_{attempt}"
            try:
                payload, raw = client.json(task, prompt, {"sample_id": str(item.get("id"))})
                annotation = apply_pair_gate(
                    item,
                    item["selected_pair"],
                    normalize_pair_annotation(payload),
                    settings,
                )
                last_annotation = annotation
                if not annotation["gate_consistent"]:
                    raise ValueError(
                        "Teacher gate inconsistent: " + ",".join(annotation["gate_errors"])
                    )
                return {
                    "index": index,
                    "annotation": annotation,
                    "teacher_raw_response": raw,
                }
            except Exception as error:
                last_error = error
                client.invalidate(task, prompt)
        rejected = dict(last_annotation or {})
        rejected.update({
            "usable": False,
            "decision": "reject",
            "accepted": False,
            "training_accepted": False,
            "gate_consistent": False,
            "gate_version": "strict_teacher_program_gate_v2",
            "error": str(last_error),
        })
        return {
            "index": index,
            "annotation": rejected,
        }

    def on_result(job: tuple[int, dict[str, Any]], result: dict[str, Any], completed: int) -> None:
        results[result["index"]] = result
        selected[result["index"]]["pair_annotation"] = result["annotation"]
        if "teacher_raw_response" in result:
            selected[result["index"]]["teacher_raw_response"] = result[
                "teacher_raw_response"
            ]
        checkpoint_every = int(config.get("crossover_sft", {}).get("checkpoint_every", 20))
        if checkpoint_path and checkpoint_every > 0 and completed % checkpoint_every == 0:
            write_jsonl(
                checkpoint_path,
                [item for item in selected if item.get("pair_annotation")],
            )
        print(
            f"crossover teacher {completed}/{len(jobs)} sample={job[1].get('id')} "
            f"decision={result['annotation']['decision']}",
            flush=True,
        )

    try:
        run_bounded(
            jobs,
            worker,
            on_result,
            max_workers=max(1, int(config.get("crossover_sft", {}).get("concurrency", 2))),
            thread_name_prefix="crossover-pair",
        )
    except BoundedJobError as error:
        raise RuntimeError(str(error)) from error.error
    for index, item in enumerate(selected):
        if item["selected_pair"] is None:
            item["pair_annotation"] = {
                "usable": False,
                "decision": "reject",
                "accepted": False,
                "training_accepted": False,
                "gate_consistent": True,
                "gate_version": "strict_teacher_program_gate_v2",
                "error": "no_distinct_pair",
            }
        elif not item.get("pair_annotation"):
            result = results[index]
            item["pair_annotation"] = result["annotation"]
            if "teacher_raw_response" in result:
                item["teacher_raw_response"] = result["teacher_raw_response"]
    if checkpoint_path:
        write_jsonl(checkpoint_path, selected)
    return selected


def crossover_prompt(
    row: dict[str, Any], parent_a: dict[str, Any], parent_b: dict[str, Any], maximum_history: int
) -> str:
    return "[CROSSOVER_TITLE]\n" + build_editor_prompt(
        row,
        "crossover",
        parent_a,
        parent_b,
        maximum_history,
        supervision_mode="plain_output_only",
    )


def build_crossover_examples(
    pair_rows: list[dict[str, Any]], config: dict[str, Any], smoke: bool
) -> list[dict[str, Any]]:
    maximum_history = int(config.get("sft_data", {}).get("maximum_history_records", 8))
    fraction = float(config.get("sft_data", {}).get("validation_fraction", 0.05))
    examples: list[dict[str, Any]] = []
    for row in pair_rows:
        pair = row.get("selected_pair")
        annotation = row.get("pair_annotation", {})
        gold = clean_title(row.get("target", ""))
        if not pair or not annotation.get("training_accepted") or not gold:
            continue
        decision = str(annotation.get("decision", "reject"))
        if decision == "merge":
            supervised_output = gold
            output_source = "gold_merge"
        elif decision == "keep_a":
            supervised_output = clean_title(pair["parent_a"]["text"])
            output_source = "parent_a"
        elif decision == "keep_b":
            supervised_output = clean_title(pair["parent_b"]["text"])
            output_source = "parent_b"
        else:
            continue
        if not supervised_output:
            continue
        sample_id = str(row.get("id", ""))
        examples.append(
            {
                "example_id": f"{sample_id}:crossover:title",
                "sample_id": sample_id,
                "user_id": str(row.get("user_id", "")),
                "task": "crossover_title",
                "operation_type": "crossover",
                "parent_a_id": pair["parent_a"]["candidate_id"],
                "parent_b_id": pair["parent_b"]["candidate_id"],
                "parent_a": pair["parent_a"]["text"],
                "parent_b": pair["parent_b"]["text"],
                "teacher_pair_decision": decision,
                "output_source": output_source,
                "prompt": crossover_prompt(
                    row, pair["parent_a"], pair["parent_b"], maximum_history
                ),
                "target": gold,
                "output": supervised_output,
                "trace_text": "",
                "output_text": supervised_output,
                "sample_weight": 1.0,
                "split": deterministic_split(sample_id, fraction),
                "student_prompt_sees_gold": False,
                "teacher_sees_gold_for_pair_gate": not annotation.get("teacher_skipped", False),
            }
        )
    if smoke and examples and not any(x["split"] == "validation" for x in examples):
        examples[-1]["split"] = "validation"
    return examples


def write_crossover_dataset(
    output_dir: Path,
    pool_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    *,
    pool_source: str,
    teacher_mode: str,
    smoke: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "01_parent_pool.jsonl", pool_rows)
    write_jsonl(output_dir / "02_crossover_pairs.jsonl", pair_rows)
    write_jsonl(output_dir / "all_sft.jsonl", examples)
    train = [x for x in examples if x["split"] == "train"]
    validation = [x for x in examples if x["split"] == "validation"]
    write_jsonl(output_dir / "train_sft.jsonl", train)
    write_jsonl(output_dir / "validation_sft.jsonl", validation)
    if not train or not validation:
        raise ValueError(
            f"Crossover SFT train/validation 不能为空：train={len(train)} validation={len(validation)}"
        )
    decisions = {
        name: sum(x.get("pair_annotation", {}).get("decision") == name for x in pair_rows)
        for name in ("merge", "keep_a", "keep_b", "reject")
    }
    accepted = decisions["merge"]
    trainable = sum(
        bool(x.get("pair_annotation", {}).get("training_accepted")) for x in pair_rows
    )
    inconsistent = sum(
        not bool(x.get("pair_annotation", {}).get("gate_consistent", True)) for x in pair_rows
    )
    report = {
        "protocol": "crossover_only_plain_title_best_of_n_strict_gate_sft_v3",
        "parent_selection_protocol": "best_of_n_quality_complement_v3",
        "source_queries": len(pool_rows),
        "queries_with_two_parents": sum(len(x.get("parent_pool", [])) >= 2 for x in pool_rows),
        "accepted_crossover_pairs": accepted,
        "pair_acceptance_rate": accepted / max(len(pool_rows), 1),
        "trainable_pair_decisions": trainable,
        "trainable_pair_rate": trainable / max(len(pool_rows), 1),
        "teacher_decisions": decisions,
        "inconsistent_teacher_responses_after_retries": inconsistent,
        "examples": len(examples),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "pool_source": pool_source,
        "teacher_mode": teacher_mode,
        "formal_result_eligible": (
            pool_source == "base_model" and teacher_mode == "api" and not smoke
        ),
        "student_prompt_sees_gold": False,
        "single_task_plain_title": True,
        "merge_output_is_exact_gold": True,
        "keep_output_copies_selected_parent": True,
        "rationale_supervision": False,
    }
    write_json(output_dir / "manifest.json", report)
    print(f"Crossover-only SFT dataset -> {output_dir}; report={report}", flush=True)
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
    # 正式 Crossover 实验必须从 Base 初始化，不能继承 Mutation Adapter。
    local["training"]["initial_adapter_path"] = ""
    trainer = _load_module("06_train_editor_lora.py", "crossover_editor_trainer")
    return trainer.train(local, max_steps_override=max_steps)


class AdapterGenerator:
    def __init__(self, config: dict[str, Any], adapter: Path):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = resolve_path(config["model"]["path"])
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(adapter, use_fast=True, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
        )
        self.model = PeftModel.from_pretrained(base, adapter).to(self.device).eval()
        self.max_length = int(config.get("training", {}).get("max_length", 2048))
        self.max_new_tokens = int(config.get("evaluation", {}).get("max_new_tokens", 64))
        if self.max_new_tokens >= self.max_length:
            raise ValueError("生成 token 数必须小于模型上下文上限")
        # max_length 表示输入和新生成文本共同使用的上下文窗口。
        self.prompt_max_length = self.max_length - self.max_new_tokens

    def generate(self, prompts: list[str], batch_size: int) -> list[str]:
        values: list[str] = []
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
                {"input_ids": batch_ids},
                return_tensors="pt",
                padding=True,
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
                raw = self.tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True)
                values.append(clean_title(raw))
        return values

    def close(self) -> None:
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def evaluate_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [score(str(row.get("prediction", "")), str(row.get("target", ""))) for row in rows]
    bleu = corpus_bleu(
        [str(row.get("prediction", "")) for row in rows],
        [str(row.get("target", "")) for row in rows],
    )
    users = {str(row.get("user_id", "")) for row in rows if str(row.get("user_id", ""))}
    return {
        "protocol": "crossover_single_output",
        "users": len(users),
        "queries": len(rows),
        "valid_predictions": sum(bool(str(row.get("prediction", "")).strip()) for row in rows),
        "rouge_1": statistics.mean(x["rouge_1"] for x in scores),
        "rouge_l": statistics.mean(x["rouge_l"] for x in scores),
        "sacrebleu": float(bleu["score"]),
    }


def evaluate_crossover(
    config: dict[str, Any], pool_rows: list[dict[str, Any]], adapter: Path, output_dir: Path
) -> dict[str, Any]:
    maximum_history = int(config.get("sft_data", {}).get("maximum_history_records", 8))
    settings = config.get("crossover_sft", {})
    selected = [(row, select_target_blind_pair(row, settings)) for row in pool_rows]
    usable = [(row, pair) for row, pair in selected if pair is not None]
    prompts = [
        crossover_prompt(row, pair["parent_a"], pair["parent_b"], maximum_history)
        for row, pair in usable
    ]
    generator = AdapterGenerator(config, adapter)
    try:
        predictions = generator.generate(
            prompts, int(config.get("evaluation", {}).get("prediction_batch_size", 1))
        )
    finally:
        generator.close()
    generated_by_id = {
        str(source.get("id", "")): (pair, prediction)
        for (source, pair), prediction in zip(usable, predictions)
    }
    rows: list[dict[str, Any]] = []
    for source, selected_pair in selected:
        sample_id = str(source.get("id", ""))
        generated_value = generated_by_id.get(sample_id)
        if generated_value is not None:
            pair, prediction = generated_value
            error = None if prediction else "empty_prediction"
        else:
            pool = source.get("parent_pool", [])
            fallback = pool[0] if pool else {"text": ""}
            pair = {
                "parent_a": fallback,
                "parent_b": {"text": ""},
            }
            prediction = clean_title(fallback.get("text", ""))
            error = "no_pair_fallback_parent"
        rows.append(
            {
                "id": sample_id,
                "user_id": str(source.get("user_id", "")),
                "source_text": str(source.get("source_text", "")),
                "target": clean_title(source.get("target", "")),
                "parent_a": pair["parent_a"]["text"],
                "parent_b": pair["parent_b"]["text"],
                "prediction": prediction,
                "error": error,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "test_predictions.jsonl", rows)
    report = evaluate_prediction_rows(rows)
    report["crossover_pair_queries"] = len(usable)
    report["crossover_pair_coverage"] = len(usable) / max(len(pool_rows), 1)
    report["no_pair_fallback"] = len(pool_rows) - len(usable)
    write_json(output_dir / "global_test_report.json", report)
    print(f"Crossover evaluation -> {output_dir}; report={report}", flush=True)
    return report


def load_source_rows(config: dict[str, Any], source: str, split: str) -> list[dict[str, Any]]:
    if source:
        path = resolve_path(source)
        if not path.exists() and path.name == "01_base_parent_records.jsonl":
            fallback = path.with_name("all_sft.jsonl")
            if fallback.exists():
                print(f"Parent records fallback: {path} -> {fallback}", flush=True)
                path = fallback
        rows = read_jsonl(path)
        # 清理后保留的 Direct SFT all_sft.jsonl 使用 sample_id，且没有完整
        # Query/History/candidates。按同一 split 的 seeds 补回原始字段，同时保留
        # 已生成的真实 Base Parent，避免重新做 greedy 生成。
        if rows and (not rows[0].get("source_text") or not rows[0].get("candidates")):
            source_rows = read_jsonl(stage_path(config, split, "seeds"))
            by_id = {str(item.get("id", "")): item for item in source_rows}
            merged: list[dict[str, Any]] = []
            for item in rows:
                sample_id = str(item.get("id", item.get("sample_id", "")))
                original = by_id.get(sample_id)
                if original is None:
                    raise ValueError(f"无法从 {split} seeds 补回 sample={sample_id}")
                value = dict(original)
                value["parent"] = item.get("parent", item.get("prediction", ""))
                value["parent_source"] = item.get("parent_source", "cached_base_model")
                merged.append(value)
            rows = merged
    else:
        rows = read_jsonl(stage_path(config, split, "seeds"))
    if not rows:
        raise ValueError(f"没有输入行：source={source!r} split={split}")
    return rows


def build_stage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    data_dir = resolve_path(args.data_dir)
    rows = load_source_rows(config, args.parent_records, "train")
    pool_rows = build_parent_pools(
        rows,
        config,
        data_dir / "01_parent_pool.jsonl",
        pool_source=args.pool_source,
        extra_parents=args.extra_parents,
        limit=args.limit,
    )
    pair_rows = annotate_pairs(
        pool_rows,
        config,
        args.teacher_mode,
        checkpoint_path=data_dir / "02_crossover_pairs.jsonl",
    )
    examples = build_crossover_examples(pair_rows, config, args.smoke)
    return write_crossover_dataset(
        data_dir,
        pool_rows,
        pair_rows,
        examples,
        pool_source=args.pool_source,
        teacher_mode=args.teacher_mode,
        smoke=args.smoke,
    )


def pool_stage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """只生成 Parent Pool，便于使用含 torch 的 hydra 环境独立运行。"""

    data_dir = resolve_path(args.data_dir)
    rows = load_source_rows(config, args.parent_records, "train")
    pool_rows = build_parent_pools(
        rows,
        config,
        data_dir / "01_parent_pool.jsonl",
        pool_source=args.pool_source,
        extra_parents=args.extra_parents,
        limit=args.limit,
    )
    report = {
        "protocol": "crossover_parent_pool_v1",
        "source_queries": len(pool_rows),
        "queries_with_two_parents": sum(
            len(item.get("parent_pool", [])) >= 2 for item in pool_rows
        ),
        "pool_source": args.pool_source,
        "output": str(data_dir / "01_parent_pool.jsonl"),
    }
    write_json(data_dir / "parent_pool_report.json", report)
    print(f"Crossover Parent Pool -> {data_dir}; report={report}", flush=True)
    return report


def pairs_stage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """只执行 Pair 选择、Teacher 门控和 SFT JSONL 编译。"""

    data_dir = resolve_path(args.data_dir)
    pool_path = data_dir / "01_parent_pool.jsonl"
    if not pool_path.exists():
        raise FileNotFoundError(f"缺少 Parent Pool；先运行 --stage pool: {pool_path}")
    pool_rows = read_jsonl(pool_path)
    if args.limit > 0:
        pool_rows = pool_rows[: args.limit]
    pair_rows = annotate_pairs(
        pool_rows,
        config,
        args.teacher_mode,
        checkpoint_path=data_dir / "02_crossover_pairs.jsonl",
    )
    examples = build_crossover_examples(pair_rows, config, args.smoke)
    return write_crossover_dataset(
        data_dir,
        pool_rows,
        pair_rows,
        examples,
        pool_source=args.pool_source,
        teacher_mode=args.teacher_mode,
        smoke=args.smoke,
    )


def train_stage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    max_steps = args.max_steps if args.max_steps > 0 else None
    return train_adapter(
        config, resolve_path(args.data_dir), resolve_path(args.run_dir) / "editor", max_steps
    )


def eval_stage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    data_dir = resolve_path(args.data_dir)
    test_pool_path = data_dir / "test_parent_pool.jsonl"
    test_rows = load_source_rows(config, args.test_parent_records, "test")
    test_pool = build_parent_pools(
        test_rows,
        config,
        test_pool_path,
        pool_source=args.pool_source,
        extra_parents=args.extra_parents,
        limit=args.test_limit,
    )
    adapter = resolve_path(args.run_dir) / "editor" / "final_adapter"
    if not adapter.exists():
        raise FileNotFoundError(f"缺少 Crossover Adapter: {adapter}")
    return evaluate_crossover(config, test_pool, adapter, resolve_path(args.run_dir) / "crossover")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crossover-only SFT 数据、训练与评估")
    parser.add_argument(
        "--config", default=str(HERE / "config_crossover_sft.yaml")
    )
    parser.add_argument(
        "--stage",
        choices=("pool", "pairs", "build", "train", "eval", "all"),
        default="all",
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--parent-records", default=DEFAULT_PARENT_RECORDS)
    parser.add_argument("--test-parent-records", default="")
    parser.add_argument(
        "--pool-source", choices=("base_model", "existing_candidates"), default="base_model"
    )
    parser.add_argument("--teacher-mode", choices=("api", "heuristic", "none"), default="api")
    parser.add_argument(
        "--extra-parents",
        type=int,
        default=0,
        help="采样 Parent 数；0 表示读取 crossover_sft.best_of_n_samples",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.setdefault("crossover_sft", {})
    if args.extra_parents <= 0:
        args.extra_parents = int(config["crossover_sft"].get("best_of_n_samples", 12))
    if args.smoke:
        print(
            "SMOKE MODE: only validates code/data/model contracts; do not report its metrics "
            "as formal results.",
            flush=True,
        )
    result: dict[str, Any] = {}
    if args.stage == "pool":
        result["pool"] = pool_stage(args, config)
    elif args.stage == "pairs":
        result["pairs"] = pairs_stage(args, config)
    elif args.stage in {"build", "all"}:
        result["build"] = build_stage(args, config)
    if args.stage in {"train", "all"}:
        result["train"] = train_stage(args, config)
    if args.stage in {"eval", "all"}:
        result["eval"] = eval_stage(args, config)
    print(f"CROSSOVER_SFT_DONE={result}", flush=True)


if __name__ == "__main__":
    main()
