"""2026-07-31 两阶段个性化候选优化的公共契约。

这条实验路线只允许两种个性化通道：

1. Editor 直接读取当前 Query 检索到的历史输入/输出样例；
2. Scorer 使用由用户历史 Leave-One-Out 数据拟合的独立 Head/Adapter。

阶段一支持两种可比较监督：S0 只把 Gold 作为 Editor 输出标签，Gold-aware
Trace 则允许 Teacher 解释 Parent 到 Gold。Seed、本地 Editor、Scorer 及所有
validation/test 推理始终严格 target-blind。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import re
import string
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

from common.metrics import score  # noqa: E402
from common.teacher import TeacherClient  # noqa: E402


NORMALIZE = re.compile(r"[^a-z0-9]+")
FORBIDDEN_TEACHER_TERMS = re.compile(
    r"\b(?:gold|ground[ -]?truth|rouge|reference answer|target title)\b",
    re.IGNORECASE,
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """加载本实验自己的 YAML，避免触发主工程 experiment.auto_paths。"""

    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("配置根节点必须是 mapping")
    base = config.pop("base", None)
    if base:
        base_path = Path(base)
        if not base_path.is_absolute():
            local = config_path.parent / base_path
            project = PROJECT_ROOT / base_path
            base_path = local if local.exists() else project
        config = _deep_merge(load_config(base_path), config)
    config["_config_path"] = str(config_path)
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"JSONL 格式错误 {path}:{line_number}: {error}") from error
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(destination)
    except OSError:
        # Keep failures diagnosable while avoiding a multi-gigabyte orphan when
        # a filesystem fills during atomic JSONL writes.
        temporary.unlink(missing_ok=True)
        raise


def truncate_prompt_ids(input_ids: list[int], maximum: int) -> list[int]:
    """统一截断过长 Prompt，同时保留任务开头和末尾控制信息。

    Editor Prompt 的 Query 位于前部，而 Parent、输出格式和 ``OUTPUT:`` 位于
    末尾。默认右截断会删除编辑任务末尾，使模型退化成 History 续写。训练、
    Parent 生成和 Adapter 推理共同调用本函数，避免输入处理口径不一致。
    """

    if maximum <= 1:
        raise ValueError("Prompt token 上限必须大于 1")
    if len(input_ids) <= maximum:
        return input_ids
    head = max(1, maximum // 2)
    return input_ids[:head] + input_ids[-(maximum - head) :]


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(destination)


def load_project_stage(relative_path: str, module_name: str):
    """复用主工程稳定的数据读取/Ranker 实现，不复制大段旧代码。"""

    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载阶段文件: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_dir(config: dict[str, Any], split: str) -> Path:
    spec = config["splits"].get(split)
    if not isinstance(spec, dict):
        raise ValueError(f"配置中不存在 split={split}")
    return resolve_path(config["paths"]["candidate_root"]) / str(
        spec["processed_split"]
    )


STAGE_FILES = {
    "prepare": "01_prepared.jsonl",
    "retrieve": "02_retrieved.jsonl",
    "seeds": "03_seeds.jsonl",
    "evolved": "04_evolved_scored.jsonl",
    "gold_traces": "04_gold_aware_traces.jsonl",
    "atomic_traces": "04_atomic_traces.jsonl",
    "editor": "07_editor_scored.jsonl",
}


def stage_path(config: dict[str, Any], split: str, stage: str) -> Path:
    return split_dir(config, split) / STAGE_FILES[stage]


def normalized_text(text: str) -> str:
    return NORMALIZE.sub("", str(text).casefold())


def visible_history(row: dict[str, Any], maximum: int) -> list[dict[str, str]]:
    """只返回目标隔离后的检索历史；ID 用于校验 Trace 证据。"""

    values = []
    # Gold 仅作为 deny-list，防止数据源中的异常重复历史把当前答案泄漏给 Teacher。
    target_key = normalized_text(row.get("target", ""))
    for item in row.get("retrieved_profile", [])[:maximum]:
        abstract = str(item.get("abstract", "")).strip()
        title = str(item.get("title", "")).strip()
        if target_key and normalized_text(title) == target_key:
            continue
        if title:
            values.append(
                {
                    "id": str(item.get("id", "")),
                    "input": abstract,
                    "output": title,
                }
            )
    return values


def render_local_prompt(name: str, **values: Any) -> str:
    template = (HERE / "prompts" / name).read_text(encoding="utf-8")
    return string.Template(template).substitute(**values)


def teacher_client(config: dict[str, Any]) -> TeacherClient:
    settings = dict(config["teacher"])
    cache_dir = resolve_path(settings["cache_dir"])
    settings["cache_dir"] = str(cache_dir)
    return TeacherClient(settings, cache_dir)


def deterministic_split(sample_id: str, fraction: float) -> str:
    bucket = int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16)
    ratio = bucket / 0xFFFFFFFF
    return "validation" if ratio < fraction else "train"


def candidate(candidate_id: str, kind: str, text: str, **extra: Any) -> dict[str, Any]:
    title = str(text).strip().strip('"').strip()
    if not title or "\n" in title or len(title) > 300:
        raise ValueError(f"无效候选 candidate_id={candidate_id}")
    return {"candidate_id": candidate_id, "type": kind, "text": title, **extra}


def compact_signal(value: Any, valid_evidence: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("profile_signal 必须是 object")
    evidence_ids = list(
        dict.fromkeys(str(item) for item in value.get("evidence_ids", []))
    )
    unknown = set(evidence_ids) - valid_evidence
    if unknown:
        raise ValueError(f"profile_signal 引用了不可见历史: {sorted(unknown)}")
    observation = str(value.get("observation", "")).strip()
    if not observation or len(observation) > 240:
        raise ValueError("profile_signal.observation 必须是简短非空文本")
    if FORBIDDEN_TEACHER_TERMS.search(observation):
        raise ValueError("profile_signal 包含禁止的 Gold/指标术语")
    return {"evidence_ids": evidence_ids, "observation": observation}


def validate_mutation(
    value: Any,
    parent: dict[str, Any],
    valid_evidence: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Mutation 必须是 object")
    if str(value.get("parent_id", "")) != str(parent["candidate_id"]):
        raise ValueError("Mutation parent_id 不匹配")
    decision = str(value.get("decision", "")).lower()
    if decision not in {"revise", "keep"}:
        raise ValueError("Mutation decision 必须是 revise 或 keep")
    signal = compact_signal(value.get("profile_signal"), valid_evidence)
    action = str(value.get("edit_action", "")).strip()
    if not action or len(action) > 240 or FORBIDDEN_TEACHER_TERMS.search(action):
        raise ValueError("edit_action 无效或包含 Gold/指标术语")
    output = str(value.get("output", "")).strip().strip('"').strip()
    if not output or "\n" in output or len(output) > 300:
        raise ValueError("Mutation output 无效")
    if decision == "keep" and normalized_text(output) != normalized_text(parent["text"]):
        raise ValueError("decision=keep 时 output 必须等于 Parent")
    return {
        "decision": decision,
        "profile_signal": signal,
        "edit_action": action,
        "output": output,
    }


def validate_crossover(
    value: Any,
    parent_a: dict[str, Any],
    parent_b: dict[str, Any],
    valid_evidence: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Crossover 必须是 object")
    if str(value.get("parent_a_id", "")) != str(parent_a["candidate_id"]):
        raise ValueError("Crossover parent_a_id 不匹配")
    if str(value.get("parent_b_id", "")) != str(parent_b["candidate_id"]):
        raise ValueError("Crossover parent_b_id 不匹配")
    decision = str(value.get("decision", "")).lower()
    if decision not in {"merge", "keep_a", "keep_b"}:
        raise ValueError("Crossover decision 必须是 merge/keep_a/keep_b")
    signal = compact_signal(value.get("profile_signal"), valid_evidence)
    action = str(value.get("merge_action", "")).strip()
    if not action or len(action) > 240 or FORBIDDEN_TEACHER_TERMS.search(action):
        raise ValueError("merge_action 无效或包含 Gold/指标术语")
    output = str(value.get("output", "")).strip().strip('"').strip()
    if not output or "\n" in output or len(output) > 300:
        raise ValueError("Crossover output 无效")
    expected = None
    if decision == "keep_a":
        expected = parent_a["text"]
    elif decision == "keep_b":
        expected = parent_b["text"]
    if expected and normalized_text(output) != normalized_text(expected):
        raise ValueError(f"decision={decision} 时 output 必须等于对应 Parent")
    return {
        "decision": decision,
        "profile_signal": signal,
        "merge_action": action,
        "output": output,
    }


def lexical_jaccard(left: str, right: str) -> float:
    a = set(re.findall(r"[a-z0-9]+", left.casefold()))
    b = set(re.findall(r"[a-z0-9]+", right.casefold()))
    return len(a & b) / len(a | b) if a or b else 1.0


def choose_crossover_pairs(
    candidates: list[dict[str, Any]], count: int
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """目标隔离地选择互补父候选：优先来源不同且词汇差异较大的 pair。"""

    options = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            source_penalty = 0 if left.get("type") != right.get("type") else 1
            options.append(
                (
                    source_penalty,
                    lexical_jaccard(left["text"], right["text"]),
                    str(left["candidate_id"]),
                    str(right["candidate_id"]),
                    left,
                    right,
                )
            )
    options.sort(key=lambda item: item[:4])
    selected = []
    used_pairs = set()
    for _, _, _, _, left, right in options:
        key = tuple(sorted((str(left["candidate_id"]), str(right["candidate_id"]))))
        if key in used_pairs:
            continue
        selected.append((left, right))
        used_pairs.add(key)
        if len(selected) >= count:
            break
    return selected


def score_candidate_pool(row: dict[str, Any], metric: str, margin: float) -> None:
    """Gold 首次在这里出现；它只写入标签字段，不进入任何模型 Prompt。"""

    all_candidates = row["candidates"] + row.get("mutations", [])
    by_id = {str(item["candidate_id"]): item for item in all_candidates}
    for item in all_candidates:
        item["scores"] = score(str(item["text"]), str(row["target"]))
    preferences = []
    for child in row.get("mutations", []):
        if child["type"] == "mutation":
            parents = [by_id[str(child["parent_id"])]]
        else:
            parents = [
                by_id[str(child["parent_a_id"])],
                by_id[str(child["parent_b_id"])],
            ]
        parent = max(parents, key=lambda item: float(item["scores"][metric]))
        delta = float(child["scores"][metric]) - float(parent["scores"][metric])
        child["delta"] = round(delta, 8)
        if abs(delta) < margin:
            continue
        chosen, rejected = (child, parent) if delta > 0 else (parent, child)
        preferences.append(
            {
                "id": f"{row['id']}:{child['candidate_id']}",
                "sample_id": str(row["id"]),
                "chosen_id": str(chosen["candidate_id"]),
                "chosen": str(chosen["text"]),
                "rejected_id": str(rejected["candidate_id"]),
                "rejected": str(rejected["text"]),
                "metric": metric,
                "margin": round(abs(delta), 8),
            }
        )
    row["preferences"] = preferences
    row["metric_metadata"] = {
        "primary": metric,
        "preference_margin": margin,
        "gold_visible_during_generation": False,
    }


def response_parts(example: dict[str, Any]) -> tuple[str, str]:
    """把紧凑 JSON 分成 Trace span 和最终 Output span，供加权 SFT。"""

    signal = example["profile_signal"]
    key = "edit_action" if example["operation_type"] == "mutation" else "merge_action"
    prefix = {
        "decision": example["decision"],
        **(
            {"task_correction": example["task_correction"]}
            if example.get("task_correction")
            else {}
        ),
        "profile_signal": {
            "evidence_ids": signal["evidence_ids"],
            "observation": signal["observation"],
        },
        key: example[key],
    }
    encoded = json.dumps(prefix, ensure_ascii=False, separators=(",", ":"))
    trace_text = encoded[:-1] + ',"output":'
    output_text = json.dumps(example["output"], ensure_ascii=False) + "}"
    return trace_text, output_text


def build_editor_prompt(
    row: dict[str, Any],
    operation_type: str,
    parent_a: dict[str, Any],
    parent_b: dict[str, Any] | None,
    maximum_history: int,
    supervision_mode: str = "gold_aware_trace",
    history_input_max_chars: int = 0,
    history_output_max_chars: int = 0,
) -> str:
    history = visible_history(row, maximum_history)
    # Top-8 完整 Abstract 很容易挤占当前 Query、Parent 和输出标签的上下文。
    # 新的简化 Trace 路线只压缩送给模型的历史副本，不修改磁盘上的检索结果。
    # 旧实验未传这两个参数，因而保持完全相同的输入。
    if history_input_max_chars > 0 or history_output_max_chars > 0:
        compact = []
        for item in history:
            value = dict(item)
            if history_input_max_chars > 0:
                value["input"] = value["input"][:history_input_max_chars].rstrip()
            if history_output_max_chars > 0:
                value["output"] = value["output"][:history_output_max_chars].rstrip()
            compact.append(value)
        history = compact
    payload: dict[str, Any] = {
        "operation_type": operation_type,
        "current_input": str(row["source_text"]),
        "retrieved_history": history,
        "parent_a": {
            "candidate_id": str(parent_a["candidate_id"]),
            "text": str(parent_a["text"]),
        },
    }
    if parent_b is not None:
        payload["parent_b"] = {
            "candidate_id": str(parent_b["candidate_id"]),
            "text": str(parent_b["text"]),
        }
    if supervision_mode == "simple_conditional_trace_idpo":
        # Top-8 只作为证据候选池。响应最多引用两条历史，把“偏好”和
        # “当前适用性”压缩为一句 edit_reason，避免1.5B模型学习多层 JSON。
        schema = (
            '{"evidence_ids":["..."],"edit_reason":"...",'
            '"edit_action":"...","output":"..."}'
        )
        instruction = (
            "You are a target-blind personalized output editor. Use only CURRENT_INPUT, "
            "PARENT candidate(s), and RETRIEVED_HISTORY. Select one or two visible history "
            "records only when they support a useful editing preference that applies to the "
            "current input. In one short edit_reason, state what the selected history suggests "
            "and why it applies now. In one short edit_action, state the concrete revision. "
            "Then produce the final output. Cite only visible history IDs. Do not require two "
            "records when one is sufficient. Never mention a reference answer, metric, or gold "
            "output. Return JSON only."
        )
    elif supervision_mode == "conditional_preference_trace_idpo":
        # 阶段二 Trace-aware IDPO 使用与阶段一有效个性化样本相同的
        # ``历史分析 -> 条件偏好 -> 当前适用性 -> 编辑计划 -> 输出`` 结构。
        # Gold 不在 payload 中；模型必须只根据当前输入、Parent 和可见历史
        # 自己生成 target-blind Trace。
        schema = (
            '{"history_analysis":{"evidence_ids":["..."]},'
            '"preference":{"condition":"...","preferred_behavior":"..."},'
            '"applicability":{"decision":"apply","matched_input_span":"..."},'
            '"edit_plan":["..."],"output":"..."}'
        )
        instruction = (
            "You are a target-blind personalized output editor. Use only CURRENT_INPUT, "
            "PARENT candidate(s), and RETRIEVED_HISTORY. First infer one conditional user "
            "preference supported by at least two visible history records: an observable "
            "input condition followed by a recurring output behavior. Cite only visible "
            "history IDs. Then verify that the condition applies to CURRENT_INPUT by copying "
            "an exact matched input span, write one or two concrete edit steps, and produce "
            "the final output. Do not use topic recurrence or generic writing advice as a "
            "preference. Never infer or mention a reference answer, metric, or gold output. "
            "Return JSON only."
        )
    elif supervision_mode == "global_mutation_edit":
        schema = (
            '{"evidence_ids":["..."],"edit_reason":"...","edit_action":"...",'
            '"output":"..."}'
        )
        instruction = (
            "You are an on-policy mutation editor. Make exactly one localized edit to "
            "PARENT that improves the title for CURRENT_INPUT while preserving its factual "
            "contribution. Use RETRIEVED_HISTORY only as optional evidence of a recurring "
            "user-specific editing behavior. The edit_action must describe the single "
            "observable operation (insert, delete, replace, reorder, or compress) and "
            "the output must be the complete revised title. Do not write a long rationale, "
            "do not mention gold or metrics, and return JSON only."
        )
    elif supervision_mode == "plain_output_only":
        schema = "<one-line-final-title>"
        instruction = (
            "You are a target-blind personalized output editor. Use only CURRENT_INPUT, "
            "PARENT candidate(s), and RETRIEVED_HISTORY. Improve the parent into the best "
            "final title supported by the current input. Use history only when it provides "
            "applicable recurring output behavior. Return exactly one line containing only "
            "the final title. Do not output JSON, Markdown, explanation, or a prefix."
        )
    elif supervision_mode in {
        "output_only",
        "conditional_preference_trace",
        "simple_conditional_trace",
    }:
        # S0 不训练 Teacher 事后解释。模型只生成最终编辑结果，decision 与
        # provenance 等诊断字段由 Stage 07 根据输出和 Parent 确定。
        schema = '{"output":"..."}'
        instruction = (
            "You are a target-blind personalized output editor. Use only CURRENT_INPUT, "
            "PARENT candidate(s), and RETRIEVED_HISTORY. Improve the parent into the best "
            "final output supported by the current input. Use history only when it provides "
            "applicable recurring output behavior. Never infer or mention a reference answer, "
            "metric, or gold output. Return JSON only."
        )
    elif supervision_mode == "gold_aware_trace":
        schema = (
        '{"decision":"revise|keep","task_correction":"...","profile_signal":{"evidence_ids":[],"observation":"..."},'
        '"edit_action":"...","output":"..."}'
        if operation_type == "mutation"
        else '{"decision":"merge|keep_a|keep_b","profile_signal":{"evidence_ids":[],"observation":"..."},'
        '"merge_action":"...","output":"..."}'
        )
        instruction = (
            "You are a target-blind personalized output editor. Use only CURRENT_INPUT, "
            "PARENT candidate(s), and RETRIEVED_HISTORY. Never infer or mention a reference "
            "answer, metric, or gold output. task_correction must summarize the task-content "
            "correction. The profile signal must be local to this edit, supported by the cited "
            "visible history IDs, and must not be generic advice. If history does not support "
            "a useful change, keep the best parent. Return JSON only."
        )
    elif supervision_mode == "atomic_trace":
        schema = (
            '{"task_operations":[{"type":"add|remove|replace|reorder|compress|expand|format|preserve",'
            '"source_span":"...","target_span":"..."}],'
            '"personalized_operations":[{"type":"add|remove|replace|reorder|compress|expand|format|preserve",'
            '"source_span":"...","target_span":"...","evidence_ids":["..."],'
            '"history_pattern":"...","application":"..."}],"output":"..."}'
        )
        instruction = (
            "You are a target-blind personalized output editor. Use only CURRENT_INPUT, "
            "PARENT candidate(s), and RETRIEVED_HISTORY. Decompose the edit into concise "
            "local task_operations; add personalized_operations only when at least two visible "
            "history records support the same applicable pattern. Do not mention a reference "
            "answer, metric, or gold output. Return JSON only."
        )
    else:
        raise ValueError(f"未知 Editor supervision_mode={supervision_mode}")
    return (
        instruction
        + "\n\n"
        f"REQUIRED_SCHEMA:\n{schema}\n\nPAYLOAD:\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n\nOUTPUT:\n"
    )


def build_base_text_prompt(
    row: dict[str, Any], parent: dict[str, Any], maximum_history: int = 8
) -> str:
    """Base/Trace 标准评估使用的纯文本提示。

    这是与 ``30_generate_base_predictions.py`` 相同的协议：不要求 JSON，
    只要求模型生成一行标题。Output-only 因果消融复用该提示，避免把
    Prompt、上下文组织和解析规则混入监督方式差异。
    """

    history = visible_history(row, maximum_history)
    examples = "\n".join(
        f"Example input: {item['input'][:300]}\nExample title: {item['output'][:180]}"
        for item in history
    )
    return (
        "Write one concise academic paper title. Use the abstract and the user title "
        "examples as guidance. Preserve factual content. Output only the title on one line.\n\n"
        f"ABSTRACT:\n{row['source_text']}\n\n"
        f"USER TITLE EXAMPLES:\n{examples}\n\n"
        f"PARENT TITLE:\n{parent['text']}\n\nTITLE:\n"
    )


def build_configured_editor_prompt(
    config: dict[str, Any],
    row: dict[str, Any],
    operation_type: str,
    parent_a: dict[str, Any],
    parent_b: dict[str, Any] | None,
    maximum_history: int,
    **kwargs: Any,
) -> str:
    """按配置选择标准 Base-text 或全局结构化 Editor Prompt。"""

    protocol = str(config.get("sft_data", {}).get("prompt_protocol", "")).strip()
    if protocol == "base_text":
        if operation_type != "mutation" or parent_b is not None:
            raise ValueError("base_text prompt 只支持单 Parent mutation")
        return build_base_text_prompt(row, parent_a, maximum_history)
    return build_editor_prompt(
        row,
        operation_type,
        parent_a,
        parent_b,
        maximum_history,
        **kwargs,
    )


def mock_title(source_text: str, salt: str, words: int = 8) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", source_text)[:words]
    return " ".join(tokens).title() + f" {salt}"


def rng_for(seed: int, *values: Any) -> random.Random:
    return random.Random(":".join([str(seed), *(str(value) for value in values)]))
