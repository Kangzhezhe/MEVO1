"""Teacher 客户端：缓存、并发安全、兼容 JSON 提取和失败重试。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

try:
    from json_repair import repair_json as _repair_json
except ImportError:  # 允许合法 JSON 在尚未安装新依赖的旧环境中继续运行。
    _repair_json = None


def _extract_json(text: str, allow_repaired_scalar: bool = False) -> Any:
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    candidates = [text]
    start_candidates = [position for position in (text.find("{"), text.find("[")) if position >= 0]
    if start_candidates:
        start = min(start_candidates)
        end = max(text.rfind("}"), text.rfind("]"))
        if end >= start:
            candidates.append(text[start : end + 1])

    error = None
    for candidate in candidates:
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        repaired = re.sub(r"([\[,]\s*)\{\s*\{(?=\s*\")", r"\1{", repaired)
        # Some compatible endpoints emit LaTeX-like text such as ``\_`` or
        # ``\alpha`` directly inside JSON strings. JSON only permits a small
        # escape set, so preserve those backslashes by escaping them once.
        escape_repaired = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', repaired)
        for value in (candidate, repaired, escape_repaired):
            for strict in (True, False):
                try:
                    # strict=False is a fallback for compatible endpoints that
                    # place literal newlines/tabs inside a JSON string. It does
                    # not relax object structure or downstream schema checks.
                    return json.loads(value, strict=strict)
                except json.JSONDecodeError as exc:
                    error = exc

    # 正则只处理少数已知兼容问题；更复杂的缺引号、缺括号等语法错误交给
    # json-repair。修复结果必须仍是结构化 JSON，避免把任意说明文本当作成功。
    repair_error: Exception | None = None
    if _repair_json is not None:
        for candidate in candidates:
            try:
                parsed = _repair_json(candidate, return_objects=True)
                if isinstance(parsed, (dict, list)):
                    return parsed
                # 某些 OpenAI-compatible 端点在“只需一个候选”时会
                # 直接返回一行标题。只有显式开启的业务阶段可以接收
                # json-repair 修复后的 scalar；其他结构化任务仍保持严格。
                if allow_repaired_scalar and isinstance(parsed, str):
                    # json-repair 对某些无引号自然语言会返回空字符串；
                    # 此时保留原始单行值，由 Seed 层继续做长度、换行、
                    # 非空和去重校验。多行说明文不会在这里放行。
                    scalar = parsed.strip() or candidate.strip().strip('"').strip()
                    if scalar and "\n" not in scalar and "\r" not in scalar:
                        return scalar
                repair_error = ValueError(
                    f"json-repair returned {type(parsed).__name__}, expected object or array"
                )
            except Exception as exc:
                repair_error = exc

    detail = (
        f"json-repair failed: {repair_error}"
        if _repair_json is not None
        else "json-repair is not installed"
    )
    # TeacherClient._openai 捕获 JSONDecodeError 后会重新请求 Teacher；这里不
    # 返回猜测值，也不放宽下游 Mutation/Crossover/Seed 的 Schema 校验。
    raise json.JSONDecodeError(detail, text, 0) from (repair_error or error)


class TeacherClient:
    def __init__(self, config: dict[str, Any], cache_dir: Path):
        self.config = config
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_locks: dict[Path, threading.Lock] = {}
        self._cache_locks_guard = threading.Lock()
        self._endpoint_health_lock = threading.Lock()
        self._primary_circuit_open = False
        self._primary_disabled_until = 0.0
        self._primary_probe_in_flight = False

    def _cache_lock(self, path: Path) -> threading.Lock:
        with self._cache_locks_guard:
            return self._cache_locks.setdefault(path, threading.Lock())

    def _cache_path(self, task: str, prompt: str) -> Path:
        material = json.dumps(
            {
                "task": task,
                "prompt": prompt,
                "provider": self.config["provider"],
                "base_url": self.config.get("base_url"),
                "model": self.config["model"],
                "temperature": self.config["temperature"],
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return self.cache_dir / f"{hashlib.sha256(material.encode()).hexdigest()}.json"

    def invalidate(self, task: str, prompt: str) -> None:
        """删除已解析但未通过下游业务 Schema 的缓存响应。"""

        cache_path = self._cache_path(task, prompt)
        with self._cache_lock(cache_path):
            cache_path.unlink(missing_ok=True)

    def json(self, task: str, prompt: str, mock_context: dict[str, Any]) -> tuple[Any, str]:
        cache_path = self._cache_path(task, prompt)
        # A per-key lock prevents duplicate API calls and partial cache writes
        # when concurrent operations happen to produce the same prompt.
        with self._cache_lock(cache_path):
            if cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                return cached["parsed"], cached["raw_response"]
            if self.config["provider"] == "mock":
                parsed = self._mock(task, mock_context)
                raw = json.dumps(parsed, ensure_ascii=False)
            elif self.config["provider"] == "openai_compatible":
                parsed, raw = self._openai(prompt)
            else:
                raise ValueError(f"Unsupported teacher.provider={self.config['provider']}")
            temporary = cache_path.with_suffix(f"{cache_path.suffix}.{threading.get_ident()}.tmp")
            temporary.write_text(
                json.dumps({"task": task, "parsed": parsed, "raw_response": raw}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(cache_path)
            return parsed, raw

    def _fallback_settings(self) -> dict[str, Any] | None:
        fallback = self.config.get("fallback")
        if not isinstance(fallback, dict) or not bool(fallback.get("enabled", True)):
            return None
        settings = dict(self.config)
        settings.pop("fallback", None)
        settings.update(fallback)
        if settings.get("provider") != "openai_compatible":
            raise ValueError("teacher.fallback 仅支持 openai_compatible")
        return settings

    def _should_try_primary(self) -> bool:
        with self._endpoint_health_lock:
            if not self._primary_circuit_open:
                return True
            if time.monotonic() < self._primary_disabled_until:
                return False
            if self._primary_probe_in_flight:
                return False
            self._primary_probe_in_flight = True
            return True

    def _record_primary_success(self) -> None:
        with self._endpoint_health_lock:
            self._primary_circuit_open = False
            self._primary_disabled_until = 0.0
            self._primary_probe_in_flight = False

    def _record_primary_failure(self) -> None:
        cooldown = max(1.0, float(self.config.get("primary_cooldown_seconds", 300)))
        with self._endpoint_health_lock:
            self._primary_circuit_open = True
            self._primary_disabled_until = time.monotonic() + cooldown
            self._primary_probe_in_flight = False

    def _openai(self, prompt: str) -> tuple[Any, str]:
        fallback = self._fallback_settings()
        if fallback is None:
            return self._openai_endpoint(prompt, self.config)

        if self._should_try_primary():
            try:
                result = self._openai_endpoint(prompt, self.config)
                self._record_primary_success()
                return result
            except RuntimeError as primary_error:
                self._record_primary_failure()
                print(
                    "Teacher primary unavailable; switching to fallback "
                    f"model={fallback['model']}: {primary_error}",
                    flush=True,
                )
        return self._openai_endpoint(prompt, fallback)

    def _openai_endpoint(
        self, prompt: str, settings: dict[str, Any]
    ) -> tuple[Any, str]:
        import httpx
        from openai import OpenAI

        api_key = os.getenv(settings["api_key_env"])
        if not api_key:
            raise RuntimeError(f"Set {settings['api_key_env']} or use teacher.provider=mock")
        http_client = httpx.Client(
            verify=bool(settings.get("verify_ssl", True)),
            trust_env=bool(settings.get("trust_env", False)),
        )
        client = OpenAI(
            api_key=api_key,
            base_url=settings.get("base_url"),
            timeout=settings["timeout_seconds"],
            max_retries=0,
            http_client=http_client,
        )
        try:
            request_retries = max(1, int(settings.get("max_retries", 3)))
            json_parse_retries = max(0, int(settings.get("json_parse_retries", 5)))
            request_prompt = prompt
            parse_failures = 0

            while True:
                response = None
                request_error = None
                for attempt in range(request_retries):
                    try:
                        # Do not require response_format=json_object: VisGPT is
                        # only assumed to expose a chat-completions-compatible
                        # endpoint. JSON validity is handled explicitly below.
                        request = {
                            "model": settings["model"],
                            "messages": [{"role": "user", "content": request_prompt}],
                            "temperature": settings["temperature"],
                        }
                        # Local OpenAI-compatible servers often default to a
                        # very small generation budget. Structured factor and
                        # candidate responses need an explicit upper bound.
                        if settings.get("max_tokens") is not None:
                            request["max_tokens"] = int(settings["max_tokens"])
                        response = client.chat.completions.create(**request)
                        break
                    except Exception as exc:
                        request_error = exc
                        if attempt + 1 < request_retries:
                            time.sleep(min(2**attempt, 4))
                if response is None:
                    raise RuntimeError(
                        f"Teacher request failed after {request_retries} attempts: {request_error}"
                    ) from request_error

                raw = response.choices[0].message.content or ""
                try:
                    return _extract_json(
                        raw,
                        allow_repaired_scalar=bool(
                            settings.get("allow_repaired_json_scalar", False)
                        ),
                    ), raw
                except json.JSONDecodeError as error:
                    if parse_failures >= json_parse_retries:
                        raise RuntimeError(
                            "Teacher returned invalid JSON after "
                            f"{parse_failures + 1} responses: {error}"
                        ) from error
                    parse_failures += 1
                    # Retry the API call rather than trying to invent missing
                    # string content locally. A short corrective instruction
                    # makes a deterministic/low-temperature model less likely
                    # to reproduce the same truncated response.
                    request_prompt = (
                        prompt
                        + "\n\nIMPORTANT JSON RETRY: The previous response could not be parsed as "
                        "complete JSON. Return the complete JSON object again, close every string "
                        "and bracket, and output no text outside the JSON object."
                    )
                    time.sleep(min(2 ** (parse_failures - 1), 4))
        finally:
            # A fresh client is intentionally used per concurrent request. It
            # must be closed here or long Teacher runs accumulate CLOSE-WAIT
            # sockets and eventually stall despite available worker threads.
            client.close()

    @staticmethod
    def _mock(task: str, context: dict[str, Any]) -> Any:
        profiles = context.get("retrieved_profile", [])
        best = profiles[0] if profiles else {"id": "none", "title": "Concise Scientific Title"}
        if task.startswith("factor_validation_task_drafts"):
            return {
                "drafts": [
                    {
                        "id": str(record["id"]),
                        "draft": " ".join(re.findall(r"[A-Za-z0-9]+", str(record["abstract"]))[:6]).title(),
                    }
                    for record in context.get("records", [])
                ]
            }
        if task == "factor_validation_extract":
            record_ids = [str(record["id"]) for record in context.get("records", [])]
            return {
                "factors": [
                    {
                        "operation_type": "compression",
                        "direction": "Compress the title to its central contribution.",
                        "condition": "when the draft contains secondary implementation details",
                        "evidence_ids": record_ids,
                        "reason": "The references consistently remove secondary details.",
                    },
                    {
                        "operation_type": "structure",
                        "direction": "Use a concise scientific noun phrase.",
                        "condition": "when a noun phrase clearly expresses the contribution",
                        "evidence_ids": record_ids,
                        "reason": "The references consistently use noun-phrase titles.",
                    },
                ]
            }
        if task == "factor_validation_merge":
            candidates = context.get("factor_candidates", [])
            merged = []
            for operation_type in ("compression", "structure"):
                matching = [item for item in candidates if item.get("operation_type") == operation_type]
                if not matching:
                    continue
                merged.append({
                    "operation_type": operation_type,
                    "direction": matching[0]["direction"],
                    "condition": matching[0]["condition"],
                    "support_ids": sorted({value for item in matching for value in item.get("evidence_ids", [])}),
                })
            return {"factors": merged}
        if task.startswith("factor_validation_generate"):
            factor_ids = [str(factor["factor_id"]) for factor in context.get("factors", [])]
            return {
                "predictions": [
                    {
                        "id": str(record["id"]),
                        "factor_conditioned": " ".join(re.findall(r"[A-Za-z0-9]+", str(record["abstract"]))[:5]).title(),
                        "used_factor_ids": factor_ids[:2],
                    }
                    for record in context.get("records", [])
                ]
            }
        if task == "factors":
            if context.get("factor_quality_selection"):
                count = int(context.get("factor_proposal_count", 5))
                evidence_ids = [str(item["id"]) for item in profiles[:2]]
                if len(evidence_ids) < 2:
                    evidence_ids = [str(best["id"]), "mock-second-evidence"]
                types = ["structure", "compression", "structure", "content", "compression"]
                patterns = [
                    "method_for_problem",
                    "concise_noun_phrase",
                    "concept_colon_focus",
                    "content_focus",
                    "acronym_if_defined",
                ]
                return {
                    "factors": [{
                        "factor_id": f"f{index + 1}",
                        "type": types[index % len(types)],
                        "pattern_id": patterns[index % len(patterns)],
                        "evidence_ids": evidence_ids,
                        "evidence_summary": "Two historical titles support this stable pattern.",
                        "direction": f"Apply stable title pattern {index + 1} when supported.",
                        "condition": "when compatible with the current abstract",
                        "stability_score": 0.9 - index * 0.05,
                        "applicability_score": 0.9 - index * 0.04,
                        "risk_score": 0.1 + index * 0.03,
                    } for index in range(count)]
                }
            return {
                "factors": [
                    {"factor_id": "a1", "type": "content", "evidence_ids": [best["id"]], "evidence_summary": "Top retrieved work shares the main technical topic.", "direction": f"Preserve central terminology reflected by: {best['title']}", "condition": "scientific paper title generation"},
                    {"factor_id": "a2", "type": "structure", "evidence_ids": [best["id"]], "evidence_summary": "The retrieved title is a concise noun phrase.", "direction": "Use a concise, informative noun-phrase title.", "condition": "scientific paper title generation"},
                ]
            }
        if task.startswith("factor_selection"):
            factors = context.get("factor_candidates", [])
            selected = sorted(
                factors,
                key=lambda factor: (
                    factor.get("type") == "content",
                    -float(factor.get("stability_score", 0.0)),
                    -float(factor.get("applicability_score", 0.0)),
                ),
            )[:3]
            return {
                "selected_factor_ids": [factor["factor_id"] for factor in selected],
                "selection_reason": "Mock target-blind stable-factor selection.",
            }
        if task.startswith("seeds_task"):
            count = int(context.get("seed_count", 2))
            words = re.findall(r"[A-Za-z0-9]+", context.get("source_text", ""))[:6]
            topic = " ".join(words).title() or "The Current Problem"
            candidates = [f"A Study of {topic}", f"Methods for {topic}"]
            candidates.extend(
                f"{label} {topic}"
                for label in (
                    "Understanding",
                    "Analyzing",
                    "Learning",
                    "Improving",
                    "Revisiting",
                    "Modeling",
                    "Exploring",
                    "Towards",
                )
            )
            candidates.extend(
                f"Perspective {index}: {topic}"
                for index in range(1, max(1, count - len(candidates)) + 1)
            )
            return {"candidates": candidates[:count]}
        if task.startswith("factor_free_task_seeds"):
            count = int(context.get("count", context.get("seed_count", 2)))
            words = re.findall(r"[A-Za-z0-9]+", context.get("current_input", context.get("source_text", "")))[:6]
            topic = " ".join(words).title() or "The Current Problem"
            return {"candidates": [f"Methods for {topic} {index + 1}" for index in range(count)]}
        if task.startswith("factor_free_profile_seeds"):
            count = int(context.get("count", context.get("seed_count", 2)))
            best = (context.get("retrieved_history") or [{}])[0]
            title = str(best.get("output", "A Personalized Scholarly Title"))
            return {"candidates": [f"{title} {index + 1}" for index in range(count)]}
        if task.startswith("simple_conditional_trace"):
            parents = context.get("parents", [])
            history = context.get("retrieved_history", [])
            return {"traces": [{
                "parent_id": str(parent.get("parent_id", "")),
                "evidence_ids": [str(history[0]["id"])] if history else [],
                "edit_reason": "The visible history supports a concise edit that applies to this input." if history else "",
                "edit_action": "Preserve the contribution and refine the parent wording." if history else "",
            } for parent in parents]}
        if task.startswith("global_idpo_teacher_crossover"):
            left = context.get("parent_a", {})
            return {"evidence_ids": [], "edit_reason": "Combine compatible parent wording.", "edit_action": "Keep the clearest sequence-level formulation.", "output": str(left.get("text", ""))}
        if task == "seeds_factor":
            count = int(context.get("seed_count", 2))
            candidates = [best["title"], f"A Study of {best['title']}"]
            candidates.extend(
                f"Personalized Perspective {index}: {best['title']}"
                for index in range(1, max(1, count - len(candidates)) + 1)
            )
            return {"candidates": candidates[:count]}
        if task == "mutation":
            parent = context["parent"]
            direction = context["factor"]["direction"]
            return {"operation": "mutation", "operation_trace": f"Revise the parent according to: {direction}", "output": best["title"] if best["title"] != parent else f"Improved {parent}"}
        if task.startswith("teacher_listwise_rank"):
            candidate_ids = list(context.get("candidates", []))
            return {
                "selected_id": candidate_ids[0],
                "ranking": [
                    {"candidate_id": candidate_id, "score": 100 - index, "reason": "Mock deterministic order."}
                    for index, candidate_id in enumerate(candidate_ids)
                ],
            }
        if task.startswith("candidate_slate"):
            count = int(context.get("slate_count", 6))
            strategies = ["content", "content", "user_style", "user_style", "compressed", "compressed"]
            return {
                "candidates": [
                    {"strategy": strategies[index], "title": f"Mock Candidate Title {index + 1}"}
                    for index in range(count)
                ]
            }
        if task.startswith("candidate_pipeline_slate"):
            return {
                "task_seeds": ["Mock Task Seed One", "Mock Task Seed Two"],
                "factor_seeds": ["Mock Factor Seed One", "Mock Factor Seed Two"],
                "mutations": [
                    {"parent": parent, "factor_id": factor_id, "title": f"Mock Mutation {index + 1}"}
                    for index, (parent, factor_id) in enumerate(
                        (
                            ("task_0", "f1"),
                            ("task_1", "f1"),
                            ("factor_0", "f2"),
                            ("factor_1", "f2"),
                            ("task_0", "f3"),
                            ("factor_0", "f3"),
                        )
                    )
                ],
            }
        if task in {"rpm_history_features", "rpm_target_features"}:
            words = re.findall(r"[A-Za-z][A-Za-z0-9-]+", context.get("abstract", ""))
            unique = []
            for word in words:
                if word.lower() not in {value.lower() for value in unique}:
                    unique.append(word)
                if len(unique) == 3:
                    break
            return {
                "features": [
                    {"feature_name": word, "context": f"The abstract discusses {word}."}
                    for word in (unique or ["research topic"])
                ]
            }
        if task == "rpm_factors":
            count = int(context.get("rpm_factor_count", 2))
            return {"factors": [f"Mock Factor {index + 1}" for index in range(count)]}
        if task in {"rpm_factor_assignment", "rpm_target_factor_assignment"}:
            factor_ids = [factor["factor_id"] for factor in context.get("rpm_factors", [])]
            # The faithful LaMP-5 adapter passes plain factor names during
            # historical factorization and target preprocessing.  Keep the
            # mock useful for both representations.
            if not factor_ids:
                factor_ids = list(context.get("factors", []))
            return {"assignments": [1] if factor_ids else [0]}
        if task == "rpm_reasoning_memory":
            return {"reasoning": "The historical features and title pattern support this observed concise title."}
        if task == "rpm_inference":
            selected = set(context.get("rpm_retrieved_profile_ids", []))
            title = next(
                (
                    history["title"]
                    for history in context.get("histories", [])
                    if history["id"] in selected
                ),
                "A Personalized Scholarly Title",
            )
            return {"reasoning": "Apply the retrieved author-specific title pattern.", "predicted_title": title}
        raise ValueError(f"Unknown mock task: {task}")
