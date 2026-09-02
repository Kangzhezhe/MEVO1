"""Faithful RPM pipeline with a LaMP-5 task adapter.

This module mirrors the stage order of the released RPM implementation:

    11 feature extraction -> 12 iterative factor construction ->
    13 feature/factor postprocessing -> 14 factor statistics ->
    15 reasoning personalization -> 21 target preprocessing ->
    22 feature-retrieved black-box inference.

Only the task-specific surface is adapted: LaMP-5 supplies paper abstracts and
historical paper titles instead of product reviews and 1--5 ratings.  The
algorithm does not use MeVO candidates, mutations, ranker data or a Ranker.
Historical titles are labels for historical reasoning memories; the current
sample's target title is never passed to a Teacher prompt.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from common.teacher import TeacherClient
from common.utils import load_config, read_jsonl, resolve_path, write_jsonl


_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value).strip()
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _validate_features(payload: Any) -> None:
    if not isinstance(payload, dict) or not isinstance(payload.get("features"), list):
        raise ValueError("expected {features: [...]} object")
    if not payload["features"]:
        raise ValueError("features must not be empty")
    for index, feature in enumerate(payload["features"]):
        if not isinstance(feature, dict) or not str(feature.get("feature_name", "")).strip():
            raise ValueError(f"features[{index}] has no feature_name")
        if "context" not in feature:
            raise ValueError(f"features[{index}] has no context")


def _validate_factor_names(payload: Any) -> None:
    if not isinstance(payload, dict) or not isinstance(payload.get("factors"), list):
        raise ValueError("expected {factors: [...]} object")
    if not _unique(str(x) for x in payload["factors"]):
        raise ValueError("factor list is empty")


def _validate_assignment(payload: Any) -> None:
    if not isinstance(payload, dict) or "assignments" not in payload:
        raise ValueError("expected {assignments: ...} object")


def _validate_reasoning(payload: Any) -> None:
    if not isinstance(payload, dict) or not str(payload.get("reasoning", "")).strip():
        raise ValueError("expected non-empty reasoning")


def _validate_prediction(payload: Any) -> None:
    if not isinstance(payload, dict) or not str(payload.get("predicted_title", "")).strip():
        raise ValueError("expected non-empty predicted_title")


class RPMLaMP5:
    """Run the RPM stages on normalized LaMP-5 JSONL rows."""

    def __init__(self, config: dict[str, Any], force: bool = False):
        self.config = config
        self.settings = config.get("rpm", {})
        experiment = config.get("experiment", {})
        output_value = self.settings.get("output_dir") or experiment.get("result_dir")
        if not output_value:
            raise ValueError("rpm.output_dir or experiment.result_dir is required")
        self.output_dir = resolve_path(output_value)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cache_value = self.settings.get("cache_dir", "dataset/cache/rpm-lamp5-visgpt")
        teacher_config = dict(config["teacher"])
        teacher_config["cache_dir"] = str(resolve_path(cache_value))
        self.teacher = TeacherClient(teacher_config, resolve_path(cache_value))
        self.force = force
        self.workers = max(1, int(self.settings.get("teacher_concurrency", 4)))
        self.schema_retries = max(0, int(self.settings.get("schema_retries", 3)))
        self.prompt_dir = resolve_path(
            self.settings.get("prompt_dir", "prompt/rpm_lamp5")
        )
        self.audit: list[dict[str, Any]] = []
        self._retriever: tuple[Any, Any, Any] | None = None

    @staticmethod
    def log(message: str) -> None:
        print(f"[RPM] {message}", flush=True)

    def prompt(self, name: str, **fields: Any) -> str:
        template = (self.prompt_dir / name).read_text(encoding="utf-8")
        return template.format(**fields)

    def call(
        self,
        task: str,
        prompt: str,
        context: dict[str, Any],
        validator: Callable[[Any], None],
        *,
        sample_id: str,
        target: str = "",
    ) -> Any:
        """Call Teacher with schema retries and record a leakage audit entry."""
        # The target title is allowed only when explaining a historical item in
        # stage 15.  All target-side stages pass target="" here.
        target_text = str(target or "").strip()
        if not target_text and context.get("target_title"):
            target_text = str(context["target_title"]).strip()
        target_in_prompt = bool(target_text and target_text.casefold() in prompt.casefold())
        self.audit.append(
            {
                "sample_id": str(sample_id),
                "task": task,
                "target_in_prompt": target_in_prompt,
                "target_allowed": task == "rpm_reasoning_memory",
            }
        )
        if target_in_prompt and task != "rpm_reasoning_memory":
            raise RuntimeError(
                f"gold title leakage detected in task={task}, sample={sample_id}"
            )

        last_error: Exception | None = None
        for attempt in range(self.schema_retries + 1):
            attempt_prompt = prompt
            if attempt:
                attempt_prompt += (
                    "\n\nIMPORTANT: Your previous response failed schema validation. "
                    "Return a complete JSON object only, with all required fields."
                )
                if task in {"rpm_history_features", "rpm_target_features"}:
                    attempt_prompt += (
                        " The features array must contain at least one concrete feature "
                        "from the abstract; never return an empty features array."
                    )
            try:
                parsed, _ = self.teacher.json(task, attempt_prompt, context)
                validator(parsed)
                return parsed
            except Exception as exc:  # schema and transport failures are retried by design
                last_error = exc
                self.log(f"retry task={task} sample={sample_id} attempt={attempt + 1}: {exc}")
        raise RuntimeError(
            f"RPM Teacher task={task} failed after {self.schema_retries + 1} attempts: {last_error}"
        ) from last_error

    def parallel(self, jobs: list[Callable[[], Any]]) -> list[Any]:
        if not jobs:
            return []
        results: list[Any] = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(job): index for index, job in enumerate(jobs)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results

    def _stage_rows(self, filename: str) -> list[dict[str, Any]]:
        path = self.output_dir / filename
        if path.exists() and not self.force:
            return read_jsonl(path)
        return []

    def _write_stage(self, filename: str, rows: list[dict[str, Any]]) -> None:
        write_jsonl(self.output_dir / filename, rows)

    def stage11_history_features(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        existing = {str(row["sample_id"]): row for row in self._stage_rows("11_history_features.jsonl")}
        output: list[dict[str, Any]] = []
        self.log(f"stage 11 start: {len(rows)} samples")
        for row_index, row in enumerate(rows, 1):
            sample_id = str(row["id"])
            if sample_id in existing:
                output.append(existing[sample_id])
                self.log(f"stage 11 sample {row_index}/{len(rows)} reused: {sample_id}")
                continue
            profiles = row.get("profile", [])
            jobs = []
            for item in profiles:
                item_id = str(item["id"])
                abstract = str(item.get("abstract", "")).strip()
                title = str(item.get("title", "")).strip()
                if abstract.casefold() in {"no abstract available", "no abstract available."}:
                    # The official LaMP data contains a small number of
                    # placeholder abstracts.  RPM's strict extractor would
                    # receive no semantic evidence here; retain the history
                    # item for accounting but do not invent a feature.
                    jobs.append(
                        lambda item_id=item_id, abstract=abstract, title=title: {
                            "id": item_id,
                            "abstract": abstract,
                            "title": title,
                            "features": [],
                            "skipped": "missing_abstract",
                        }
                    )
                    continue
                prompt = self.prompt("11_extract_feature.txt", abstract=abstract)
                jobs.append(
                    lambda item_id=item_id, abstract=abstract, title=title, prompt=prompt, sample_id=sample_id: self._history_feature_call(
                        sample_id, item_id, abstract, title, prompt
                    )
                )
            history = self.parallel(jobs)
            value = {"sample_id": sample_id, "query": row["source_text"], "target": row["target"], "history": history}
            output.append(value)
            self._write_stage("11_history_features.jsonl", output)
            self.log(f"stage 11 sample {row_index}/{len(rows)} done: {sample_id}, history={len(history)}")
        return output

    def _history_feature_call(self, sample_id: str, item_id: str, abstract: str, title: str, prompt: str) -> dict[str, Any]:
        try:
            payload = self.call(
                "rpm_history_features",
                prompt,
                {"abstract": abstract},
                _validate_features,
                sample_id=f"{sample_id}:{item_id}",
            )
        except RuntimeError as error:
            # One malformed response must not discard thousands of completed
            # profile calls.  After all request/JSON/schema retries are
            # exhausted, retain the history item without inventing features.
            return {
                "id": item_id,
                "abstract": abstract,
                "title": title,
                "features": [],
                "skipped": "teacher_failure",
                "error": str(error),
            }
        features = []
        for feature in payload["features"]:
            features.append(
                {
                    "feature_name": str(feature["feature_name"]).strip(),
                    "context": str(feature.get("context", "")).strip(),
                    "item_id": item_id,
                }
            )
        return {"id": item_id, "abstract": abstract, "title": title, "features": features}

    def _propose_factors(self, sample_id: str, features: list[dict[str, Any]], previous: list[str], uncovered: int) -> list[str]:
        focus = features[:32]
        if uncovered and len(features) > 32:
            focus = features[:uncovered]
        prompt = self.prompt(
            "12_propose_factors.txt",
            num_factors=int(self.settings.get("proposer_num_factors", 16)),
            feature_examples="\n".join(
                f"Feature {i + 1}: {f['feature_name']} — Context: {f['context']}"
                for i, f in enumerate(focus)
            ),
            previous_factors="\n".join(f"- {x}" for x in previous[:4]) or "(none)",
            iteration_note=(
                f"There are {uncovered} uncovered features. Propose new factors for them."
                if uncovered
                else ""
            ),
        )
        payload = self.call(
            "rpm_factors",
            prompt,
            {"features": focus, "previous_factors": previous},
            _validate_factor_names,
            sample_id=sample_id,
        )
        return _unique(str(value) for value in payload["factors"])

    def _assign_factor(self, sample_id: str, feature: dict[str, Any], factors: list[str], task: str = "rpm_factor_assignment") -> list[int]:
        if not factors:
            return []
        prompt = self.prompt(
            "12_assign_to_factors.txt",
            feature=f"{feature['feature_name']} (Context: {feature['context']})",
            formatted_factors="\n".join(f"{i + 1}. {factor}" for i, factor in enumerate(factors)),
        )
        payload = self.call(
            task,
            prompt,
            {"feature": feature, "factors": factors},
            _validate_assignment,
            sample_id=sample_id,
        )
        vector = [0] * len(factors)
        value = payload.get("assignments")
        # The official LaMP-3 code accepts a list or a string but uses only the
        # first valid assignment; retain that one-factor-per-feature behavior.
        if isinstance(value, list) and value:
            value = value[0]
        if isinstance(value, str):
            value = value.replace(" ", "").split(",")[0]
        try:
            index = int(value) - 1
        except (TypeError, ValueError):
            index = -1
        if 0 <= index < len(vector):
            vector[index] = 1
        return vector

    @staticmethod
    def _prune_factors(
        factors: list[str], matrix: list[list[int]], min_fraction: float, max_fraction: float, iteration: int
    ) -> tuple[list[str], list[list[int]], list[int]]:
        """The same coverage-size pruning and fallback policy as RPM 12_construct_factor."""
        if not factors:
            return [], [[] for _ in matrix], []
        n = len(matrix)
        min_features = max(1, int(n * min_fraction))
        max_features = int(n * max_fraction)
        if iteration > 0:
            min_features = 1
        keep = [
            index
            for index in range(len(factors))
            if min_features <= sum(row[index] for row in matrix) <= max_features
        ]
        if iteration > 0 and len(keep) < 10:
            keep = [
                index for index in range(len(factors))
                if index not in keep and sum(row[index] for row in matrix) > 0
            ] + keep
        if not keep:
            ordered = sorted(
                range(len(factors)), key=lambda index: sum(row[index] for row in matrix), reverse=True
            )
            keep = [index for index in ordered[:4] if sum(row[index] for row in matrix) > 0]
        pruned = [factors[index] for index in keep]
        pruned_matrix = [[row[index] for index in keep] for row in matrix]
        return pruned, pruned_matrix, keep

    @staticmethod
    def _select_factors(
        matrix: list[list[int]],
        factors: list[str],
        iteration: int,
        overlap_penalty: float,
        not_cover_penalty: float,
        preferred_indices: list[int] | None = None,
        new_indices: list[int] | None = None,
    ) -> list[int]:
        """Port of RPM's greedy coverage/overlap selector."""
        if not factors or not matrix:
            return []
        n = len(matrix)
        d = len(factors)
        limit = min(8 + 2 * iteration, d)
        selected: list[int] = []
        current = [0] * n

        for group in (new_indices or [], preferred_indices or []):
            for index in group:
                if index >= d or index in selected:
                    continue
                if group is preferred_indices:
                    new_coverage = sum(
                        min(matrix[row][index], 1 - min(current[row], 1)) for row in range(n)
                    )
                    if new_coverage <= 0:
                        continue
                elif sum(row[index] for row in matrix) <= 0:
                    continue
                selected.append(index)
                current = [current[row] + matrix[row][index] for row in range(n)]
                if len(selected) >= limit:
                    return selected

        while len(selected) < limit:
            best_index = -1
            best_score = -float("inf")
            for index in range(d):
                if index in selected:
                    continue
                new_coverage = [current[row] + matrix[row][index] for row in range(n)]
                new_features = sum(
                    min(matrix[row][index], 1 - min(current[row], 1)) for row in range(n)
                )
                overlap = sum(max(0, value - 1) for value in new_coverage)
                not_covered = n - sum(min(value, 1) for value in new_coverage)
                score = new_features * 2
                if iteration == 0:
                    score -= overlap_penalty * overlap - not_cover_penalty * not_covered
                else:
                    score -= overlap_penalty * 0.5 * overlap - not_cover_penalty * 0.8 * not_covered
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index < 0:
                break
            selected.append(best_index)
            current = [current[row] + matrix[row][best_index] for row in range(n)]

        if not selected:
            coverage = [sum(row[index] for row in matrix) for index in range(d)]
            selected = [index for index in sorted(range(d), key=lambda i: coverage[i], reverse=True)[:5] if coverage[index] > 0]
        return selected

    def _factorize(self, sample_id: str, feature_rows: list[dict[str, Any]]) -> dict[str, Any]:
        features = [feature for item in feature_rows for feature in item["features"]]
        if not features:
            return {"selected_factors": [], "factors": {}, "uncovered_features": []}
        max_rounds = int(self.settings.get("factor_max_rounds", 3))
        threshold = float(self.settings.get("factor_coverage_threshold", 0.9))
        overlap_penalty = float(self.settings.get("factor_overlap_penalty", 4.0))
        not_cover_penalty = float(self.settings.get("factor_not_cover_penalty", 8.0))
        all_factors: list[str] = []
        all_matrix: list[list[int]] = [[] for _ in features]
        selected_memory: dict[str, int] = {}
        selected: list[str] = []
        selected_matrix: list[list[int]] = [[] for _ in features]
        uncovered = [True] * len(features)

        for iteration in range(max_rounds):
            coverage = 1.0 - (sum(uncovered) / len(features))
            if coverage >= threshold:
                break
            uncovered_count = sum(uncovered)
            if iteration >= 1 and uncovered_count > 0.3 * len(features):
                overlap_penalty *= 0.5
            if iteration >= 1 and uncovered_count:
                focus = [feature for index, feature in enumerate(features) if uncovered[index]]
            else:
                focus = [feature for index, feature in enumerate(features) if uncovered[index]][:32]
                if len(focus) < 32:
                    focus.extend(
                        [feature for index, feature in enumerate(features) if not uncovered[index]][: 32 - len(focus)]
                    )
            # RPM's proposer samples at most 32 features before prompting.
            if len(focus) > 32:
                focus = random.Random(f"42:{sample_id}:{iteration}").sample(focus, 32)

            max_attempts = 3 if iteration > 0 and uncovered_count > 10 else 1
            new_factors: list[str] = []
            for _ in range(max_attempts):
                proposed = self._propose_factors(sample_id, focus, all_factors, uncovered_count)
                proposed = [factor for factor in proposed if factor not in all_factors and factor not in new_factors]
                if proposed:
                    new_factors = proposed
                    break
            if not new_factors:
                continue
            assignments = self.parallel(
                [lambda feature=feature: self._assign_factor(sample_id, feature, new_factors) for feature in features]
            )
            all_factors.extend(new_factors)
            for row_index, vector in enumerate(assignments):
                all_matrix[row_index].extend(vector)

            pruned_factors, pruned_matrix, _ = self._prune_factors(
                all_factors,
                all_matrix,
                float(self.settings.get("factor_min_fraction", 0.0)),
                float(self.settings.get("factor_max_fraction", 0.4)),
                iteration,
            )
            if not pruned_factors:
                continue
            new_factor_indices = [index for index, factor in enumerate(pruned_factors) if factor in new_factors]
            preferred_indices = [index for index, factor in enumerate(pruned_factors) if factor in selected_memory]
            selected_indices = self._select_factors(
                pruned_matrix,
                pruned_factors,
                iteration,
                overlap_penalty,
                not_cover_penalty,
                preferred_indices=preferred_indices,
                new_indices=new_factor_indices,
            )
            selected = [pruned_factors[index] for index in selected_indices]
            selected_matrix = [
                [row[index] for index in selected_indices] for row in pruned_matrix
            ]
            for factor in selected:
                selected_memory[factor] = selected_memory.get(factor, 0) + 1
            uncovered = [not any(row) for row in selected_matrix]

        if selected and any(uncovered):
            remaining = [features[index] for index, flag in enumerate(uncovered) if flag]
            assignments = self.parallel(
                [lambda feature=feature: self._assign_factor(sample_id, feature, selected) for feature in remaining]
            )
            cursor = 0
            for index, flag in enumerate(uncovered):
                if flag:
                    selected_matrix[index] = assignments[cursor]
                    cursor += 1
            uncovered = [not any(row) for row in selected_matrix]

        factor_map = {
            factor: [features[index] for index, row in enumerate(selected_matrix) if column < len(row) and row[column]]
            for column, factor in enumerate(selected)
        }
        return {
            "selected_factors": selected,
            "factors": factor_map,
            "uncovered_features": [features[index] for index, flag in enumerate(uncovered) if flag],
        }

    def stage12_factors(self, history_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        existing = {str(row["sample_id"]): row for row in self._stage_rows("12_factors.jsonl")}
        output: list[dict[str, Any]] = []
        self.log(f"stage 12 start: {len(history_rows)} samples")
        for row_index, row in enumerate(history_rows, 1):
            sample_id = str(row["sample_id"])
            value = existing.get(sample_id) or {"sample_id": sample_id, **self._factorize(sample_id, row["history"])}
            output.append(value)
            self._write_stage("12_factors.jsonl", output)
            self.log(f"stage 12 sample {row_index}/{len(history_rows)} done: {sample_id}, factors={len(value.get('selected_factors', []))}")
        return output

    def stage13_postprocess(self, history_rows: list[dict[str, Any]], factor_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.log(f"stage 13 start: {len(history_rows)} samples")
        factor_by_id = {str(row["sample_id"]): row for row in factor_rows}
        output: list[dict[str, Any]] = []
        for row in history_rows:
            factors = factor_by_id[str(row["sample_id"])]
            assignments: dict[tuple[str, str], list[str]] = {}
            for factor, features in factors.get("factors", {}).items():
                for feature in features:
                    assignments.setdefault((str(feature["item_id"]), str(feature["feature_name"])), []).append(factor)
            history = []
            for item in row["history"]:
                copied = dict(item)
                copied["features"] = []
                for feature in item["features"]:
                    key = (str(item["id"]), str(feature["feature_name"]))
                    copied["features"].append({**feature, "factors": assignments.get(key, ["Unclassified"])})
                history.append(copied)
            output.append({"sample_id": row["sample_id"], "history": history, "selected_factors": factors.get("selected_factors", [])})
        self._write_stage("13_feature_factors.jsonl", output)
        self.log("stage 13 complete")
        return output

    def stage14_statistics(self, rows: list[dict[str, Any]], factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.log(f"stage 14 start: {len(rows)} samples")
        factor_by_id = {str(row["sample_id"]): row for row in factors}
        output: list[dict[str, Any]] = []
        for row in rows:
            profile = row["history"]
            lengths = [len(_TOKEN.findall(str(item.get("title", "")))) for item in profile]
            stats: dict[str, Any] = {
                "total_profile_samples": len(profile),
                "average_title_token_count": sum(lengths) / len(lengths) if lengths else 0.0,
                "factors": {},
            }
            for factor in factor_by_id[str(row["sample_id"])].get("selected_factors", []):
                covered = [item for item in profile if any(factor in feature.get("factors", []) for feature in item["features"])]
                covered_lengths = [len(_TOKEN.findall(str(item.get("title", "")))) for item in covered]
                stats["factors"][factor] = {
                    "support_count": len(covered),
                    "support_fraction": len(covered) / len(profile) if profile else 0.0,
                    "average_title_token_count": sum(covered_lengths) / len(covered_lengths) if covered_lengths else 0.0,
                    "title_examples": [str(item.get("title", "")) for item in covered[:3]],
                }
            output.append({"sample_id": row["sample_id"], "statistics": stats})
        self._write_stage("14_factor_statistics.jsonl", output)
        self.log("stage 14 complete")
        return output

    def stage15_reasoning(self, rows: list[dict[str, Any]], stats_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stats_by_id = {str(row["sample_id"]): row["statistics"] for row in stats_rows}
        output: list[dict[str, Any]] = []
        self.log(f"stage 15 start: {len(rows)} samples")
        for row_index, row in enumerate(rows, 1):
            sample_id = str(row["sample_id"])
            stats = stats_by_id[sample_id]
            jobs = []
            for item in row["history"]:
                prompt = self.prompt(
                    "15_reasoning_personalization.txt",
                    abstract=item["abstract"],
                    features=_json(item["features"]),
                    factors=_json(stats),
                    actual_title=item.get("title", ""),
                )
                jobs.append(
                    lambda item=item, prompt=prompt, sample_id=sample_id: self._reasoning_call(
                        sample_id, item, stats, prompt
                    )
                )
            memories = self.parallel(jobs)
            output.append({"sample_id": sample_id, "memory": memories})
            self._write_stage("15_reasoning_memory.jsonl", output)
            self.log(f"stage 15 sample {row_index}/{len(rows)} done: {sample_id}, memories={len(memories)}")
        return output

    def _reasoning_call(self, sample_id: str, item: dict[str, Any], stats: dict[str, Any], prompt: str) -> dict[str, Any]:
        payload = self.call(
            "rpm_reasoning_memory",
            prompt,
            {"abstract": item["abstract"], "features": item["features"], "factors": stats, "target_title": item.get("title", "")},
            _validate_reasoning,
            sample_id=f"{sample_id}:{item['id']}",
            target=str(item.get("title", "")),
        )
        return {
            "id": item["id"],
            "abstract": item["abstract"],
            "title": item.get("title", ""),
            "features": item["features"],
            "reasoning": payload["reasoning"],
        }

    def stage21_target(self, rows: list[dict[str, Any]], factor_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        factor_by_id = {str(row["sample_id"]): row for row in factor_rows}
        existing = {str(row["sample_id"]): row for row in self._stage_rows("21_target_features.jsonl")}
        output: list[dict[str, Any]] = []
        self.log(f"stage 21 start: {len(rows)} samples")
        for row_index, row in enumerate(rows, 1):
            sample_id = str(row["id"])
            if sample_id in existing:
                output.append(existing[sample_id])
                continue
            prompt = self.prompt("21_extract_target_feature.txt", abstract=row["source_text"])
            target_features = self.call(
                "rpm_target_features",
                prompt,
                {"abstract": row["source_text"]},
                _validate_features,
                sample_id=sample_id,
                target=str(row.get("target", "")),
            )["features"]
            target_features = [
                {"feature_name": str(x["feature_name"]), "context": str(x.get("context", ""))}
                for x in target_features
            ]
            factors = factor_by_id[sample_id].get("selected_factors", [])
            assignments = self.parallel(
                [lambda feature=feature: self._assign_factor(sample_id, feature, factors, "rpm_target_factor_assignment") for feature in target_features]
            ) if factors else [[] for _ in target_features]
            for feature, vector in zip(target_features, assignments):
                feature["factors"] = [factor for factor, value in zip(factors, vector) if value]
            value = {"sample_id": sample_id, "abstract": row["source_text"], "features": target_features}
            output.append(value)
            self._write_stage("21_target_features.jsonl", output)
            self.log(f"stage 21 sample {row_index}/{len(rows)} done: {sample_id}")
        return output

    @staticmethod
    def _retrieval_text(item: dict[str, Any]) -> str:
        texts = []
        for feature in item.get("features", []):
            value = str(feature.get("feature_name", ""))
            factors = feature.get("factors", [])
            if factors:
                value += " [factors: " + ", ".join(map(str, factors)) + "]"
            if feature.get("context"):
                value += " [context: " + str(feature["context"]) + "]"
            texts.append(value)
        return " ".join(texts)

    def _retriever_model(self) -> tuple[Any, Any, Any]:
        """Load Contriever once, matching RPM's process-level encoder reuse."""
        if self._retriever is not None:
            return self._retriever
        model_name = str(self.settings.get("retrieval_model", "facebook/contriever"))
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("RPM retrieval requires torch and transformers") from exc
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        self._retriever = tokenizer, model, device
        return self._retriever

    def _retrieve(self, target: dict[str, Any], memories: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not memories:
            return []
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            raise RuntimeError("RPM retrieval requires torch and transformers") from exc
        tokenizer, model, device = self._retriever_model()
        texts = [self._retrieval_text(target)] + [self._retrieval_text(item) for item in memories]
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).bool()
        embeddings = (output.masked_fill(~mask, 0.0).sum(dim=1) / encoded["attention_mask"].sum(dim=1, keepdim=True)).cpu().numpy()
        similarities = np.dot(embeddings[0:1], embeddings[1:].T)[0]
        indices = np.argsort(similarities)[-top_k:][::-1]
        return [memories[int(index)] for index in indices]

    def stage22_inference(self, rows: list[dict[str, Any]], stats_rows: list[dict[str, Any]], memories_rows: list[dict[str, Any]], targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stats_by_id = {str(row["sample_id"]): row["statistics"] for row in stats_rows}
        memories_by_id = {str(row["sample_id"]): row["memory"] for row in memories_rows}
        targets_by_id = {str(row["sample_id"]): row for row in targets}
        top_k = int(self.settings.get("retrieval_top_k", 3))
        output: list[dict[str, Any]] = []
        self.log(f"stage 22 start: {len(rows)} samples")
        for row_index, row in enumerate(rows, 1):
            sample_id = str(row["id"])
            target = targets_by_id[sample_id]
            retrieved = self._retrieve(target, memories_by_id[sample_id], top_k)
            exemplars = []
            for index, memory in enumerate(retrieved, 1):
                exemplars.append(
                    f"Example {index}:\n"
                    f"historical abstract: {memory['abstract']}\n"
                    f"historical features: {_json(memory['features'])}\n"
                    f"personalized reasoning: {memory['reasoning']}\n"
                    f"historical title: {memory['title']}"
                )
            prompt = self.prompt(
                "22_bbox_inference.txt",
                abstract=row["source_text"],
                target_features=_json(target["features"]),
                user_factors=_json(stats_by_id[sample_id]),
                exemplars="\n\n".join(exemplars),
            )
            payload = self.call(
                "rpm_inference",
                prompt,
                {"abstract": row["source_text"], "target_features": target["features"], "factors": stats_by_id[sample_id], "histories": retrieved},
                _validate_prediction,
                sample_id=sample_id,
                target=str(row.get("target", "")),
            )
            output.append(
                {
                    "sample_id": sample_id,
                    "prediction": str(payload["predicted_title"]).strip(),
                    "reasoning": str(payload.get("reasoning", "")),
                    "retrieved_ids": [str(memory["id"]) for memory in retrieved],
                }
            )
            self._write_stage("22_predictions.jsonl", output)
            self.log(f"stage 22 sample {row_index}/{len(rows)} done: {sample_id}")
        return output

    def run(self) -> dict[str, Any]:
        input_path = resolve_path(self.settings["input_path"])
        rows = read_jsonl(input_path)
        self.log(f"run start: input={input_path}, samples={len(rows)}, output={self.output_dir}")
        history = self.stage11_history_features(rows)
        factors = self.stage12_factors(history)
        postprocessed = self.stage13_postprocess(history, factors)
        statistics = self.stage14_statistics(postprocessed, factors)
        memories = self.stage15_reasoning(postprocessed, statistics)
        targets = self.stage21_target(rows, factors)
        predictions = self.stage22_inference(rows, statistics, memories, targets)
        audit_path = self.output_dir / "leakage_audit.jsonl"
        write_jsonl(audit_path, self.audit)
        manifest = {
            "method": "RPM faithful stage order with LaMP-5 task adapter",
            "input_path": str(input_path),
            "sample_count": len(rows),
            "retrieval_model": self.settings.get("retrieval_model", "facebook/contriever"),
            "retrieval_top_k": int(self.settings.get("retrieval_top_k", 3)),
            "target_gold_in_nonhistorical_prompt": any(
                item["target_in_prompt"] and not item["target_allowed"] for item in self.audit
            ),
            "stage_files": sorted(path.name for path in self.output_dir.glob("*.jsonl")),
        }
        (self.output_dir / "run_manifest.json").write_text(_json(manifest), encoding="utf-8")
        self.log("run complete")
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Faithful RPM LaMP-5 baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = RPMLaMP5(config, force=args.force).run()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
