"""在相同 Per-Pcs Test 前 50 用户上评估 Qwen3-32B RAG/PAG 基线。

RAG：当前 Query + BM25 Top-k 历史输入/输出示例。
PAG：先从用户完整历史标题构建稳定 Profile，再输入当前 Query + Profile。

两个基线都只生成一个最终标题，不使用 Gold、候选池、Editor 或 Ranker。Gold
只在所有 API 调用完成后用于离线指标计算。脚本支持 API 缓存、结果 checkpoint
和失败任务多轮重试，适合高并发长时间运行。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from common.concurrency import run_bounded  # noqa: E402
from common.metrics import (  # noqa: E402
    corpus_bleu,
    corpus_score_with_ci,
    score,
)
from pipeline_common import (  # noqa: E402
    load_config,
    normalized_text,
    read_jsonl,
    resolve_path,
    stage_path,
    teacher_client,
    write_json,
    write_jsonl,
)


def _select_rows(rows: list[dict[str, Any]], user_limit: int) -> list[dict[str, Any]]:
    users = sorted({str(row["user_id"]) for row in rows})
    selected = set(users[:user_limit] if user_limit > 0 else users)
    return [row for row in rows if str(row["user_id"]) in selected]


def _targets_by_user(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        key = normalized_text(row.get("target", ""))
        if key:
            values[str(row["user_id"])].add(key)
    return values


def _safe_profile_titles(
    row: dict[str, Any], forbidden_targets: set[str]
) -> list[dict[str, str]]:
    """返回完整历史标题，并防御性排除任意当前 Test Gold 的异常重复。"""

    values = []
    seen = set()
    for item in row.get("profile", []):
        title = str(item.get("title", "")).strip()
        key = normalized_text(title)
        if not title or not key or key in seen or key in forbidden_targets:
            continue
        values.append({"id": str(item.get("id", "")), "title": title})
        seen.add(key)
    return values


def _safe_retrieved_history(
    row: dict[str, Any], forbidden_targets: set[str], top_k: int
) -> list[dict[str, str]]:
    values = []
    seen = set()
    for item in row.get("retrieved_profile", []):
        title = str(item.get("title", "")).strip()
        key = normalized_text(title)
        if not title or not key or key in seen or key in forbidden_targets:
            continue
        values.append(
            {
                "id": str(item.get("id", "")),
                "input": str(item.get("abstract", "")).strip(),
                "output": title,
            }
        )
        seen.add(key)
        if len(values) >= top_k:
            break
    return values


def _profile_prompt(user_id: str, histories: list[dict[str, str]]) -> str:
    return f"""You are building a stable writing profile for one user in a personalized generation benchmark.

Infer only recurring output preferences supported by the user's complete historical outputs. Focus on reusable choices such as output structure, length, phrasing, specificity, punctuation, and information selection. Do not summarize research topics, copy individual titles, or invent preferences. Return one concise profile with 4-8 evidence-grounded tendencies.

USER_ID: {user_id}
HISTORICAL_OUTPUTS:
{json.dumps(histories, ensure_ascii=False)}

Return JSON only:
{{"profile": "concise reusable profile"}}"""


def _rag_prompt(row: dict[str, Any], histories: list[dict[str, str]]) -> str:
    return f"""Generate the best scholarly paper title for CURRENT_ABSTRACT.

Use RETRIEVED_HISTORY as examples of this user's preferred title style and information selection, while keeping every factual claim faithful to CURRENT_ABSTRACT. Do not copy unrelated historical content. Return exactly one concise title and no explanation.

CURRENT_ABSTRACT:
{row['source_text']}

RETRIEVED_HISTORY:
{json.dumps(histories, ensure_ascii=False)}

Return JSON only:
{{"title": "final title"}}"""


def _pag_prompt(row: dict[str, Any], profile: str) -> str:
    return f"""Generate the best scholarly paper title for CURRENT_ABSTRACT.

Follow the reusable preferences in USER_PROFILE when they apply, while keeping every factual claim faithful to CURRENT_ABSTRACT. The profile describes style and information-selection preferences, not facts to insert. Return exactly one concise title and no explanation.

CURRENT_ABSTRACT:
{row['source_text']}

USER_PROFILE:
{profile}

Return JSON only:
{{"title": "final title"}}"""


def _text_field(payload: Any, keys: tuple[str, ...]) -> str:
    if isinstance(payload, str):
        value = payload
    elif isinstance(payload, dict):
        value = next((payload[key] for key in keys if key in payload), "")
        if isinstance(value, list):
            value = "\n".join(str(item) for item in value)
        elif isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False)
    else:
        value = ""
    return str(value).strip().strip('"').strip()


def _parse_profile(payload: Any) -> str:
    value = _text_field(payload, ("profile", "summary", "user_profile", "output"))
    if not value or len(value) > 4000:
        raise ValueError("PAG profile 必须是 1--4000 字符的非空文本")
    return value


def _parse_title(payload: Any) -> str:
    value = _text_field(payload, ("title", "output", "prediction", "answer"))
    value = " ".join(value.split())
    if not value or len(value) > 300:
        raise ValueError("生成标题必须是 1--300 字符的非空单行文本")
    return value


def _schema_request(
    client,
    task: str,
    prompt: str,
    context: dict[str, Any],
    parser: Callable[[Any], str],
    schema_retries: int,
) -> str:
    error: Exception | None = None
    for attempt in range(schema_retries + 1):
        current = prompt
        if attempt:
            current += (
                "\n\nJSON SCHEMA RETRY: Return the requested complete JSON object only. "
                f"The previous response violated the schema: {error}"
            )
        current_task = f"{task}_schema_{attempt}"
        payload, _ = client.json(current_task, current, context)
        try:
            return parser(payload)
        except (TypeError, ValueError) as exc:
            client.invalidate(current_task, current)
            error = exc
    raise ValueError(f"Teacher schema retries exhausted: {error}")


def _read_map(path: Path, key: Callable[[dict[str, Any]], str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {key(row): row for row in read_jsonl(path)}


def _run_resumable(
    jobs: list[Any],
    worker: Callable[[Any], dict[str, Any]],
    on_success: Callable[[dict[str, Any]], None],
    *,
    concurrency: int,
    rounds: int,
    label: str,
) -> None:
    pending = list(jobs)
    for round_index in range(1, rounds + 1):
        failures = []

        def safe_worker(job: Any) -> dict[str, Any]:
            try:
                return {"ok": True, "value": worker(job)}
            except Exception as exc:  # 单个接口故障不应丢掉已完成的数百条结果。
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        def accept(job: Any, result: dict[str, Any], completed: int) -> None:
            if result["ok"]:
                on_success(result["value"])
            else:
                failures.append(job)
                print(
                    f"{label} error round={round_index} completed={completed}/{len(pending)} "
                    f"job={job} error={result['error']}",
                    flush=True,
                )

        run_bounded(
            pending,
            safe_worker,
            accept,
            max_workers=concurrency,
            thread_name_prefix=label,
        )
        if not failures:
            return
        pending = failures
        print(f"{label} retry round={round_index} pending={len(pending)}", flush=True)
        time.sleep(min(5 * round_index, 30))
    raise RuntimeError(f"{label} still has {len(pending)} failed jobs after {rounds} rounds")


def _paired_bootstrap(values: list[float], samples: int = 10000) -> dict[str, float]:
    rng = random.Random(42)
    means = []
    for _ in range(samples):
        means.append(statistics.fmean(values[rng.randrange(len(values))] for _ in values))
    means.sort()
    return {
        "mean": statistics.fmean(values),
        "low": means[math.floor(0.025 * (samples - 1))],
        "high": means[math.ceil(0.975 * (samples - 1))],
    }


def _evaluate(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    row_by_id = {str(row["id"]): row for row in rows}
    by_method: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for item in predictions:
        by_method[str(item["method"])][str(item["sample_id"])] = item

    report: dict[str, Any] = {"queries": len(rows), "users": len({r['user_id'] for r in rows})}
    per_example: dict[str, dict[str, dict[str, float]]] = {}
    for method in ("rag", "pag"):
        ordered = [by_method[method][str(row["id"])] for row in rows]
        outputs = [item["prediction"] for item in ordered]
        targets = [str(row["target"]) for row in rows]
        example_scores = [score(output, target) for output, target in zip(outputs, targets)]
        per_example[method] = {
            str(row["id"]): value for row, value in zip(rows, example_scores)
        }
        user_values: dict[str, list[dict[str, float]]] = defaultdict(list)
        for row, value in zip(rows, example_scores):
            user_values[str(row["user_id"])].append(value)
        report[method] = {
            "rouge": corpus_score_with_ci(outputs, targets),
            "sacrebleu": corpus_bleu(outputs, targets),
            "query_macro": {
                metric: statistics.fmean(value[metric] for value in example_scores)
                for metric in ("rouge_1", "rouge_l")
            },
            "user_macro": {
                metric: statistics.fmean(
                    statistics.fmean(value[metric] for value in values)
                    for values in user_values.values()
                )
                for metric in ("rouge_1", "rouge_l")
            },
        }

    comparison = {}
    for metric in ("rouge_1", "rouge_l"):
        deltas = [
            per_example["pag"][str(row["id"])][metric]
            - per_example["rag"][str(row["id"])][metric]
            for row in rows
        ]
        comparison[metric] = {
            "pag_minus_rag": _paired_bootstrap(deltas),
            "pag_wins": sum(value > 1e-12 for value in deltas),
            "ties": sum(abs(value) <= 1e-12 for value in deltas),
            "rag_wins": sum(value < -1e-12 for value in deltas),
        }
    report["comparison"] = comparison
    return report


def run(config: dict[str, Any]) -> Path:
    settings = config["baseline"]
    source_split = str(settings.get("source_split", "test"))
    rows = _select_rows(
        read_jsonl(stage_path(config, source_split, "retrieve")),
        int(settings.get("user_limit", 0)),
    )
    if not rows:
        raise ValueError("RAG/PAG baseline 没有选中任何 Test Query")
    targets_by_user = _targets_by_user(rows)
    users = sorted({str(row["user_id"]) for row in rows})
    first_by_user = {user: next(row for row in rows if str(row["user_id"]) == user) for user in users}

    output_dir = resolve_path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "pag_profiles.jsonl"
    prediction_path = output_dir / "predictions.jsonl"
    report_path = output_dir / "metrics.json"
    profiles = _read_map(profile_path, lambda row: str(row["user_id"]))
    predictions = _read_map(
        prediction_path, lambda row: f"{row['method']}:{row['sample_id']}"
    )
    client = teacher_client(config)
    concurrency = int(settings.get("concurrency", 1))
    schema_retries = int(settings.get("schema_retries", 2))
    rounds = int(settings.get("failed_job_rounds", 3))
    checkpoint_every = int(settings.get("checkpoint_every", 10))

    profile_jobs = [user for user in users if user not in profiles]
    profile_completed = 0

    def build_profile(user: str) -> dict[str, Any]:
        histories = _safe_profile_titles(first_by_user[user], targets_by_user[user])
        prompt = _profile_prompt(user, histories)
        value = _schema_request(
            client,
            "rag_pag_profile",
            prompt,
            {"user_id": user, "history_count": len(histories)},
            _parse_profile,
            schema_retries,
        )
        return {"user_id": user, "history_count": len(histories), "profile": value}

    def save_profile(value: dict[str, Any]) -> None:
        nonlocal profile_completed
        profiles[str(value["user_id"])] = value
        profile_completed += 1
        if profile_completed % checkpoint_every == 0 or len(profiles) == len(users):
            write_jsonl(profile_path, [profiles[user] for user in users if user in profiles])
        print(
            f"PAG profile progress {len(profiles)}/{len(users)} user={value['user_id']} "
            f"history={value['history_count']}",
            flush=True,
        )

    print(
        f"RAG/PAG baseline queries={len(rows)} users={len(users)} concurrency={concurrency} "
        f"pending_profiles={len(profile_jobs)}",
        flush=True,
    )
    _run_resumable(
        profile_jobs,
        build_profile,
        save_profile,
        concurrency=concurrency,
        rounds=rounds,
        label="pag-profile",
    )

    jobs = [
        (method, row)
        for method in ("rag", "pag")
        for row in rows
        if f"{method}:{row['id']}" not in predictions
    ]
    generated = 0

    def generate(job: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        method, row = job
        user = str(row["user_id"])
        if method == "rag":
            history = _safe_retrieved_history(
                row, targets_by_user[user], int(settings.get("rag_top_k", 8))
            )
            prompt = _rag_prompt(row, history)
            context_size = len(history)
        else:
            prompt = _pag_prompt(row, str(profiles[user]["profile"]))
            context_size = int(profiles[user]["history_count"])
        title = _schema_request(
            client,
            f"rag_pag_{method}_generate",
            prompt,
            {"method": method, "sample_id": row["id"], "user_id": user},
            _parse_title,
            schema_retries,
        )
        return {
            "method": method,
            "sample_id": str(row["id"]),
            "user_id": user,
            "prediction": title,
            "target": str(row["target"]),
            "context_size": context_size,
            "gold_visible_during_generation": False,
        }

    def save_prediction(value: dict[str, Any]) -> None:
        nonlocal generated
        key = f"{value['method']}:{value['sample_id']}"
        predictions[key] = value
        generated += 1
        if generated % checkpoint_every == 0 or len(predictions) == len(rows) * 2:
            ordered = [
                predictions[f"{method}:{row['id']}"]
                for method in ("rag", "pag")
                for row in rows
                if f"{method}:{row['id']}" in predictions
            ]
            write_jsonl(prediction_path, ordered)
        print(
            f"baseline generation progress {len(predictions)}/{len(rows) * 2} "
            f"method={value['method']} sample={value['sample_id']}",
            flush=True,
        )

    _run_resumable(
        jobs,
        generate,
        save_prediction,
        concurrency=concurrency,
        rounds=rounds,
        label="rag-pag-generation",
    )
    ordered_predictions = [
        predictions[f"{method}:{row['id']}"]
        for method in ("rag", "pag")
        for row in rows
    ]
    write_jsonl(prediction_path, ordered_predictions)
    report = _evaluate(rows, ordered_predictions)
    report["protocol"] = {
        "model": config["teacher"]["model"],
        "temperature": config["teacher"]["temperature"],
        "selected_users": users,
        "rag": "current query + BM25 Top-8 historical input/output examples",
        "pag": "current query + profile summarized once from complete historical outputs",
        "single_generation": True,
        "candidate_pool": False,
        "ranker": False,
        "test_gold_in_api_prompt": False,
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"RAG/PAG metrics -> {report_path}", flush=True)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-32B RAG/PAG 前50 Test用户基线")
    parser.add_argument("--config", default=str(HERE / "config_rag_pag_test_first50.yaml"))
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()

