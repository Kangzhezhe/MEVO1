"""Build stable residual-rewrite factors from each sample's full profile."""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from common.concurrency import BoundedJobError, run_bounded
from common.factor_validation import _drafts, _extract, _merge, _unique_profile
from common.teacher import TeacherClient
from common.utils import read_jsonl, write_jsonl


_NORMALIZE = re.compile(r"[^a-z0-9]+")
LogCallback = Callable[[str], None]


def _key(value: str) -> str:
    return _NORMALIZE.sub("", str(value).lower())


def _target_isolated_profile(row: dict[str, Any]) -> tuple[list[dict[str, str]], int]:
    target_key = _key(row.get("target", ""))
    query_key = _key(row.get("source_text", ""))
    profiles = _unique_profile(row)
    retained = [
        profile
        for profile in profiles
        if (not target_key or _key(profile["title"]) != target_key)
        and (not query_key or _key(profile["abstract"]) != query_key)
    ]
    return retained, len(profiles) - len(retained)


def _item_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return sum(len(item) for item in value.values() if isinstance(item, list))
    return 0


class _ProgressTeacherClient:
    """Add request-level progress without making every Teacher run verbose."""

    def __init__(
        self,
        client: TeacherClient,
        sample_id: str,
        log: LogCallback,
    ) -> None:
        self._client = client
        self.config = client.config
        self.sample_id = sample_id
        self.log = log
        self.request_count = 0

    def json(
        self,
        task: str,
        prompt: str,
        context: dict[str, Any],
    ) -> tuple[Any, str]:
        self.request_count += 1
        request_id = self.request_count
        input_items = max(
            (_item_count(context.get(key)) for key in ("records", "factor_candidates")),
            default=0,
        )
        cache_path = self._client._cache_path(task, prompt)
        if self.config["provider"] == "mock":
            source = "mock"
        else:
            source = "cache" if cache_path.exists() else "api"
        self.log(
            f"profile factor request sample={self.sample_id} request={request_id} "
            f"task={task} items={input_items} source={source} status=start"
        )
        started = time.monotonic()
        try:
            payload, raw = self._client.json(task, prompt, context)
        except Exception as error:
            elapsed = time.monotonic() - started
            self.log(
                f"profile factor request sample={self.sample_id} request={request_id} "
                f"task={task} source={source} status=error "
                f"elapsed={elapsed:.1f}s error={type(error).__name__}"
            )
            raise
        elapsed = time.monotonic() - started
        self.log(
            f"profile factor request sample={self.sample_id} request={request_id} "
            f"task={task} source={source} status=done "
            f"elapsed={elapsed:.1f}s output_items={_item_count(payload)}"
        )
        return payload, raw

    def _cache_path(self, task: str, prompt: str) -> Path:
        return self._client._cache_path(task, prompt)


def _build_one(
    row: dict[str, Any],
    settings: dict[str, Any],
    client: TeacherClient,
    log: LogCallback | None = None,
) -> dict[str, Any]:
    profiles, target_matches_filtered = _target_isolated_profile(row)
    if len(profiles) < int(settings["min_profile_count"]):
        raise ValueError(
            f"sample={row['id']} has {len(profiles)} usable profile records; "
            f"requires {settings['min_profile_count']}"
        )
    sample_id = str(row["id"])
    active_client = (
        _ProgressTeacherClient(client, sample_id, log) if log is not None else client
    )
    if log is not None:
        batches = (len(profiles) + int(settings["batch_size"]) - 1) // int(
            settings["batch_size"]
        )
        log(
            f"profile factor user sample={sample_id} status=start "
            f"profiles={len(profiles)} batches={batches}"
        )
    drafts = _drafts(profiles, settings, active_client)
    if log is not None:
        log(
            f"profile factor phase sample={sample_id} phase=drafts status=done "
            f"records={len(drafts)} requests={active_client.request_count}"
        )
    proposals = _extract(drafts, settings, active_client)
    if log is not None:
        log(
            f"profile factor phase sample={sample_id} phase=extract status=done "
            f"proposals={len(proposals)} requests={active_client.request_count}"
        )
        log(
            f"profile factor phase sample={sample_id} phase=merge status=start "
            f"proposals={len(proposals)}"
        )
    factors = _merge(proposals, settings, active_client)
    if log is not None:
        log(
            f"profile factor phase sample={sample_id} phase=merge status=done "
            f"factors={len(factors)} requests={active_client.request_count}"
        )
    if not factors:
        raise ValueError(f"sample={row['id']} produced no stable factors")
    for factor in factors:
        factor["evidence_ids"] = list(factor["support_ids"])
        factor["evidence_summary"] = (
            f"Stable residual rewrite direction supported by "
            f"{factor['support_count']} full-profile records."
        )
    row["factors"] = factors
    row["factor_metadata"] = {
        "method": "full_profile_residual_rewrite_bank_v1",
        "profile_records": len(profiles),
        "target_matches_filtered": target_matches_filtered,
        "proposal_count": len(proposals),
        "factor_count": len(factors),
        "prompt_version": str(settings["prompt_version"]),
        "model": str(client.config["model"]),
        "target_used_for_factor_construction": False,
        "historical_titles_used": True,
    }
    return row


def build(
    source: Path,
    destination: Path,
    config: dict[str, Any],
    client: TeacherClient,
) -> None:
    rows = read_jsonl(source)
    settings = dict(config["profile_factors"])
    settings["seed"] = int(config["project"]["seed"])
    concurrency = int(settings.get("concurrency", 1))
    results: list[dict[str, Any] | None] = [None] * len(rows)
    jobs = list(enumerate(rows))
    log_lock = threading.Lock()
    print(
        f"profile factor samples={len(rows)}, concurrency={concurrency}, "
        f"batch_size={settings['batch_size']}",
        flush=True,
    )

    def log(message: str) -> None:
        with log_lock:
            print(message, flush=True)

    def worker(job: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        return _build_one(job[1], settings, client, log)

    def on_result(
        job: tuple[int, dict[str, Any]], result: dict[str, Any], completed: int
    ) -> None:
        results[job[0]] = result
        print(
            f"profile factor progress {completed}/{len(rows)} "
            f"sample={job[1]['id']} factors={len(result['factors'])}",
            flush=True,
        )

    try:
        run_bounded(
            jobs,
            worker,
            on_result,
            max_workers=concurrency,
            thread_name_prefix="mevo-profile-factor",
        )
    except BoundedJobError as failure:
        raise RuntimeError(
            f"profile factor construction failed for sample={failure.job[1]['id']}: "
            f"{failure.error}"
        ) from failure.error

    complete = [row for row in results if row is not None]
    write_jsonl(destination, complete)
    print(f"profile factors for {len(complete)} samples -> {destination}")
