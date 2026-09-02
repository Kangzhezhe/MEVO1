"""公共配置、路径和 JSONL 工具。

所有编号脚本都通过这里加载 base + experiment 配置；``experiment.name`` 会
自动隔离 ranker_sets、result、logs，保证不同实验不会互相覆盖。
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root() / path


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


EXPERIMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _experiment_paths(config: dict[str, Any]) -> dict[str, Any]:
    experiment = config.get("experiment")
    if not isinstance(experiment, dict) or not experiment.get("name"):
        return config
    if not bool(experiment.get("auto_paths", True)):
        return config

    name = str(experiment["name"])
    if not EXPERIMENT_NAME.fullmatch(name):
        raise ValueError(
            "experiment.name must contain only letters, numbers, '.', '_' or '-'"
        )
    roots = config.get("paths", {})
    ranker_root = Path(str(roots.get("ranker_sets_root", "dataset/ranker_sets")))
    result_root = Path(str(roots.get("result_root", "result")))
    logs_root = Path(str(roots.get("logs_root", "logs")))

    kind = str(experiment.get("kind", "global_ranker"))
    ranker_data_experiment = name
    if kind == "per_user_adaptation":
        ranker_data_experiment = str(
            experiment.get("ranker_data_experiment", experiment.get("parent_run", ""))
        )
        if not EXPERIMENT_NAME.fullmatch(ranker_data_experiment):
            raise ValueError(
                "per-user experiment requires a valid parent_run or ranker_data_experiment"
            )

    ranker_set_dir = ranker_root / ranker_data_experiment
    result_dir = result_root / name
    log_dir = logs_root / name
    experiment["ranker_set_dir"] = str(ranker_set_dir)
    experiment["result_dir"] = str(result_dir)
    experiment["log_dir"] = str(log_dir)

    ranker = config.setdefault("ranker", {})
    ranker["data_dir"] = str(ranker_set_dir)
    ranker["output_dir"] = str(result_dir)

    if kind == "per_user_adaptation":
        parent_run = str(experiment["parent_run"])
        parent_result = result_root / parent_run
        adaptation = config.setdefault("user_adaptation", {})
        adaptation["global_model_dir"] = str(parent_result)
        adaptation["validation_candidates"] = str(
            ranker_set_dir / "validation_candidates.jsonl"
        )
        adaptation["validation_data_dir"] = str(ranker_set_dir)
        adaptation["output_dir"] = str(result_dir)

    for path in (ranker_set_dir, result_dir, log_dir):
        resolve_path(path).mkdir(parents=True, exist_ok=True)
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping")
    base_value = config.pop("base", None)
    if base_value:
        base_path = Path(base_value)
        if not base_path.is_absolute():
            project_candidate = project_root() / base_path
            local_candidate = config_path.parent / base_path
            base_path = project_candidate if project_candidate.exists() else local_candidate
        config = _deep_merge(load_config(base_path), config)
    return _experiment_paths(config)


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(destination)


def limited(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return rows
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    chosen = set(indices[:limit])
    return [row for index, row in enumerate(rows) if index in chosen]
