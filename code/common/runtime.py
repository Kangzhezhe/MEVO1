"""运行时约定：默认配置、阶段文件名和数字文件的动态加载。"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

from common.teacher import TeacherClient
from common.utils import project_root, resolve_path


GLOBAL_CONFIG = str(project_root() / "config/experiment/baseline/global500_last2.yaml")
USER_CONFIG = str(project_root() / "config/experiment/baseline/user_head_m5.yaml")

STAGE_FILES = {
    "prepare": "01_prepared.jsonl",
    "retrieve": "02_retrieved.jsonl",
    "factors": "03_factors.jsonl",
    "seeds": "04_seeds.jsonl",
    "mutate": "05_mutations.jsonl",
    "score": "06_scored.jsonl",
}


def config_parser(description: str, default: str = GLOBAL_CONFIG) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default)
    return parser


def candidate_root(config: dict) -> Path:
    data = config["data"]
    return resolve_path(data["processed_root"]) / str(
        data.get("processed_split", data["split"])
    )


def stage_path(config: dict, stage: str) -> Path:
    return candidate_root(config) / STAGE_FILES[stage]


def teacher_client(config: dict) -> TeacherClient:
    return TeacherClient(config["teacher"], resolve_path(config["teacher"]["cache_dir"]))


def load_stage(filename: str) -> ModuleType:
    """Load a numbered stage without making numeric filenames import syntax."""
    path = project_root() / "code" / filename
    module_name = "mevo_stage_" + path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load stage: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
