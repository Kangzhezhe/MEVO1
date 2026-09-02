"""为一次正式运行生成带时间戳的独立结果配置。"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from pipeline_common import load_config  # noqa: E402


SAFE_NAME = re.compile(r"[^a-z0-9]+")


def safe_name(value: str) -> str:
    normalized = SAFE_NAME.sub("_", value.casefold()).strip("_")
    return normalized or "experiment"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a timestamped runtime config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()

    if not re.fullmatch(r"\d{8}_\d{6}", args.timestamp):
        raise ValueError("timestamp 必须为 YYYYMMDD_HHMMSS")

    config = load_config(args.config)
    config.pop("_config_path", None)
    experiment = safe_name(str(config.get("experiment", {}).get("name", "experiment")))
    run_id = f"{args.timestamp}_{experiment}"
    run_root = Path("result") / run_id

    paths = config.setdefault("paths", {})
    paths["editor_output_dir"] = str(run_root / "editor")
    paths["prediction_dir"] = str(run_root / "predictions")
    paths["reports_dir"] = str(run_root / "reports")
    config.setdefault("experiment", {})["run_id"] = run_id

    destination = Path(args.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(run_id)


if __name__ == "__main__":
    main()
