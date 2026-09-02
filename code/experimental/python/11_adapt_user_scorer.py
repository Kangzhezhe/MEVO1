"""阶段 11：冻结共享 DeBERTa，为每个评估用户拟合独立 Head/Adapter。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from common import user_head  # noqa: E402
from pipeline_common import load_config, resolve_path, stage_path, write_jsonl  # noqa: E402


def compatibility_config(config: dict, split: str) -> dict:
    data_dir = resolve_path(config["paths"]["scorer_data_dir"])
    output_dir = resolve_path(config["paths"]["user_scorer_output_dir"]) / split
    adaptation_split = f"adaptation_{split}"
    settings = {
        **config["user_adaptation"],
        "output_dir": str(output_dir),
        "global_model_dir": str(resolve_path(config["paths"]["scorer_output_dir"])),
        "adaptation_source": str(stage_path(config, adaptation_split, "editor")),
        "validation_candidates": str(data_dir / f"{split}_candidates.jsonl"),
        # ``all`` is used by the IDPO LOO protocol: each user's complete
        # history contributes one adaptation query, and users may have
        # different history sizes.
        "profiles_per_user": config["profile_augmentation"]["profiles_per_user"],
        # Ranker candidate group 保留真实 user_id；一个 Head 服务该用户全部 Query。
        "validation_user_field": "user_id",
    }
    ranker = {
        **config["scorer"],
        "shuffle_factor_directions": False,
        "factor_dropout_probability": 0.0,
    }
    if str(ranker["input_mode"]) != "task_only":
        raise ValueError("Per-user Scorer 只能读取 q+c")
    return {
        "project": {"seed": int(config["project"]["seed"])},
        "metric": dict(config["metric"]),
        "ranker": ranker,
        "user_adaptation": settings,
    }


def adapt_and_predict(config: dict, split: str) -> dict:
    compatibility = compatibility_config(config, split)
    report = user_head.adapt(compatibility)
    predictions = user_head.predict(compatibility)
    # 旧模块固定写 validation_predictions；另存带 split 名的副本便于最终评估。
    output_dir = resolve_path(compatibility["user_adaptation"]["output_dir"])
    destination = output_dir / f"{split}_predictions.jsonl"
    write_jsonl(destination, predictions)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="11 - 训练并应用 per-user Head/Adapter")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    adapt_and_predict(load_config(args.config), args.split)


if __name__ == "__main__":
    main()
