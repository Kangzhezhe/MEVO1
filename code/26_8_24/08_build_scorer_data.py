"""阶段 08：仅从本地 LoRA Editor 候选构建共享 Scorer 数据。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline_common import (  # noqa: E402
    load_config,
    load_project_stage,
    resolve_path,
    stage_path,
)


def build(config: dict) -> dict:
    if str(config["scorer"]["input_mode"]) != "task_only":
        raise ValueError("无 Factor 路线要求 scorer.input_mode=task_only")
    module = load_project_stage("code/26_8_24/07_build_ranker_data.py", "factor_free_ranker_data")
    test_source = stage_path(config, "test", "editor")
    return module.build_from_splits(
        train_source=stage_path(config, "train", "editor"),
        validation_source=stage_path(config, "validation", "editor"),
        test_source=test_source if test_source.exists() else None,
        output_dir=resolve_path(config["paths"]["scorer_data_dir"]),
        seed=int(config["project"]["seed"]),
        pair_strategy=str(config["scorer"]["pair_strategy"]),
        metric=str(config["metric"]["primary"]),
        minimum_margin=float(config["scorer"]["pair_minimum_margin"]),
        max_pairs_per_sample=int(config["scorer"]["max_pairs_per_sample"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="08 - 构建 q+c Scorer 数据")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    args = parser.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
