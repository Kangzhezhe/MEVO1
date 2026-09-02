"""阶段 09：训练共享 DeBERTa Scorer，并输出开发/测试预测。"""

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
)


def ranker_config(config: dict) -> dict:
    """把本实验 YAML 映射成主工程 Ranker 的最小兼容配置。"""

    settings = {
        **config["scorer"],
        "shuffle_factor_directions": False,
        "factor_dropout_probability": 0.0,
    }
    if str(settings["input_mode"]) != "task_only":
        raise ValueError("Scorer 只能看 q+c，input_mode 必须是 task_only")
    return {
        "project": {"seed": int(config["project"]["seed"])},
        "metric": dict(config["metric"]),
        "ranker": settings,
    }


def train_and_predict(config: dict) -> dict:
    module = load_project_stage("code/08_train_global_ranker.py", "factor_free_ranker")
    compatibility = ranker_config(config)
    data_dir = resolve_path(config["paths"]["scorer_data_dir"])
    output_dir = resolve_path(config["paths"]["scorer_output_dir"])
    report = module.train(compatibility, data_dir, output_dir)
    for split in ("validation", "test"):
        if (data_dir / f"{split}_candidates.jsonl").exists():
            module.predict(
                compatibility,
                data_dir,
                output_dir,
                split,
                output_dir / f"{split}_predictions.jsonl",
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="09 - 训练共享 q+c Scorer")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    args = parser.parse_args()
    train_and_predict(load_config(args.config))


if __name__ == "__main__":
    main()
