"""辅助预测入口：按实验类型调用 Global 或 per-user 推理。"""

from __future__ import annotations

from common.runtime import GLOBAL_CONFIG, config_parser, load_stage
from common.utils import load_config, resolve_path


def main() -> None:
    parser = config_parser("11 - Predict with the global Ranker or per-user heads", GLOBAL_CONFIG)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    config = load_config(args.config)
    kind = str(config.get("experiment", {}).get("kind", "global_ranker"))
    if kind == "per_user_adaptation":
        from common.user_head import predict as predict_user

        predict_user(config)
        return

    ranker = config["ranker"]
    output_dir = resolve_path(ranker["output_dir"])
    load_stage("08_train_global_ranker.py").predict(
        config,
        resolve_path(ranker["data_dir"]),
        output_dir,
        args.split,
        output_dir / f"{args.split}_predictions.jsonl",
    )


if __name__ == "__main__":
    main()
