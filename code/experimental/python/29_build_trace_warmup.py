"""阶段 29：构造平衡的共享 Trace Warm-up 数据。

原始条件偏好 SFT 中，大多数样本因历史证据不足而使用 output-only，完整
Trace 仅占约 6%。本阶段不重新调用 Teacher，而是复用已经通过盲审的
personalized_trace，并按 split 抽取等量 output-only 样本，继续训练现有共享
LoRA。这样既强化 Trace Schema/推理路径，又保留无偏好时的诚实回退能力。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline_common import (  # noqa: E402
    load_config,
    read_jsonl,
    resolve_path,
    write_json,
    write_jsonl,
)


def _order(row: dict[str, Any], seed: int) -> str:
    value = f"{seed}:{row['example_id']}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def build(config: dict[str, Any]) -> dict[str, Any]:
    settings = config["trace_warmup"]
    source_dir = resolve_path(settings["source_sft_dir"])
    source = read_jsonl(source_dir / "all_sft.jsonl")
    if not source:
        raise ValueError(f"Warm-up源SFT为空: {source_dir}")

    seed = int(settings.get("seed", config["training"]["seed"]))
    selected: list[dict[str, Any]] = []
    split_report: dict[str, dict[str, int]] = {}
    for split in ("train", "validation"):
        rows = [row for row in source if str(row.get("split")) == split]
        traces = [
            row for row in rows
            if str(row.get("supervision_tier")) == "personalized_trace"
            and str(row.get("trace_text", ""))
        ]
        output_only = [row for row in rows if not str(row.get("trace_text", ""))]
        if not traces:
            raise ValueError(f"split={split} 没有经过验证的个性化Trace")
        ratio = float(settings.get("output_only_ratio", 1.0))
        output_count = min(len(output_only), round(len(traces) * ratio))
        sampled_output = sorted(output_only, key=lambda row: _order(row, seed))[:output_count]
        values = traces + sampled_output
        for row in values:
            item = dict(row)
            # Warm-up按监督样本平衡；不再继承原始每Query 1/4的总权重。
            item["sample_weight"] = 1.0
            item["warmup_supervision"] = (
                "personalized_trace" if str(item.get("trace_text", "")) else "output_only"
            )
            selected.append(item)
        split_report[split] = {
            "personalized_trace": len(traces),
            "output_only": output_count,
            "examples": len(values),
        }

    output_dir = resolve_path(config["paths"]["sft_dir"])
    train = [row for row in selected if row["split"] == "train"]
    validation = [row for row in selected if row["split"] == "validation"]
    write_jsonl(output_dir / "all_sft.jsonl", selected)
    write_jsonl(output_dir / "train_sft.jsonl", train)
    write_jsonl(output_dir / "validation_sft.jsonl", validation)
    report = {
        "protocol": "balanced_conditional_trace_warmup_v1",
        "source_sft_dir": str(source_dir),
        "output_sft_dir": str(output_dir),
        "output_only_ratio": float(settings.get("output_only_ratio", 1.0)),
        "sample_weight": 1.0,
        "splits": split_report,
        "examples": len(selected),
    }
    write_json(output_dir / "manifest.json", report)
    print(f"balanced Trace warm-up data -> {output_dir}; report={report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="29 - 构造平衡式Trace Warm-up数据")
    parser.add_argument(
        "--config", default=str(HERE / "config_conditional_trace_warmup.yaml")
    )
    args = parser.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
