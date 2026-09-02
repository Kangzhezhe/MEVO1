"""Evaluate RPM-LaMP5 predictions with the same ROUGE backend as MeVO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.metrics import corpus_score_with_ci, score
from common.utils import load_config, read_jsonl, resolve_path


def evaluate(config: dict) -> dict:
    settings = config["rpm"]
    output_dir = resolve_path(settings.get("output_dir", config["experiment"]["result_dir"]))
    input_rows = {str(row["id"]): row for row in read_jsonl(resolve_path(settings["input_path"]))}
    prediction_rows = {str(row["sample_id"]): row for row in read_jsonl(output_dir / "22_predictions.jsonl")}
    if set(input_rows) != set(prediction_rows):
        raise ValueError(
            f"RPM prediction IDs do not match input: input={len(input_rows)}, predictions={len(prediction_rows)}"
        )
    predictions = [prediction_rows[sample_id]["prediction"] for sample_id in input_rows]
    references = [input_rows[sample_id]["target"] for sample_id in input_rows]
    intervals = corpus_score_with_ci(predictions, references)
    per_sample = []
    for sample_id in input_rows:
        metrics = score(prediction_rows[sample_id]["prediction"], input_rows[sample_id]["target"])
        per_sample.append(
            {
                "sample_id": sample_id,
                "prediction": prediction_rows[sample_id]["prediction"],
                "target": input_rows[sample_id]["target"],
                **metrics,
            }
        )
    report = {
        "method": "RPM faithful stage order with LaMP-5 task adapter",
        "sample_count": len(predictions),
        "rouge": intervals,
        "mean_rouge_1": sum(row["rouge_1"] for row in per_sample) / len(per_sample),
        "mean_rouge_l": sum(row["rouge_l"] for row in per_sample) / len(per_sample),
        "predictions": per_sample,
    }
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = [
        "# RPM LaMP-5 evaluation",
        "",
        f"- samples: {report['sample_count']}",
        f"- mean ROUGE-1 (per sample): {report['mean_rouge_1']:.6f}",
        f"- mean ROUGE-L (per sample): {report['mean_rouge_l']:.6f}",
        f"- official-compatible ROUGE-1 bootstrap midpoint: {intervals['rouge_1']['mid']:.6f}",
        f"- official-compatible ROUGE-L bootstrap midpoint: {intervals['rouge_l']['mid']:.6f}",
    ]
    (output_dir / "evaluation_report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RPM-LaMP5 predictions")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = evaluate(load_config(args.config))
    print(json.dumps({k: v for k, v in report.items() if k != "predictions"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
