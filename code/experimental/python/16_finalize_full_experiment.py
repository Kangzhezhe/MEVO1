"""汇总 S0/S1/S2 子集选优与最佳方法的全量 Per-Pcs 结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline_common import load_config, resolve_path, stage_path, write_json


HERE = Path(__file__).resolve().parent
FULL_CONFIGS = {
    "S0": HERE / "config_s0_full.yaml",
    "S1": HERE / "config_s1_full.yaml",
    "S2": HERE / "config_s2_full.yaml",
}


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 根节点必须是 object: {path}")
    return value


def _line_count(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def _shared(report: dict[str, Any]) -> dict[str, Any]:
    value = (report.get("scorers") or {}).get("shared")
    if not isinstance(value, dict):
        raise ValueError("全量报告缺少 shared Scorer")
    return value


def _method_row(name: str, item: dict[str, Any]) -> str:
    validation = _shared(item["validation"])
    test = _shared(item["test"])
    vr = validation["ranking"]
    tr = test["ranking"]
    return (
        f"| {name} | {validation['rouge_l']:.4f} | {vr['hit_at']['1']:.4f} | "
        f"{vr['hit_at']['5']:.4f} | {vr['mean_regret']:.4f} | "
        f"{test['rouge_l']:.4f} | {tr['hit_at']['1']:.4f} | "
        f"{tr['hit_at']['5']:.4f} | {tr['mean_regret']:.4f} |"
    )


def finalize(selection_path: Path, subset_path: Path, output: Path) -> dict[str, Any]:
    selection = _load(selection_path)
    subset = _load(subset_path)
    method = str(selection["best_method"])
    if method not in FULL_CONFIGS:
        raise ValueError(f"未知最佳方法: {method}")
    config_path = FULL_CONFIGS[method]
    config = load_config(config_path)
    reports_dir = resolve_path(config["paths"]["reports_dir"])
    validation = _load(reports_dir / "validation_report.json")
    test = _load(reports_dir / "test_report.json")
    sft = _load(resolve_path(config["paths"]["sft_dir"]) / "manifest.json")
    editor = _load(
        resolve_path(config["paths"]["editor_output_dir"]) / "training_report.json"
    )
    ranker = _load(
        resolve_path(config["paths"]["scorer_output_dir"]) / "training_report.json"
    )
    ranker_manifest = _load(
        resolve_path(config["paths"]["scorer_data_dir"]) / "manifest.json"
    )
    split_counts = {
        split: _line_count(stage_path(config, split, "prepare"))
        for split in ("train", "validation", "test")
    }
    full = {
        "method": method,
        "config": str(config_path),
        "split_counts": split_counts,
        "sft": sft,
        "editor": editor,
        "ranker_data": ranker_manifest,
        "ranker": ranker,
        "validation": validation,
        "test": test,
    }
    result = {
        "protocol": "s0_s1_s2_dev_selection_then_full_perpcs_test_once",
        "selection": selection,
        "subset_comparison": subset,
        "full": full,
    }
    write_json(output, result)

    lines = [
        "# S0 / S1 / S2 选优与全量 Per-Pcs 最终报告",
        "",
        f"按 Dev ROUGE-L、Hit@1、Mean Regret 的预设顺序选择 **{method}**；Test 不参与方法选择。",
        "",
        "## 子集统一对比",
        "",
        "| 方法 | Dev ROUGE-L | Dev Hit@1 | Dev Hit@5 | Dev Regret | Test ROUGE-L | Test Hit@1 | Test Hit@5 | Test Regret |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in subset["methods"].items():
        lines.append(_method_row(name, item))

    lines += [
        "",
        "## 全量数据与训练",
        "",
        f"- Train / Validation / Test Query：`{split_counts['train']} / {split_counts['validation']} / {split_counts['test']}`",
        f"- Editor SFT train / validation：`{sft['train_examples']} / {sft['validation_examples']}`",
        f"- Editor 最终 train / eval loss：`{editor['train_loss']:.6f} / {editor['eval_loss']:.6f}`",
        f"- Ranker 最佳 epoch：`{ranker['best_epoch']}`；checkpoint 选择：`{ranker['checkpoint_selection']}`",
        "",
        "## 全量候选空间",
        "",
        "| Split | Task Seed ROUGE-L | 4-Seed Oracle | 10-Candidate Oracle | Mutation Δ | Crossover Δ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, report in (("validation", validation), ("test", test)):
        pool = report["candidate_pool"]
        metrics = pool["metrics"]
        lines.append(
            f"| {split} | {metrics['task_seed_0']['rouge_l']:.4f} | "
            f"{metrics['best_of_four_seeds']['rouge_l']:.4f} | "
            f"{metrics['ten_candidate_oracle']['rouge_l']:.4f} | "
            f"{pool['mutation_delta']['mean']:+.4f} | "
            f"{pool['crossover_delta']['mean']:+.4f} |"
        )

    lines += [
        "",
        "## 全量 Shared Ranker",
        "",
        "| Split | ROUGE-1 | ROUGE-L | BLEU | Hit@1 | Hit@5 | Mean Regret | Pair Acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, report in (("validation", validation), ("test", test)):
        shared = _shared(report)
        ranking = shared["ranking"]
        lines.append(
            f"| {split} | {shared['rouge_1']:.4f} | {shared['rouge_l']:.4f} | "
            f"{shared['bleu']:.4f} | {ranking['hit_at']['1']:.4f} | "
            f"{ranking['hit_at']['5']:.4f} | {ranking['mean_regret']:.4f} | "
            f"{ranking['pairwise_accuracy']:.4f} |"
        )

    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"final full report -> {output.with_suffix('.md')}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        default="result/s0_s1_s2_comparison/best_method.json",
    )
    parser.add_argument(
        "--subset",
        default="result/s0_s1_s2_comparison/report.json",
    )
    parser.add_argument(
        "--output",
        default="result/s0_s1_s2_comparison/final_full_report.json",
    )
    args = parser.parse_args()
    finalize(resolve_path(args.selection), resolve_path(args.subset), resolve_path(args.output))


if __name__ == "__main__":
    main()
