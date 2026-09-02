"""汇总 S0/S1/S2 的监督质量、Editor 和共享 Ranker 指标。"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any

from pipeline_common import load_config, normalized_text, read_jsonl, resolve_path, stage_path, write_json
from common.metrics import score

TOKENS = re.compile(r"[a-z0-9]+", re.IGNORECASE)
FORBIDDEN = re.compile(
    r"\b(?:gold(?:en)?(?: answer| title| output)?|ground[ -]?truth|"
    r"reference(?:'s)?|desired[_ -]?"
    r"(?:output|title|answer)|target (?:answer|title|output)|"
    r"rouge(?:-[a-z])?|evaluation metric)\b",
    re.IGNORECASE,
)


def _tokens(value: str) -> set[str]:
    return set(TOKENS.findall(str(value).casefold()))


def _coverage(parent: str, gold: str, text: str) -> float:
    changed = _tokens(parent) ^ _tokens(gold)
    return 1.0 if not changed else len(changed & _tokens(text)) / len(changed)


def _quality(config: dict[str, Any], mode: str) -> dict[str, Any]:
    stage = {
        "output_only": "seeds",
        "gold_aware_trace": "gold_traces",
        "atomic_trace": "atomic_traces",
    }.get(mode)
    if stage is None:
        raise ValueError(mode)
    rows = read_jsonl(stage_path(config, "train", stage))
    result: dict[str, Any] = {"queries": len(rows), "examples": len(rows) * 4}
    if mode == "output_only":
        result.update({"traces": 0, "exact_gold_outputs": len(rows) * 4})
        return result
    if mode == "gold_aware_trace":
        traces = []
        for row in rows:
            traces.extend(
                (row, {x["candidate_id"]: x["text"] for x in row["candidates"]}, trace)
                for trace in row.get("gold_aware_traces", [])
            )
        text_values = [
            " ".join(
                [trace.get("task_correction", ""), trace.get("profile_signal", {}).get("observation", ""), trace.get("edit_action", "")]
            )
            for _, _, trace in traces
        ]
        coverage = [
            _coverage(parent[str(trace["parent_id"])], row["target"], text)
            for (row, parent, trace), text in zip(traces, text_values)
        ]
        result.update({
            "traces": len(traces),
            "exact_gold_outputs": len(traces),
            "forbidden_or_reference_leakage": sum(bool(FORBIDDEN.search(text)) for text in text_values),
            "full_gold_copied_in_edit_action": sum(
                normalized_text(row["target"]) in normalized_text(trace.get("edit_action", ""))
                for row, _, trace in traces
            ),
            "nonempty_profile_evidence": sum(bool(trace.get("profile_signal", {}).get("evidence_ids")) for _, _, trace in traces),
            "mean_changed_token_coverage": statistics.mean(coverage) if coverage else 0.0,
        })
        return result
    if mode != "atomic_trace":
        raise ValueError(mode)
    traces = []
    rejected = 0
    operations = []
    personal = []
    coverage = []
    for row in rows:
        rejected += int(
            bool((row.get("atomic_metadata") or {}).get("quality_rejected", False))
        )
        parents = {x["candidate_id"]: x["text"] for x in row["candidates"]}
        for trace in row.get("atomic_traces", []):
            traces.append((row, parents, trace))
            ops = trace.get("task_operations", []) + trace.get("personalized_operations", [])
            operations.extend(ops)
            personal.extend(trace.get("personalized_operations", []))
            text = " ".join(str(v) for op in ops for v in op.values() if not isinstance(v, list))
            coverage.append(_coverage(parents[trace["parent_id"]], row["target"], text))
    result.update({
        "accepted_queries": len(rows) - rejected,
        "quality_rejected_queries": rejected,
        "traces": len(traces),
        # 质量拒绝的 Query 在 Stage 05 中回退为 output-only，仍有 4 个精确 Gold 输出。
        "exact_gold_outputs": len(rows) * 4,
        "atomic_operations": len(operations),
        "mean_operations_per_trace": len(operations) / max(len(traces), 1),
        "personalized_operations": len(personal),
        "traces_with_personalized_operation": sum(bool(x.get("personalized_operations")) for _, _, x in traces),
        "forbidden_or_reference_leakage": sum(bool(FORBIDDEN.search(json.dumps(x, ensure_ascii=False))) for x in operations),
        "mean_changed_token_coverage": statistics.mean(coverage) if coverage else 0.0,
    })
    return result


def _load_if_exists(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _per_example_rouge_l(config: dict[str, Any], split: str) -> dict[str, float]:
    """读取同一 Query 的最终预测，供配对置信区间使用。"""

    data_dir = resolve_path(config["paths"]["scorer_data_dir"])
    prediction_path = (
        resolve_path(config["paths"]["scorer_output_dir"])
        / f"{split}_predictions.jsonl"
    )
    label_path = data_dir / f"{split}_labels.jsonl"
    if not prediction_path.exists() or not label_path.exists():
        return {}
    labels = {str(row["sample_id"]): str(row["target"]) for row in read_jsonl(label_path)}
    predictions = {
        str(row["sample_id"]): str(row["prediction"])
        for row in read_jsonl(prediction_path)
    }
    return {
        sample_id: score(predictions[sample_id], target)["rouge_l"]
        for sample_id, target in labels.items()
        if sample_id in predictions
    }


def _paired_bootstrap(
    left: dict[str, float],
    right: dict[str, float],
    *,
    seed: int = 20260801,
    samples: int = 20_000,
) -> dict[str, Any]:
    ids = sorted(set(left) & set(right))
    if not ids:
        return {}
    deltas = [left[sample_id] - right[sample_id] for sample_id in ids]
    rng = random.Random(seed)
    count = len(deltas)
    draws = sorted(
        sum(deltas[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    return {
        "queries": count,
        "mean_per_example_delta": statistics.mean(deltas),
        "ci95": {
            "low": draws[int(0.025 * samples)],
            "high": draws[int(0.975 * samples)],
        },
        "wins": sum(value > 1.0e-12 for value in deltas),
        "ties": sum(abs(value) <= 1.0e-12 for value in deltas),
        "losses": sum(value < -1.0e-12 for value in deltas),
    }


def compare(config_paths: dict[str, str], output: Path) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    configs: dict[str, dict[str, Any]] = {}
    for name, path in config_paths.items():
        config = load_config(path)
        configs[name] = config
        mode = str(config["sft_data"]["supervision_mode"])
        methods[name] = {
            "config": str(Path(path).resolve()),
            "mode": mode,
            "supervision_quality": _quality(config, mode),
            "sft": _load_if_exists(resolve_path(config["paths"]["sft_dir"]) / "manifest.json"),
            "editor": _load_if_exists(resolve_path(config["paths"]["editor_output_dir"]) / "training_report.json"),
            "ranker": _load_if_exists(resolve_path(config["paths"]["scorer_output_dir"]) / "training_report.json"),
            "validation": _load_if_exists(resolve_path(config["paths"]["reports_dir"]) / "validation_report.json"),
            "test": _load_if_exists(resolve_path(config["paths"]["reports_dir"]) / "test_report.json"),
        }
    paired: dict[str, Any] = {}
    for split in ("validation", "test"):
        values = {
            name: _per_example_rouge_l(config, split)
            for name, config in configs.items()
        }
        paired[split] = {
            f"{left}-{right}": _paired_bootstrap(values[left], values[right])
            for left, right in (("S1", "S0"), ("S2", "S0"), ("S1", "S2"))
        }
    report = {
        "protocol": "s0_s1_s2_same_train300_dev100_test100",
        "methods": methods,
        "paired_per_example_rouge_l": paired,
    }
    write_json(output, report)
    lines = [
        "# S0 / S1 / S2 监督与 Ranker 对比",
        "",
        "三组使用相同的 Per-Pcs Train300、Dev100、Test100 和 target-blind 四 Parent；差异只在 Editor SFT 监督形式。",
        "",
        "## 监督质量",
        "",
        "| 方法 | Trace | 拒绝 Query | 元答案泄漏 | 完整 Gold 复制 | 原子操作 | 个性化操作 | 覆盖率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in methods.items():
        q = item["supervision_quality"]
        lines.append(
            f"| {name} | {q.get('traces', 0)} | {q.get('quality_rejected_queries', 0)} | "
            f"{q.get('forbidden_or_reference_leakage', 0)} | {q.get('full_gold_copied_in_edit_action', 0)} | "
            f"{q.get('atomic_operations', 0)} | {q.get('personalized_operations', 0)} | "
            f"{q.get('mean_changed_token_coverage', 0.0):.3f} |"
        )
    lines += [
        "",
        "## 配对 ROUGE-L 差异",
        "",
        "同一 Query 上计算每例 ROUGE-L 差异并做 20,000 次配对 bootstrap；CI 跨 0 表示当前样本不足以确认稳定优势。",
        "",
        "| Split | 对比 | 平均差 | 95% CI | 胜/平/负 |",
        "|---|---|---:|---:|---:|",
    ]
    for split, comparisons in paired.items():
        for name, values in comparisons.items():
            interval = values.get("ci95", {})
            lines.append(
                f"| {split} | {name} | {values.get('mean_per_example_delta', 0.0):+.4f} | "
                f"[{interval.get('low', 0.0):+.4f}, {interval.get('high', 0.0):+.4f}] | "
                f"{values.get('wins', 0)}/{values.get('ties', 0)}/{values.get('losses', 0)} |"
            )
    lines += [
        "",
        "## 共享 Ranker",
        "",
        "| 方法 | Dev ROUGE-L | Dev Hit@1 | Dev Hit@5 | Dev Mean Regret | Dev Pair Acc | Test ROUGE-L | Test Hit@1 | Test Hit@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in methods.items():
        values = []
        for split in ("validation", "test"):
            report = item.get(split) or {}
            shared = (report.get("scorers") or {}).get("shared") or {}
            ranking = shared.get("ranking") or {}
            values.extend([
                float(shared.get("rouge_l", 0.0)),
                float(ranking.get("hit_at", {}).get("1", 0.0)),
                float(ranking.get("hit_at", {}).get("5", 0.0)),
                float(ranking.get("mean_regret", 0.0)),
                float(ranking.get("pairwise_accuracy", 0.0)),
            ])
        lines.append(
            f"| {name} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | {values[3]:.4f} | {values[4]:.4f} | "
            f"{values[5]:.4f} | {values[6]:.4f} | {values[7]:.4f} |"
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="result/s0_s1_s2_comparison/report.json")
    args = parser.parse_args()
    compare(
        {"S0": "code/config_s0.yaml", "S1": "code/config_s1.yaml", "S2": "code/config_s2.yaml"},
        resolve_path(args.output),
    )


if __name__ == "__main__":
    main()
