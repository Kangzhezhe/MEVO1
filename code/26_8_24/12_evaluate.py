"""阶段 12：统一报告候选质量、共享 Scorer 与 per-user Scorer 效果。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from common.metrics import corpus_bleu, corpus_score_with_ci, score  # noqa: E402
from pipeline_common import load_config, normalized_text, read_jsonl, resolve_path, write_json  # noqa: E402


EPSILON = 1.0e-12


def _text_metrics(predictions: list[str], references: list[str]) -> dict[str, Any]:
    intervals = corpus_score_with_ci(predictions, references)
    return {
        "rouge_1": intervals["rouge_1"]["mid"],
        "rouge_l": intervals["rouge_l"]["mid"],
        "bleu": corpus_bleu(predictions, references)["score"],
        "confidence_intervals": intervals,
    }


def _prediction_metrics(
    predictions: list[dict[str, Any]],
    groups: dict[str, dict[str, Any]],
    labels: dict[str, str],
    margin: float,
) -> dict[str, Any]:
    texts, references, regrets, reciprocal_ranks = [], [], [], []
    hits = {1: [], 3: [], 5: [], 10: []}
    pair_correct = pair_total = 0
    for prediction in predictions:
        sample_id = str(prediction["sample_id"])
        group = groups[sample_id]
        target = labels[sample_id]
        candidate_by_id = {str(item["candidate_id"]): item for item in group["candidates"]}
        ranked = prediction["ranked_candidates"]
        ranked_ids = [str(item["candidate_id"]) for item in ranked]
        if set(ranked_ids) != set(candidate_by_id):
            raise ValueError(f"sample={sample_id} 的排序不是候选全集的排列")
        gold = {
            candidate_id: float(score(item["text"], target)["rouge_l"])
            for candidate_id, item in candidate_by_id.items()
        }
        best = max(gold.values())
        oracle_ids = {key for key, value in gold.items() if abs(value - best) <= EPSILON}
        selected_id = str(prediction["selected_id"])
        texts.append(str(candidate_by_id[selected_id]["text"]))
        references.append(target)
        regrets.append(best - gold[selected_id])
        rank = next(index for index, value in enumerate(ranked_ids, 1) if value in oracle_ids)
        reciprocal_ranks.append(1.0 / rank)
        for k in hits:
            hits[k].append(float(any(value in oracle_ids for value in ranked_ids[:k])))
        for left in range(len(ranked)):
            for right in range(left + 1, len(ranked)):
                gold_delta = gold[ranked_ids[left]] - gold[ranked_ids[right]]
                if abs(gold_delta) < margin:
                    continue
                predicted_delta = float(ranked[left]["ranker_score"]) - float(
                    ranked[right]["ranker_score"]
                )
                pair_correct += int(predicted_delta * gold_delta > 0)
                pair_total += 1
    metric = _text_metrics(texts, references)
    metric["ranking"] = {
        "hit_at": {str(k): float(np.mean(values)) for k, values in hits.items()},
        "mrr_of_oracle": float(np.mean(reciprocal_ranks)),
        "mean_regret": float(np.mean(regrets)),
        "median_regret": float(np.median(regrets)),
        "pairwise_accuracy": pair_correct / pair_total if pair_total else 0.0,
        "evaluated_pairs": pair_total,
    }
    return metric


def _pool_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = {
        "task_seed_0": [],
        "best_of_four_seeds": [],
        "ten_candidate_oracle": [],
    }
    references = []
    source_oracles: Counter[str] = Counter()
    unique_rates, mutation_deltas, crossover_deltas = [], [], []
    for row in rows:
        target = str(row["target"])
        seeds = row["candidates"]
        pool = seeds + row.get("mutations", [])
        scored = {str(item["candidate_id"]): score(item["text"], target) for item in pool}
        choose = lambda values: max(  # noqa: E731
            values,
            key=lambda item: (
                float(scored[str(item["candidate_id"])]["rouge_l"]),
                float(scored[str(item["candidate_id"])]["rouge_1"]),
                str(item["candidate_id"]),
            ),
        )
        task = next((item for item in seeds if item["type"] == "task_seed"), seeds[0])
        seed_oracle, pool_oracle = choose(seeds), choose(pool)
        methods["task_seed_0"].append(str(task["text"]))
        methods["best_of_four_seeds"].append(str(seed_oracle["text"]))
        methods["ten_candidate_oracle"].append(str(pool_oracle["text"]))
        references.append(target)
        source_oracles[str(pool_oracle["type"])] += 1
        unique_rates.append(len({normalized_text(item["text"]) for item in pool}) / len(pool))
        for child in row.get("mutations", []):
            child_score = float(scored[str(child["candidate_id"])]["rouge_l"])
            if child["type"] == "mutation":
                parent_ids = [str(child["parent_id"])]
            else:
                parent_ids = [str(child["parent_a_id"]), str(child["parent_b_id"])]
            parent_score = max(float(scored[parent_id]["rouge_l"]) for parent_id in parent_ids)
            delta = child_score - parent_score
            (mutation_deltas if child["type"] == "mutation" else crossover_deltas).append(delta)

    def delta(values: list[float]) -> dict[str, float | int]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": len(values),
            "mean": float(array.mean()) if len(array) else 0.0,
            "win_rate": float((array > 0).mean()) if len(array) else 0.0,
            "significant_win_rate_0.02": float((array >= 0.02).mean()) if len(array) else 0.0,
        }

    return {
        "sample_count": len(rows),
        "metrics": {name: _text_metrics(values, references) for name, values in methods.items()},
        "mean_exact_unique_rate": float(np.mean(unique_rates)),
        "oracle_source_counts": dict(source_oracles),
        "mutation_delta": delta(mutation_deltas),
        "crossover_delta": delta(crossover_deltas),
    }


def evaluate(config: dict, split: str) -> dict[str, Any]:
    data_dir = resolve_path(config["paths"]["scorer_data_dir"])
    labels = {
        str(row["sample_id"]): str(row["target"])
        for row in read_jsonl(data_dir / f"{split}_labels.jsonl")
    }
    groups = {
        str(row["sample_id"]): row
        for row in read_jsonl(data_dir / f"{split}_candidates.jsonl")
    }
    # 跟随公共 stage 映射，避免实验协议改名后评估读取到旧候选池。
    from pipeline_common import stage_path

    source_rows = read_jsonl(stage_path(config, split, "editor"))
    shared_path = resolve_path(config["paths"]["scorer_output_dir"]) / f"{split}_predictions.jsonl"
    user_path = (
        resolve_path(config["paths"]["user_scorer_output_dir"])
        / split
        / f"{split}_predictions.jsonl"
    )
    report: dict[str, Any] = {
        "protocol": {
            "dataset": "Per-Pcs scholarly_title",
            "split": split,
            "candidate_budget": int(config["evolution"]["candidate_budget"]),
            "editor_input": "current_input + parent(s) + retrieved_history",
            "scorer_input": "query + candidate",
            "explicit_user_factors": False,
            "gold_visible_during_generation": False,
        },
        "candidate_pool": _pool_metrics(source_rows),
        "scorers": {},
    }
    margin = float(config["scorer"]["pair_minimum_margin"])
    if shared_path.exists():
        report["scorers"]["shared"] = _prediction_metrics(
            read_jsonl(shared_path), groups, labels, margin
        )
    if user_path.exists():
        report["scorers"]["per_user"] = _prediction_metrics(
            read_jsonl(user_path), groups, labels, margin
        )
    output_dir = resolve_path(config["paths"]["reports_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"{split}_report.json", report)

    lines = [
        f"# Per-Pcs 无 Factor 个性化候选优化：{split}",
        "",
        "模型输入中没有持久化用户 Factor。Editor 直接读取检索历史；本次对比只评估 shared Scorer，Scorer 输入为 q+c，未执行 per-user Head 适配。",
        "",
        "## 候选空间",
        "",
        "| 方法 | ROUGE-1 | ROUGE-L | SacreBLEU |",
        "|---|---:|---:|---:|",
    ]
    labels_by_method = {
        "task_seed_0": "Task-only Seed 1",
        "best_of_four_seeds": "4-Seed Oracle",
        "ten_candidate_oracle": "10-Candidate Oracle",
    }
    for name, values in report["candidate_pool"]["metrics"].items():
        lines.append(
            f"| {labels_by_method[name]} | {values['rouge_1']:.6f} | "
            f"{values['rouge_l']:.6f} | {values['bleu']:.4f} |"
        )
    lines.extend(["", "## Scorer", "", "| Scorer | ROUGE-1 | ROUGE-L | BLEU | Hit@1 | Hit@5 | Mean Regret | Pair Acc |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for name, values in report["scorers"].items():
        ranking = values["ranking"]
        lines.append(
            f"| {name} | {values['rouge_1']:.6f} | {values['rouge_l']:.6f} | "
            f"{values['bleu']:.4f} | {ranking['hit_at']['1']:.4f} | "
            f"{ranking['hit_at']['5']:.4f} | {ranking['mean_regret']:.6f} | "
            f"{ranking['pairwise_accuracy']:.4f} |"
        )
    pool = report["candidate_pool"]
    lines.extend(
        [
            "",
            "## 演化诊断",
            "",
            f"- 候选精确去重后比例：`{pool['mean_exact_unique_rate']:.4f}`",
            f"- Mutation 相对 Parent 的平均 ROUGE-L 变化：`{pool['mutation_delta']['mean']:+.6f}`",
            f"- Crossover 相对较强 Parent 的平均 ROUGE-L 变化：`{pool['crossover_delta']['mean']:+.6f}`",
            f"- Oracle 来源：`{json.dumps(pool['oracle_source_counts'], ensure_ascii=False)}`",
        ]
    )
    (output_dir / f"{split}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"evaluation -> {output_dir / f'{split}_report.md'}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="12 - 评估无 Factor Per-Pcs 流程")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    evaluate(load_config(args.config), args.split)


if __name__ == "__main__":
    main()
