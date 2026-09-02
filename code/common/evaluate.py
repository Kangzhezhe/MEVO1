"""统一评估 Ranker、候选池上限、排序质量与 Mutation 有效性。"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from common.metrics import corpus_bleu, corpus_score_with_ci, score
from common.runtime import GLOBAL_CONFIG
from common.utils import load_config, read_json, read_jsonl, resolve_path


EPSILON = 1.0e-12
RANKING_K = (1, 3, 5, 10)


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "p90": 0.0}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "p90": float(np.quantile(array, 0.9)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Softmax temperature must be positive")
    shifted = values / temperature
    shifted -= shifted.max()
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


def _ndcg(relevances: np.ndarray, k: int) -> float:
    k = min(k, len(relevances))
    if not k:
        return 0.0
    discounts = np.log2(np.arange(2, k + 2, dtype=np.float64))
    gains = np.exp2(relevances[:k]) - 1.0
    ideal = np.exp2(np.sort(relevances)[::-1][:k]) - 1.0
    ideal_dcg = float(np.sum(ideal / discounts))
    return float(np.sum(gains / discounts) / ideal_dcg) if ideal_dcg > EPSILON else 0.0


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + end - 1) / 2.0 + 1.0
        cursor = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or left.std() <= EPSILON or right.std() <= EPSILON:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _kendall_tau_b(left: np.ndarray, right: np.ndarray) -> float | None:
    concordant = discordant = ties_left = ties_right = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            left_delta = left[first] - left[second]
            right_delta = right[first] - right[second]
            if abs(left_delta) <= EPSILON and abs(right_delta) <= EPSILON:
                continue
            if abs(left_delta) <= EPSILON:
                ties_left += 1
            elif abs(right_delta) <= EPSILON:
                ties_right += 1
            elif left_delta * right_delta > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_left)
        * (concordant + discordant + ties_right)
    )
    if denominator <= EPSILON:
        return None
    return float((concordant - discordant) / denominator)


def _mean_valid(values: Iterable[float | None]) -> tuple[float, int]:
    valid = [float(value) for value in values if value is not None and math.isfinite(value)]
    return (float(np.mean(valid)) if valid else 0.0, len(valid))


def _paired_comparison(
    ranker_scores: list[float],
    baseline_scores: list[float],
    seed: int,
    bootstrap_samples: int,
) -> dict:
    deltas = np.asarray(ranker_scores, dtype=np.float64) - np.asarray(
        baseline_scores, dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(bootstrap_samples, len(deltas)))
    bootstrap_means = deltas[indices].mean(axis=1)
    return {
        "sample_count": int(len(deltas)),
        "mean_delta_rouge_l": float(deltas.mean()),
        "median_delta_rouge_l": float(np.median(deltas)),
        "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        "wins": int((deltas > EPSILON).sum()),
        "ties": int((np.abs(deltas) <= EPSILON).sum()),
        "losses": int((deltas < -EPSILON).sum()),
    }


def _delta_report(values: list[float], significant_margin: float) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {
        **_summary(array),
        "wins": int((array > EPSILON).sum()),
        "ties": int((np.abs(array) <= EPSILON).sum()),
        "losses": int((array < -EPSILON).sum()),
        "win_rate": float((array > EPSILON).mean()),
        "significant_wins": int((array >= significant_margin).sum()),
        "significant_losses": int((array <= -significant_margin).sum()),
        "significant_win_rate": float((array >= significant_margin).mean()),
    }


def evaluate(
    predictions_path: Path,
    candidates_path: Path,
    labels_path: Path,
    manifest_path: Path,
    destination: Path,
    seed: int,
    label_temperature: float = 0.1,
    bootstrap_samples: int = 5000,
    split: str = "validation",
) -> dict:
    predictions = {row["sample_id"]: row for row in read_jsonl(predictions_path)}
    groups = {row["sample_id"]: row for row in read_jsonl(candidates_path)}
    labels = {row["sample_id"]: row["target"] for row in read_jsonl(labels_path)}
    manifest = read_json(manifest_path)
    split_ids_key = f"{split}_ids"
    if split_ids_key not in manifest:
        raise ValueError(f"Ranker manifest has no {split_ids_key}")
    expected_id_list = manifest[split_ids_key]
    expected_ids = set(expected_id_list)
    if not (set(predictions) == set(groups) == set(labels) == expected_ids):
        raise ValueError(
            "Ranker evaluation IDs do not align: "
            f"predictions={sorted(predictions)}, groups={sorted(groups)}, "
            f"labels={sorted(labels)}, expected={sorted(expected_ids)}"
        )
    if set(manifest["train_ids"]) & expected_ids:
        raise ValueError(f"Ranker manifest leaks {split} sample IDs into training")

    methods: dict[str, list[str]] = {
        "ranker_top1": [],
        "task_seed_0": [],
        "factor_seed_0": [],
        "best_task_seed": [],
        "best_factor_seed": [],
        "pool_oracle": [],
    }
    method_fallback_counts: Counter[str] = Counter()
    per_query_rouge_l: dict[str, list[float]] = {name: [] for name in methods}
    references: list[str] = []
    details = []
    selected_types: Counter[str] = Counter()
    oracle_type_groups: Counter[str] = Counter()
    candidate_type_scores: dict[str, list[float]] = defaultdict(list)
    candidate_type_counts: Counter[str] = Counter()
    candidate_scores_all: list[float] = []
    candidate_ranges: list[float] = []
    unique_rates: list[float] = []
    pairwise_candidate_similarity: list[float] = []
    regrets: list[float] = []
    normalized_regrets: list[float] = []
    reciprocal_ranks: list[float] = []
    hit_values: dict[int, list[float]] = {k: [] for k in RANKING_K}
    ndcg_values: dict[int, list[float]] = {k: [] for k in RANKING_K if k > 1}
    score_pearsons: list[float | None] = []
    spearmans: list[float | None] = []
    kendalls: list[float | None] = []
    listwise_kls: list[float] = []
    pair_correct = 0
    pair_total = 0
    per_group_pair_accuracy: list[float] = []
    mutation_deltas: list[float] = []
    mutation_by_factor: dict[str, list[float]] = defaultdict(list)
    mutation_by_parent_type: dict[str, list[float]] = defaultdict(list)
    missing_mutation_parents = 0
    minimum_margin = float(manifest.get("minimum_margin", 0.0))
    ranking_metric = str(manifest.get("metric", "rouge_l"))

    for sample_id in expected_id_list:
        group = groups[sample_id]
        target = labels[sample_id]
        candidates = group["candidates"]
        candidate_by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
        ranked = predictions[sample_id].get("ranked_candidates", [])
        ranked_ids = [str(candidate["candidate_id"]) for candidate in ranked]
        if len(ranked_ids) != len(set(ranked_ids)) or set(ranked_ids) != set(candidate_by_id):
            raise ValueError(f"Prediction ranking for sample={sample_id} is not a candidate permutation")

        gold = {
            candidate_id: score(candidate["text"], target)
            for candidate_id, candidate in candidate_by_id.items()
        }
        ranker_scores = np.asarray(
            [float(candidate["ranker_score"]) for candidate in ranked], dtype=np.float64
        )
        ranked_relevance = np.asarray(
            [float(gold[candidate_id][ranking_metric]) for candidate_id in ranked_ids],
            dtype=np.float64,
        )
        oracle_score = float(ranked_relevance.max())
        oracle_ids = {
            candidate_id
            for candidate_id, candidate_gold in gold.items()
            if abs(float(candidate_gold[ranking_metric]) - oracle_score) <= EPSILON
        }
        selected_id = str(predictions[sample_id]["selected_id"])
        if selected_id != ranked_ids[0]:
            raise ValueError(f"Prediction top-1 mismatch for sample={sample_id}")
        selected = candidate_by_id[selected_id]
        selected_score = float(gold[selected_id][ranking_metric])
        selected_types[str(selected["type"])] += 1

        task_candidates = [candidate for candidate in candidates if candidate["type"] == "task_seed"]
        factor_candidates = [candidate for candidate in candidates if candidate["type"] == "factor_seed"]
        if not task_candidates:
            raise ValueError(f"Candidate pool for sample={sample_id} has no task seed")
        task = next(
            (candidate for candidate in task_candidates if candidate["candidate_id"].endswith("_task_0")),
            task_candidates[0],
        )
        best_task = max(task_candidates, key=lambda candidate: gold[str(candidate["candidate_id"])][ranking_metric])
        if factor_candidates:
            factor = next(
                (
                    candidate
                    for candidate in factor_candidates
                    if candidate["candidate_id"].endswith("_factor_0")
                ),
                factor_candidates[0],
            )
            best_factor = max(
                factor_candidates,
                key=lambda candidate: gold[str(candidate["candidate_id"])][ranking_metric],
            )
        else:
            # Missing-abstract samples intentionally have no factor seeds.
            # Keep aggregate baseline arrays aligned while making this
            # diagnostic fallback explicit in the report.
            factor = task
            best_factor = best_task
            method_fallback_counts["factor_seed_0_to_task_seed_0"] += 1
            method_fallback_counts["best_factor_seed_to_best_task_seed"] += 1
        oracle = candidate_by_id[sorted(oracle_ids)[0]]
        row_candidates = {
            "ranker_top1": selected,
            "task_seed_0": task,
            "factor_seed_0": factor,
            "best_task_seed": best_task,
            "best_factor_seed": best_factor,
            "pool_oracle": oracle,
        }
        references.append(target)
        for method, candidate in row_candidates.items():
            methods[method].append(str(candidate["text"]))
            per_query_rouge_l[method].append(float(gold[str(candidate["candidate_id"])]["rouge_l"]))

        first_oracle_rank = next(
            index for index, candidate_id in enumerate(ranked_ids, 1) if candidate_id in oracle_ids
        )
        reciprocal_ranks.append(1.0 / first_oracle_rank)
        for k in RANKING_K:
            hit_values[k].append(float(any(candidate_id in oracle_ids for candidate_id in ranked_ids[:k])))
        for k in ndcg_values:
            ndcg_values[k].append(_ndcg(ranked_relevance, k))
        regret = oracle_score - selected_score
        score_range = oracle_score - float(ranked_relevance.min())
        regrets.append(regret)
        normalized_regrets.append(regret / score_range if score_range > EPSILON else 0.0)
        candidate_ranges.append(score_range)
        candidate_scores_all.extend(ranked_relevance.tolist())
        unique_rates.append(
            len({str(candidate["text"]).strip().casefold() for candidate in candidates})
            / len(candidates)
        )
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                pairwise_candidate_similarity.append(
                    score(str(candidates[left]["text"]), str(candidates[right]["text"]))["rouge_l"]
                )

        oracle_types = {str(candidate_by_id[candidate_id]["type"]) for candidate_id in oracle_ids}
        oracle_type_groups.update(oracle_types)
        for candidate_id, candidate in candidate_by_id.items():
            candidate_type = str(candidate["type"])
            candidate_type_counts[candidate_type] += 1
            candidate_type_scores[candidate_type].append(float(gold[candidate_id][ranking_metric]))

        score_pearsons.append(_pearson(ranker_scores, ranked_relevance))
        spearmans.append(_spearman(ranker_scores, ranked_relevance))
        kendalls.append(_kendall_tau_b(ranker_scores, ranked_relevance))
        target_distribution = _softmax(ranked_relevance, label_temperature)
        predicted_distribution = _softmax(ranker_scores)
        listwise_kls.append(
            float(
                np.sum(
                    target_distribution
                    * (np.log(target_distribution + EPSILON) - np.log(predicted_distribution + EPSILON))
                )
            )
        )

        group_correct = 0
        group_total = 0
        for left in range(len(ranked_ids)):
            for right in range(left + 1, len(ranked_ids)):
                gold_delta = ranked_relevance[left] - ranked_relevance[right]
                if abs(gold_delta) < minimum_margin:
                    continue
                ranker_delta = ranker_scores[left] - ranker_scores[right]
                correct = ranker_delta * gold_delta > 0
                group_correct += int(correct)
                group_total += 1
        pair_correct += group_correct
        pair_total += group_total
        if group_total:
            per_group_pair_accuracy.append(group_correct / group_total)

        for candidate_id, candidate in candidate_by_id.items():
            if str(candidate["type"]) != "mutation":
                continue
            parent_id = str(candidate.get("parent_id", ""))
            parent = candidate_by_id.get(parent_id)
            if parent is None:
                missing_mutation_parents += 1
                continue
            delta = float(gold[candidate_id][ranking_metric] - gold[parent_id][ranking_metric])
            mutation_deltas.append(delta)
            mutation_by_factor[str(candidate.get("factor_id", "unknown"))].append(delta)
            mutation_by_parent_type[str(parent["type"])].append(delta)

        details.append(
            {
                "sample_id": sample_id,
                "target": target,
                "selected_id": selected_id,
                "selected_type": selected["type"],
                "selected_score": selected_score,
                "oracle_ids": sorted(oracle_ids),
                "oracle_score": oracle_score,
                "oracle_rank": first_oracle_rank,
                "regret": regret,
                "predictions": {method: candidate["text"] for method, candidate in row_candidates.items()},
            }
        )

    metric_intervals = {}
    metrics = {}
    bleu_details = {}
    for offset, (method, method_predictions) in enumerate(methods.items()):
        np.random.seed(seed + offset)
        intervals = corpus_score_with_ci(method_predictions, references)
        bleu = corpus_bleu(method_predictions, references)
        metric_intervals[method] = intervals
        metrics[method] = {metric: values["mid"] for metric, values in intervals.items()}
        metrics[method]["bleu"] = bleu["score"]
        bleu_details[method] = bleu

    pearson_mean, pearson_groups = _mean_valid(score_pearsons)
    spearman_mean, spearman_groups = _mean_valid(spearmans)
    kendall_mean, kendall_groups = _mean_valid(kendalls)
    ranking_metrics = {
        "ranking_label": ranking_metric,
        "tie_aware_hit_at": {str(k): float(np.mean(values)) for k, values in hit_values.items()},
        "mrr_of_oracle": float(np.mean(reciprocal_ranks)),
        "ndcg_at": {str(k): float(np.mean(values)) for k, values in ndcg_values.items()},
        "regret": _summary(regrets),
        "normalized_regret": _summary(normalized_regrets),
        "pairwise_accuracy_micro": pair_correct / pair_total if pair_total else 0.0,
        "pairwise_accuracy_macro": float(np.mean(per_group_pair_accuracy)) if per_group_pair_accuracy else 0.0,
        "evaluated_pair_count": pair_total,
        "pair_minimum_margin": minimum_margin,
        "listwise_kl": float(np.mean(listwise_kls)),
        "listwise_label_temperature": label_temperature,
        "ranker_score_pearson": pearson_mean,
        "ranker_score_pearson_valid_groups": pearson_groups,
        "spearman": spearman_mean,
        "spearman_valid_groups": spearman_groups,
        "kendall_tau_b": kendall_mean,
        "kendall_valid_groups": kendall_groups,
    }

    ranker_rouge_l = metrics["ranker_top1"]["rouge_l"]
    oracle_rouge_l = metrics["pool_oracle"]["rouge_l"]
    oracle_gap = oracle_rouge_l - ranker_rouge_l
    gap_closure = {}
    for baseline in ("task_seed_0", "factor_seed_0"):
        baseline_score = metrics[baseline]["rouge_l"]
        available_gap = oracle_rouge_l - baseline_score
        gap_closure[baseline] = {
            "baseline_rouge_l": baseline_score,
            "absolute_gain": ranker_rouge_l - baseline_score,
            "oracle_available_gap": available_gap,
            "oracle_gap_closure": (
                (ranker_rouge_l - baseline_score) / available_gap
                if abs(available_gap) > EPSILON
                else None
            ),
        }

    comparisons = {
        baseline: _paired_comparison(
            per_query_rouge_l["ranker_top1"],
            per_query_rouge_l[baseline],
            seed + index,
            bootstrap_samples,
        )
        for index, baseline in enumerate(("task_seed_0", "factor_seed_0", "pool_oracle"), 100)
    }
    per_type = {}
    for candidate_type in sorted(candidate_type_counts):
        per_type[candidate_type] = {
            "candidate_count": candidate_type_counts[candidate_type],
            "candidate_score": _summary(candidate_type_scores[candidate_type]),
            "selected_top1_count": selected_types[candidate_type],
            "selected_top1_rate": selected_types[candidate_type] / len(expected_ids),
            "tie_aware_oracle_group_count": oracle_type_groups[candidate_type],
            "tie_aware_oracle_group_rate": oracle_type_groups[candidate_type] / len(expected_ids),
        }

    candidate_pool = {
        "sample_count": len(expected_ids),
        "candidate_count": len(candidate_scores_all),
        "mean_candidates_per_sample": len(candidate_scores_all) / len(expected_ids),
        "candidate_score": _summary(candidate_scores_all),
        "oracle_score": _summary(per_query_rouge_l["pool_oracle"]),
        "score_range": _summary(candidate_ranges),
        "mean_exact_unique_rate_per_group": float(np.mean(unique_rates)),
        "mean_pairwise_candidate_rouge_l": float(np.mean(pairwise_candidate_similarity)),
        "per_type": per_type,
    }
    mutation_analysis = {
        "overall": _delta_report(mutation_deltas, minimum_margin),
        "by_factor": {
            key: _delta_report(values, minimum_margin)
            for key, values in sorted(mutation_by_factor.items())
        },
        "by_parent_type": {
            key: _delta_report(values, minimum_margin)
            for key, values in sorted(mutation_by_parent_type.items())
        },
        "missing_parent_count": missing_mutation_parents,
        "selected_top1_count": selected_types["mutation"],
        "selected_top1_rate": selected_types["mutation"] / len(expected_ids),
        "tie_aware_oracle_group_count": oracle_type_groups["mutation"],
        "tie_aware_oracle_group_rate": oracle_type_groups["mutation"] / len(expected_ids),
    }

    report = {
        "protocol": {
            "name": manifest.get("protocol", "grouped_ranker_pilot"),
            "split": split,
            "sample_count": len(expected_ids),
            "sample_ids": expected_id_list,
            "train_validation_overlap": [],
            "warning": manifest.get(
                "warning", "Held-out development experiment; not an official test result."
            ),
        },
        "metrics": metrics,
        "metric_confidence_intervals": metric_intervals,
        "bleu_details": bleu_details,
        "ranking_metrics": ranking_metrics,
        "oracle_analysis": {
            "ranker_rouge_l": ranker_rouge_l,
            "oracle_rouge_l": oracle_rouge_l,
            "absolute_oracle_gap": oracle_gap,
            "gap_closure": gap_closure,
        },
        "paired_query_comparisons": comparisons,
        "candidate_pool": candidate_pool,
        "mutation_analysis": mutation_analysis,
        "selected_type_counts": dict(selected_types),
        "method_fallback_counts": dict(method_fallback_counts),
        "details": details,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_path = destination.with_suffix(".md")
    formal_split = manifest.get("protocol") == "official_split_ranker_experiment"
    split_label = "Test" if split == "test" else "Dev"
    markdown = [
        f"# MeVO Candidate Ranker — Official {split_label} Split"
        if formal_split
        else "# MeVO Candidate Ranker — Grouped Pilot",
        "",
        f"> {report['protocol']['warning']}",
        "",
        "## Generation quality",
        "",
        "| Method | ROUGE-1 (95% CI) | ROUGE-L (95% CI) | SacreBLEU | Selection |",
        "|---|---:|---:|---:|---|",
    ]
    labels_by_method = {
        "ranker_top1": "Ranker Top-1",
        "task_seed_0": "Task-only seed #0",
        "factor_seed_0": "Factor-conditioned seed #0",
        "best_task_seed": "Best task seed oracle",
        "best_factor_seed": "Best factor seed oracle",
        "pool_oracle": "Candidate-pool oracle",
    }
    for method, values in metrics.items():
        rouge_1_ci = metric_intervals[method]["rouge_1"]
        rouge_l_ci = metric_intervals[method]["rouge_l"]
        selection = "target-blind" if method in {"ranker_top1", "task_seed_0", "factor_seed_0"} else "gold diagnostic"
        markdown.append(
            f"| {labels_by_method[method]} | {values['rouge_1']:.6f} "
            f"[{rouge_1_ci['low']:.6f}, {rouge_1_ci['high']:.6f}] | "
            f"{values['rouge_l']:.6f} [{rouge_l_ci['low']:.6f}, {rouge_l_ci['high']:.6f}] | "
            f"{values['bleu']:.6f} | "
            f"{selection} |"
        )
    if method_fallback_counts:
        markdown.extend(
            [
                "",
                "Factor-seed baseline fallback counts: "
                + ", ".join(
                    f"`{key}={value}`"
                    for key, value in sorted(method_fallback_counts.items())
                )
                + ". These samples have no factor seed in their actual candidate pool.",
            ]
        )

    hit = ranking_metrics["tie_aware_hit_at"]
    ndcg = ranking_metrics["ndcg_at"]
    markdown.extend(
        [
            "",
            "## Ranking quality",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Tie-aware Hit@1 | {hit['1']:.6f} |",
            f"| Tie-aware Hit@3 | {hit['3']:.6f} |",
            f"| Tie-aware Hit@5 | {hit['5']:.6f} |",
            f"| MRR of oracle | {ranking_metrics['mrr_of_oracle']:.6f} |",
            f"| NDCG@5 | {ndcg['5']:.6f} |",
            f"| NDCG@10 | {ndcg['10']:.6f} |",
            f"| Mean regret | {ranking_metrics['regret']['mean']:.6f} |",
            f"| Median regret | {ranking_metrics['regret']['median']:.6f} |",
            f"| P90 regret | {ranking_metrics['regret']['p90']:.6f} |",
            f"| Pairwise accuracy (micro) | {ranking_metrics['pairwise_accuracy_micro']:.6f} |",
            f"| Pairwise accuracy (macro) | {ranking_metrics['pairwise_accuracy_macro']:.6f} |",
            f"| Listwise KL | {ranking_metrics['listwise_kl']:.6f} |",
            f"| Spearman | {ranking_metrics['spearman']:.6f} |",
            f"| Kendall tau-b | {ranking_metrics['kendall_tau_b']:.6f} |",
            "",
            "## Oracle gap",
            "",
            f"Ranker ROUGE-L: `{ranker_rouge_l:.6f}`  ",
            f"Candidate oracle ROUGE-L: `{oracle_rouge_l:.6f}`  ",
            f"Absolute oracle gap: `{oracle_gap:.6f}`",
            "",
            "| Baseline | Gain | Available oracle gap | Gap closure |",
            "|---|---:|---:|---:|",
        ]
    )
    for baseline, values in gap_closure.items():
        closure = values["oracle_gap_closure"]
        markdown.append(
            f"| {labels_by_method[baseline]} | {values['absolute_gain']:.6f} | "
            f"{values['oracle_available_gap']:.6f} | "
            f"{closure:.6f} |" if closure is not None else
            f"| {labels_by_method[baseline]} | {values['absolute_gain']:.6f} | "
            f"{values['oracle_available_gap']:.6f} | n/a |"
        )

    markdown.extend(
        [
            "",
            "## Candidate types",
            "",
            "| Type | Candidates | Mean score | Selected Top-1 | Oracle groups |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for candidate_type, values in per_type.items():
        markdown.append(
            f"| {candidate_type} | {values['candidate_count']} | "
            f"{values['candidate_score']['mean']:.6f} | "
            f"{values['selected_top1_count']} | {values['tie_aware_oracle_group_count']} |"
        )
    mutation = mutation_analysis["overall"]
    markdown.extend(
        [
            "",
            "## Mutation effectiveness",
            "",
            f"Mutations: `{mutation.get('count', 0)}`  ",
            f"Mean child-parent delta: `{mutation.get('mean', 0.0):.6f}`  ",
            f"Win rate: `{mutation.get('win_rate', 0.0):.6f}`  ",
            f"Significant win rate (margin={minimum_margin:.4f}): "
            f"`{mutation.get('significant_win_rate', 0.0):.6f}`",
            "",
            "## Paired query comparisons",
            "",
            "| Baseline | Mean ROUGE-L delta | 95% paired CI | Win / Tie / Loss |",
            "|---|---:|---:|---:|",
        ]
    )
    for baseline, values in comparisons.items():
        markdown.append(
            f"| {labels_by_method[baseline]} | {values['mean_delta_rouge_l']:.6f} | "
            f"[{values['ci95_low']:.6f}, {values['ci95_high']:.6f}] | "
            f"{values['wins']} / {values['ties']} / {values['losses']} |"
        )
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"ranker report -> {destination}, {markdown_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate target-blind ranker predictions")
    parser.add_argument("--config", default=GLOBAL_CONFIG)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["ranker"]
    data_dir = resolve_path(settings["data_dir"])
    output_dir = resolve_path(settings["output_dir"])
    evaluate(
        args.predictions or output_dir / f"{args.split}_predictions.jsonl",
        data_dir / f"{args.split}_candidates.jsonl",
        data_dir / f"{args.split}_labels.jsonl",
        data_dir / "manifest.json",
        args.destination or output_dir / f"{args.split}_report.json",
        int(config["project"]["seed"]),
        float(settings.get("listwise_temperature", 0.1)),
        int(settings.get("evaluation_bootstrap_samples", 5000)),
        split=args.split,
    )


if __name__ == "__main__":
    main()
