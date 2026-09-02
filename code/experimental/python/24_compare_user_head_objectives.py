"""在固定 IDPO 候选池上比较多种 per-user Head 训练目标。

所有目标共享一次冻结 DeBERTa 特征编码。每个用户的 LOO Query 做确定性 80/20
划分；留出历史只用于早停和选择 Global/User 融合权重，当前 Test Gold 始终只在
最终选定候选后计算指标。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from common import user_head  # noqa: E402
from common.metrics import corpus_bleu, corpus_score_with_ci, score  # noqa: E402
from idpo_common import idpo_path  # noqa: E402
from pipeline_common import load_config, read_jsonl, resolve_path, write_json, write_jsonl  # noqa: E402


OBJECTIVES = {
    "top_pairs",
    "listwise_kl",
    "oracle_set",
    "expected_regret",
    "hybrid_top1",
}


def _stable_value(seed: int, user_id: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{user_id}:{sample_id}".encode()).hexdigest()


def split_user_records(
    records: list[dict[str, Any]], validation_fraction: float, seed: int, user_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按稳定哈希划分 LOO Query，避免连续论文 ID 带来的时间/主题偏差。"""

    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction 必须在 (0, 0.5) 内")
    ordered = sorted(
        records,
        key=lambda item: _stable_value(seed, user_id, str(item["sample_id"])),
    )
    validation_count = max(5, int(round(len(ordered) * validation_fraction)))
    validation_count = min(validation_count, len(ordered) - 1)
    return ordered[validation_count:], ordered[:validation_count]


def _ranker_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = {
        **config["scorer"],
        "shuffle_factor_directions": False,
        "factor_dropout_probability": 0.0,
    }
    if str(settings.get("input_mode", "task_only")) != "task_only":
        raise ValueError("Objective matrix 要求 Ranker 输入严格为 q+c")
    return settings


def _selected_users(config: dict[str, Any], rows: list[dict[str, Any]]) -> set[str]:
    users = sorted({str(row.get("user_id", "")) for row in rows})
    limit = int(config.get("idpo", {}).get("user_limit", 0))
    return set(users[:limit] if limit > 0 else users)


@torch.no_grad()
def _encode_rows(
    rows: list[dict[str, Any]],
    *,
    user_field: str,
    encoder: nn.Module,
    tokenizer,
    ranker: dict[str, Any],
    candidate_batch_size: int,
    device: torch.device,
    include_output_metadata: bool,
) -> dict[str, list[dict[str, Any]]]:
    """按候选数动态打包多个 Query，避免短候选池浪费显存。"""

    collator = user_head._global_ranker.GroupCollator(
        tokenizer, int(ranker["max_length"]), ranker, training=False
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pending_views: list[dict[str, Any]] = []
    pending_rows: list[dict[str, Any]] = []
    pending_candidates = 0

    def flush() -> None:
        nonlocal pending_views, pending_rows, pending_candidates
        if not pending_views:
            return
        batch = collator(pending_views)
        group_sizes = list(batch.pop("group_sizes"))
        labels = batch.pop("label_scores").float().cpu()
        encoded = {key: value.to(device) for key, value in batch.items()}
        features = user_head._pooled(encoder, encoded).float().cpu()
        feature_groups = torch.split(features, group_sizes)
        label_groups = torch.split(labels, group_sizes)
        for raw, view, group_features, group_labels in zip(
            pending_rows, pending_views, feature_groups, label_groups
        ):
            user_id = str(raw[user_field])
            record: dict[str, Any] = {
                "sample_id": str(raw["id"]),
                "user_id": user_id,
                "features": group_features.contiguous(),
                "labels": group_labels.contiguous(),
            }
            if include_output_metadata:
                record.update(
                    {
                        "target": str(raw["target"]),
                        "candidates": [dict(item) for item in view["candidates"]],
                    }
                )
            grouped[user_id].append(record)
        pending_views = []
        pending_rows = []
        pending_candidates = 0

    for row in rows:
        view = user_head._listwise_view(row, "rouge_l")
        count = len(view["candidates"])
        if pending_views and pending_candidates + count > candidate_batch_size:
            flush()
        pending_views.append(view)
        pending_rows.append(row)
        pending_candidates += count
    flush()
    return dict(grouped)


def _feature_cache_key(config: dict[str, Any], adaptation_path: Path, current_path: Path) -> dict[str, Any]:
    ranker = _ranker_settings(config)
    return {
        "version": 1,
        "adaptation_size": adaptation_path.stat().st_size,
        "current_size": current_path.stat().st_size,
        "global_model_dir": str(resolve_path(config["paths"]["scorer_output_dir"])),
        "max_length": int(ranker["max_length"]),
        "input_mode": str(ranker["input_mode"]),
        "user_limit": int(config.get("idpo", {}).get("user_limit", 0)),
    }


def _load_or_encode(config: dict[str, Any]) -> dict[str, Any]:
    matrix = config["user_head_matrix"]
    round_index = int(config["idpo"]["round"])
    adaptation_path = idpo_path(
        config, round_index, "user_ranker/test_adaptation_rows.jsonl"
    )
    current_path = idpo_path(config, round_index, "test_current_editor_scored.jsonl")
    cache_path = resolve_path(matrix["feature_cache"])
    cache_key = _feature_cache_key(config, adaptation_path, current_path)
    if bool(matrix.get("reuse_feature_cache", True)) and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("cache_key") == cache_key:
            print(f"user-head matrix feature cache hit -> {cache_path}", flush=True)
            return payload

    ranker = _ranker_settings(config)
    device = user_head._device(ranker)
    seed = int(config["project"]["seed"])
    settings = {
        **config["user_adaptation"],
        "global_model_dir": str(resolve_path(config["paths"]["scorer_output_dir"])),
    }
    encoder, tokenizer, global_head, metadata = user_head._load_adaptation_components(
        ranker, settings, device, seed
    )
    adaptation_rows = read_jsonl(adaptation_path)
    current_rows = read_jsonl(current_path)
    selected = _selected_users(config, current_rows)
    adaptation_rows = [
        row for row in adaptation_rows if str(row.get("parent_sample_id", "")) in selected
    ]
    current_rows = [row for row in current_rows if str(row.get("user_id", "")) in selected]
    batch_size = int(matrix.get("encoding_batch_size", 256))
    print(
        f"user-head matrix encoding adaptation={len(adaptation_rows)} current={len(current_rows)} "
        f"users={len(selected)} candidate_batch={batch_size}",
        flush=True,
    )
    adaptation = _encode_rows(
        adaptation_rows,
        user_field="parent_sample_id",
        encoder=encoder,
        tokenizer=tokenizer,
        ranker=ranker,
        candidate_batch_size=batch_size,
        device=device,
        include_output_metadata=False,
    )
    print("user-head matrix adaptation features encoded", flush=True)
    current = _encode_rows(
        current_rows,
        user_field="user_id",
        encoder=encoder,
        tokenizer=tokenizer,
        ranker=ranker,
        candidate_batch_size=batch_size,
        device=device,
        include_output_metadata=True,
    )
    payload = {
        "cache_key": cache_key,
        "adaptation": adaptation,
        "current": current,
        "initial_state": {
            key: value.detach().float().cpu() for key, value in global_head.state_dict().items()
        },
        "hidden_size": int(global_head.in_features),
        "metadata": metadata,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"user-head matrix feature cache -> {cache_path}", flush=True)
    return payload


def _flatten(records: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    return (
        torch.cat([record["features"] for record in records]),
        torch.cat([record["labels"] for record in records]),
        [len(record["labels"]) for record in records],
    )


def objective_loss(
    objective: str,
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: list[int],
    settings: dict[str, Any],
) -> torch.Tensor:
    module = user_head._global_ranker
    if objective == "top_pairs":
        return module._group_pairwise_loss(
            scores,
            labels,
            group_sizes,
            float(settings["pair_minimum_margin"]),
            "top_pairs",
            int(settings["max_pairs_per_group"]),
        )
    if objective == "listwise_kl":
        return module._listwise_loss(
            scores,
            labels,
            group_sizes,
            float(settings["label_temperature"]),
            float(settings["score_temperature"]),
        )
    if objective == "oracle_set":
        return module._oracle_set_loss(
            scores, labels, group_sizes, float(settings["score_temperature"])
        )
    if objective == "expected_regret":
        return module._expected_regret_loss(
            scores, labels, group_sizes, float(settings["regret_temperature"])
        )
    if objective == "hybrid_top1":
        oracle = module._oracle_set_loss(
            scores, labels, group_sizes, float(settings["score_temperature"])
        )
        pairs = module._group_pairwise_loss(
            scores,
            labels,
            group_sizes,
            float(settings["pair_minimum_margin"]),
            "top_pairs",
            int(settings["max_pairs_per_group"]),
        )
        pointwise = module._pointwise_score_loss(scores, labels, group_sizes)
        return oracle + 0.3 * pairs + 0.5 * pointwise
    raise ValueError(f"Unknown user-head objective={objective}")


@torch.no_grad()
def _head_metrics(
    scorer: nn.Module,
    records: list[dict[str, Any]],
    objective: str,
    settings: dict[str, Any],
) -> dict[str, float]:
    features, labels, sizes = _flatten(records)
    scores = scorer(features).squeeze(-1)
    loss = objective_loss(objective, scores, labels, sizes, settings)
    hits = 0
    regrets = []
    pair_correct = 0
    pair_total = 0
    offset = 0
    for size in sizes:
        group_scores = scores[offset : offset + size]
        group_labels = labels[offset : offset + size]
        chosen = int(group_scores.argmax().item())
        best = float(group_labels.max().item())
        chosen_label = float(group_labels[chosen].item())
        hits += int(chosen_label >= best - 1.0e-8)
        regrets.append(best - chosen_label)
        for left in range(size):
            for right in range(left + 1, size):
                gap = float(group_labels[left] - group_labels[right])
                if abs(gap) < float(settings["pair_minimum_margin"]):
                    continue
                predicted = float(group_scores[left] - group_scores[right])
                pair_correct += int((gap > 0 and predicted > 0) or (gap < 0 and predicted < 0))
                pair_total += 1
        offset += size
    return {
        "loss": float(loss.item()),
        "hit_at_1": hits / len(records),
        "mean_regret": statistics.fmean(regrets),
        "pair_accuracy": pair_correct / max(pair_total, 1),
    }


def _new_head(hidden_size: int, initial_state: dict[str, torch.Tensor]) -> nn.Linear:
    scorer = nn.Linear(hidden_size, 1)
    scorer.load_state_dict(initial_state)
    return scorer


def _train_one(
    objective: str,
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    hidden_size: int,
    initial_state: dict[str, torch.Tensor],
    settings: dict[str, Any],
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    scorer = _new_head(hidden_size, initial_state)
    initial = {key: value.detach().clone() for key, value in scorer.state_dict().items()}
    optimizer = torch.optim.AdamW(
        scorer.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    initial_validation = _head_metrics(
        scorer, validation_records, objective, settings
    )
    best_state = copy.deepcopy(scorer.state_dict())
    best_epoch = 0
    best_key = (
        initial_validation["mean_regret"],
        -initial_validation["hit_at_1"],
        initial_validation["loss"],
    )
    history = []
    rng = random.Random(seed)
    patience = int(settings.get("early_stopping_patience", 5))
    stale = 0
    for epoch in range(1, int(settings["epochs"]) + 1):
        order = list(range(len(train_records)))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), int(settings["group_batch_size"])):
            batch = [train_records[index] for index in order[start : start + int(settings["group_batch_size"])]]
            features, labels, sizes = _flatten(batch)
            optimizer.zero_grad(set_to_none=True)
            scores = scorer(features).squeeze(-1)
            task_loss = objective_loss(objective, scores, labels, sizes, settings)
            anchor = sum(
                (parameter - initial[name]).pow(2).mean()
                for name, parameter in scorer.named_parameters()
            )
            loss = task_loss + float(settings["anchor_strength"]) * anchor
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                scorer.parameters(), float(settings["max_grad_norm"])
            )
            optimizer.step()
            losses.append(float(task_loss.detach().item()))
        validation = _head_metrics(scorer, validation_records, objective, settings)
        history.append(
            {
                "epoch": epoch,
                "train_loss": statistics.fmean(losses),
                "validation": validation,
            }
        )
        key = (
            validation["mean_regret"],
            -validation["hit_at_1"],
            validation["loss"],
        )
        if key < best_key:
            best_key = key
            best_epoch = epoch
            best_state = copy.deepcopy(scorer.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    scorer.load_state_dict(best_state)
    return (
        {key: value.detach().cpu() for key, value in best_state.items()},
        {
            "train_queries": len(train_records),
            "validation_queries": len(validation_records),
            "best_epoch": best_epoch,
            "initial_validation": initial_validation,
            "best_validation": _head_metrics(
                scorer, validation_records, objective, settings
            ),
            "history": history,
        },
    )


def _score_records(
    state: dict[str, torch.Tensor],
    initial_state: dict[str, torch.Tensor],
    hidden_size: int,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    user_scorer = _new_head(hidden_size, state).eval()
    global_scorer = _new_head(hidden_size, initial_state).eval()
    values = []
    with torch.no_grad():
        for record in records:
            values.append(
                {
                    "sample_id": record["sample_id"],
                    "global_scores": global_scorer(record["features"]).squeeze(-1),
                    "user_scores": user_scorer(record["features"]).squeeze(-1),
                    "labels": record["labels"],
                }
            )
    return values


def _select_alpha(records: list[dict[str, Any]], grid: list[float]) -> tuple[float, list[dict[str, Any]]]:
    return user_head._select_blend_alpha(records, grid)


def _evaluate_route(
    objective: str,
    route: str,
    current: dict[str, list[dict[str, Any]]],
    states: dict[str, dict[str, torch.Tensor]],
    initial_state: dict[str, torch.Tensor],
    hidden_size: int,
    alphas: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = []
    global_scorer = _new_head(hidden_size, initial_state).eval()
    for user_id, records in sorted(current.items()):
        user_scorer = _new_head(hidden_size, states[user_id]).eval()
        alpha = float(alphas[user_id])
        with torch.no_grad():
            for record in records:
                global_scores = global_scorer(record["features"]).squeeze(-1)
                user_scores = user_scorer(record["features"]).squeeze(-1)
                final_scores = (1.0 - alpha) * global_scores + alpha * user_scores
                chosen = int(final_scores.argmax().item())
                global_chosen = int(global_scores.argmax().item())
                candidate = record["candidates"][chosen]
                metrics = score(str(candidate["text"]), str(record["target"]))
                best_label = float(record["labels"].max().item())
                chosen_label = float(record["labels"][chosen].item())
                predictions.append(
                    {
                        "objective": objective,
                        "route": route,
                        "sample_id": record["sample_id"],
                        "user_id": user_id,
                        "alpha": alpha,
                        "selected_id": str(candidate["candidate_id"]),
                        "prediction": str(candidate["text"]),
                        "target": str(record["target"]),
                        "rouge_1": float(metrics["rouge_1"]),
                        "rouge_l": float(metrics["rouge_l"]),
                        "oracle_rouge_l": best_label,
                        "regret": best_label - chosen_label,
                        "hit_at_1": chosen_label >= best_label - 1.0e-8,
                        "changed_from_global": chosen != global_chosen,
                    }
                )
    outputs = [row["prediction"] for row in predictions]
    targets = [row["target"] for row in predictions]
    report = {
        "queries": len(predictions),
        "users": len(current),
        "rouge": corpus_score_with_ci(outputs, targets),
        "sacrebleu": corpus_bleu(outputs, targets),
        "query_macro": {
            "rouge_1": statistics.fmean(row["rouge_1"] for row in predictions),
            "rouge_l": statistics.fmean(row["rouge_l"] for row in predictions),
        },
        "hit_at_1": statistics.fmean(float(row["hit_at_1"]) for row in predictions),
        "mean_regret": statistics.fmean(row["regret"] for row in predictions),
        "changed_from_global": sum(row["changed_from_global"] for row in predictions),
        "alpha_mean": statistics.fmean(alphas.values()),
        "alpha_nonzero_users": sum(alpha > 0 for alpha in alphas.values()),
        "alpha_distribution": dict(sorted(Counter(alphas.values()).items())),
        "current_query_gold_used_for_training_or_alpha": False,
    }
    return report, predictions


def run(config: dict[str, Any]) -> Path:
    matrix = config["user_head_matrix"]
    objectives = [str(value) for value in matrix["objectives"]]
    unknown = set(objectives) - OBJECTIVES
    if unknown:
        raise ValueError(f"Unknown objectives: {sorted(unknown)}")
    payload = _load_or_encode(config)
    adaptation = payload["adaptation"]
    current = payload["current"]
    initial_state = payload["initial_state"]
    hidden_size = int(payload["hidden_size"])
    seed = int(config["project"]["seed"])
    users = sorted(current)
    if set(adaptation) != set(users):
        raise ValueError("Adaptation/current users do not match")
    splits = {
        user_id: split_user_records(
            adaptation[user_id],
            float(matrix["validation_fraction"]),
            seed,
            user_id,
        )
        for user_id in users
    }
    output_dir = resolve_path(matrix["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_grid = [float(value) for value in matrix["alpha_grid"]]
    all_reports: dict[str, Any] = {}
    all_predictions = []

    # 同一 IDPO 候选池上的共享 Head 是所有个性化方案的严格下界/回退点。
    global_states = {user_id: initial_state for user_id in users}
    global_alphas = {user_id: 0.0 for user_id in users}
    global_report, global_predictions = _evaluate_route(
        "global_head", "global", current, global_states, initial_state, hidden_size, global_alphas
    )
    all_reports["global_head"] = global_report
    all_predictions.extend(global_predictions)
    print(
        f"objective global rouge_l={global_report['query_macro']['rouge_l']:.6f} "
        f"hit1={global_report['hit_at_1']:.4f}",
        flush=True,
    )

    for objective in objectives:
        variant_dir = output_dir / objective
        heads_dir = variant_dir / "user_heads"
        heads_dir.mkdir(parents=True, exist_ok=True)
        states = {}
        per_user = {}
        validation_records_by_user = {}
        for index, user_id in enumerate(users, 1):
            train_records, validation_records = splits[user_id]
            state, report = _train_one(
                objective,
                train_records,
                validation_records,
                hidden_size,
                initial_state,
                matrix,
                user_head._user_seed(seed, f"{objective}:{user_id}"),
            )
            states[user_id] = state
            torch.save(state, heads_dir / f"user_{user_id}.pt")
            validation_scores = _score_records(
                state, initial_state, hidden_size, validation_records
            )
            validation_records_by_user[user_id] = validation_scores
            alpha, alpha_report = _select_alpha(validation_scores, alpha_grid)
            per_user[user_id] = {
                **report,
                "selected_alpha": alpha,
                "alpha_validation": alpha_report,
            }
            print(
                f"objective={objective} user={index}/{len(users)} id={user_id} "
                f"epoch={report['best_epoch']} val_regret={report['best_validation']['mean_regret']:.4f} "
                f"alpha={alpha:.1f}",
                flush=True,
            )
        flat_validation = [
            record for user_id in users for record in validation_records_by_user[user_id]
        ]
        shared_alpha, shared_alpha_report = _select_alpha(flat_validation, alpha_grid)
        routes = {
            "per_user_alpha": {
                user_id: float(per_user[user_id]["selected_alpha"]) for user_id in users
            },
            "shared_alpha": {user_id: shared_alpha for user_id in users},
            "user_only": {user_id: 1.0 for user_id in users},
        }
        route_reports = {}
        for route, alphas in routes.items():
            report, predictions = _evaluate_route(
                objective, route, current, states, initial_state, hidden_size, alphas
            )
            route_reports[route] = report
            all_predictions.extend(predictions)
        variant_report = {
            "objective": objective,
            "train_validation_protocol": "per-user deterministic 80/20 LOO split",
            "shared_alpha": shared_alpha,
            "shared_alpha_validation": shared_alpha_report,
            "routes": route_reports,
            "per_user": per_user,
        }
        write_json(variant_dir / "report.json", variant_report)
        all_reports[objective] = variant_report
        best_route = min(
            route_reports.items(), key=lambda item: item[1]["mean_regret"]
        )
        print(
            f"objective done={objective} best_test_route={best_route[0]} "
            f"rouge_l={best_route[1]['query_macro']['rouge_l']:.6f} "
            f"hit1={best_route[1]['hit_at_1']:.4f}",
            flush=True,
        )
    summary = {
        "protocol": "per_user_head_objective_matrix_v1",
        "queries": sum(len(values) for values in current.values()),
        "users": len(users),
        "objectives": objectives,
        "validation_fraction": float(matrix["validation_fraction"]),
        "current_query_gold_used_for_training_or_model_selection": False,
        "settings": {
            key: matrix[key]
            for key in (
                "epochs",
                "learning_rate",
                "anchor_strength",
                "label_temperature",
                "score_temperature",
                "regret_temperature",
                "alpha_grid",
            )
        },
        "results": all_reports,
    }
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "predictions.jsonl", all_predictions)
    print(f"user-head objective matrix -> {output_dir}", flush=True)
    return output_dir / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="比较 per-user Ranker Head 训练目标")
    parser.add_argument(
        "--config", default=str(HERE / "config_user_head_objective_matrix.yaml")
    )
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()

