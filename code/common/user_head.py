"""Per-user Ranker adaptation and inference.

Two initialization protocols are supported. ``global_checkpoint`` preserves
the original two-stage route. ``pretrained`` skips cross-user Ranker training:
it freezes a generic pretrained encoder and learns an isolated Linear Head or
pooled-feature Adapter from each user's own profile pseudo queries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from common.utils import load_config, read_jsonl, resolve_path, write_jsonl
from common.runtime import USER_CONFIG, load_stage


_ranker_data = load_stage("07_build_ranker_data.py")
_global_ranker = load_stage("08_train_global_ranker.py")
_pair_rows = _ranker_data._pair_rows
_listwise_view = _ranker_data._listwise_view
PairCollator = _global_ranker.PairCollator
_candidate_text = _global_ranker._candidate_text
_context_text = _global_ranker._context_text


def _user_seed(seed: int, user_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{user_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _device(settings: dict) -> torch.device:
    name = str(settings.get("device", "cuda"))
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("user-head warning: CUDA unavailable; falling back to CPU", flush=True)
        name = "cpu"
    return torch.device(name)


def _pooled(encoder: nn.Module, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = encoder(**encoded)
    hidden = outputs.last_hidden_state
    sequence_pooler = getattr(encoder, "_mevo_sequence_pooler", None)
    if sequence_pooler is not None:
        # Reproduce the parent sequence-classifier feature path exactly.  For
        # DeBERTa this is its trained ContextPooler (first-token + dense +
        # activation), not mean pooling.
        return sequence_pooler(hidden)
    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


@torch.no_grad()
def _encode_pairs(
    rows: list[dict],
    encoder: nn.Module,
    tokenizer,
    ranker_settings: dict,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    collator = PairCollator(
        tokenizer,
        int(ranker_settings["max_length"]),
        ranker_settings,
        training=False,
    )
    features = []
    weights = []
    encoder.eval()
    for start in range(0, len(rows), batch_size):
        batch = collator(rows[start : start + batch_size])
        pair_count = int(batch.pop("pair_count"))
        pair_weights = batch.pop("weights")
        encoded = {key: value.to(device) for key, value in batch.items()}
        pooled = _pooled(encoder, encoded).view(pair_count, 2, -1)
        features.append(pooled.float().cpu())
        weights.append(pair_weights.float().cpu())
    return torch.cat(features), torch.cat(weights)


def _pair_metrics(
    scorer: nn.Module, features: torch.Tensor, weights: torch.Tensor
) -> tuple[float, float]:
    with torch.no_grad():
        scores = scorer(features).squeeze(-1)
        deltas = scores[:, 0] - scores[:, 1]
        loss = (F.softplus(-deltas) * weights).mean()
        accuracy = (deltas > 0).float().mean()
    return float(loss.item()), float(accuracy.item())


class UserAdapterScorer(nn.Module):
    """A light per-user adapter on frozen pooled backbone features.

    The zero-initialized up projection makes the adapter start exactly at the
    supplied initial head. Only this module is copied and optimized per user;
    the Transformer backbone remains frozen. The initial head can come from a
    Global Ranker or be an untrained deterministic head on a generic encoder.
    """

    def __init__(self, hidden_size: int, rank: int, global_head: nn.Linear):
        super().__init__()
        if rank < 1:
            raise ValueError("user_adaptation.adapter_rank must be positive")
        self.down = nn.Linear(hidden_size, rank)
        self.up = nn.Linear(rank, hidden_size)
        self.scorer = nn.Linear(hidden_size, 1)
        self.scorer.load_state_dict(global_head.state_dict())
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        adapted = features + self.up(torch.tanh(self.down(features)))
        return self.scorer(adapted)


class UserResidualScorer(nn.Module):
    """Frozen content score plus a zero-initialized per-user residual.

    The base head is kept fixed.  At initialization this scorer is exactly the
    base Ranker and adaptation only learns a small user-specific correction
    ``delta_u(features)``.  This makes the personalized component explicit and
    keeps the content model from being overwritten by sparse profile data.
    """

    def __init__(self, hidden_size: int, global_head: nn.Linear):
        super().__init__()
        self.base = nn.Linear(hidden_size, 1)
        self.base.load_state_dict(global_head.state_dict())
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.delta = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.base(features) + self.delta(features)


class ClassificationFeaturePooler(nn.Module):
    """Expose the penultimate feature used by an XLM-R classification head."""

    def __init__(self, dense: nn.Linear):
        super().__init__()
        self.dense = dense

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.dense(hidden[:, 0]))


def _build_user_scorer(
    mode: str, hidden_size: int, global_head: nn.Linear, adapter_rank: int
) -> nn.Module:
    mode = str(mode).lower()
    if mode == "linear_head":
        scorer = nn.Linear(hidden_size, 1)
        scorer.load_state_dict(global_head.state_dict())
        return scorer
    if mode == "adapter":
        return UserAdapterScorer(hidden_size, adapter_rank, global_head)
    if mode == "residual_linear":
        return UserResidualScorer(hidden_size, global_head)
    raise ValueError(
        "user_adaptation.mode must be linear_head, residual_linear, or adapter"
    )


def _load_global_components(global_model_dir: Path, device: torch.device):
    """Load a shared global backbone and its one global scoring head.

    Per-user adaptation needs the *full* backbone checkpoint, not only a
    trainable delta.  The loader still accepts delta checkpoints when the
    corresponding base model is available locally, which keeps old runs
    usable while making the requirement explicit in the error message.
    """
    report_path = global_model_dir / "training_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Global backbone report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    architecture = str(report.get("model_architecture", "encoder_pooling"))
    if (global_model_dir / "encoder").exists():
        if architecture == "sequence_classifier":
            tokenizer = AutoTokenizer.from_pretrained(
                global_model_dir / "encoder", local_files_only=True
            )
            # The sequence-classifier checkpoint is loaded once to recover its
            # trained classifier; adaptation itself only receives its base.
            classifier_full = AutoModelForSequenceClassification.from_pretrained(
                global_model_dir / "encoder", num_labels=1, local_files_only=True
            )
            base = getattr(classifier_full, classifier_full.base_model_prefix)
            global_head = getattr(classifier_full, "classifier", None)
            if not isinstance(global_head, nn.Linear):
                raise ValueError("Global sequence-classifier checkpoint has no Linear classifier")
            encoder = base
            encoder._mevo_sequence_pooler = classifier_full.pooler
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                global_model_dir / "encoder", local_files_only=True
            )
            encoder = AutoModel.from_pretrained(
                global_model_dir / "encoder", local_files_only=True
            )
            head_path = global_model_dir / "ranker_head.pt"
            if not head_path.exists():
                raise FileNotFoundError(
                    "Per-user adaptation requires a full global checkpoint with ranker_head.pt"
                )
            global_head = nn.Linear(int(encoder.config.hidden_size), 1)
            global_head.load_state_dict(torch.load(head_path, map_location="cpu"))
    elif (global_model_dir / "model_delta.pt").exists():
        # Delta checkpoints can be reconstructed if the base model is cached.
        model, tokenizer = _global_ranker._load_model(global_model_dir, device)
        if architecture == "sequence_classifier":
            classifier_full = model.encoder
            global_head = getattr(classifier_full, "classifier", None)
            # CandidateRanker.encoder is the complete
            # *ForSequenceClassification model in this architecture.  Per-user
            # heads need pooled hidden features, so expose only its base
            # Transformer; otherwise forward() returns SequenceClassifierOutput
            # without last_hidden_state.
            encoder = getattr(classifier_full, classifier_full.base_model_prefix)
            encoder._mevo_sequence_pooler = classifier_full.pooler
        else:
            encoder = model.encoder
            global_head = model.scorer
        if not isinstance(global_head, nn.Linear):
            raise ValueError("Global delta checkpoint does not expose a Linear scoring head")
    else:
        raise FileNotFoundError(
            f"Global backbone checkpoint not found under {global_model_dir}; "
            "run the global backbone stage first"
        )
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder, tokenizer, global_head


def _load_pretrained_components(
    model_name: str,
    device: torch.device,
    seed: int,
    local_files_only: bool = True,
):
    """Load an immutable generic encoder and a reproducible untrained head."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    hidden_size = int(encoder.config.hidden_size)
    initial_head = nn.Linear(hidden_size, 1)
    nn.init.normal_(initial_head.weight, mean=0.0, std=0.02)
    nn.init.zeros_(initial_head.bias)
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder, tokenizer, initial_head


def _load_pretrained_reranker_components(
    model_name: str,
    device: torch.device,
    local_files_only: bool = True,
):
    """Load a frozen sequence-classification reranker and preserve its logits."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    full_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, local_files_only=local_files_only
    )
    encoder = getattr(full_model, full_model.base_model_prefix)
    classifier = getattr(full_model, "classifier", None)
    dense = getattr(classifier, "dense", None)
    out_proj = getattr(classifier, "out_proj", None)
    if not isinstance(dense, nn.Linear) or not isinstance(out_proj, nn.Linear):
        raise ValueError(
            f"Unsupported pretrained reranker classification head: {type(classifier).__name__}"
        )
    # In eval mode the original classification head is exactly:
    # first token -> dense -> tanh -> out_proj. Attach the first three steps to
    # the frozen encoder so _pooled() returns the feature expected by out_proj.
    encoder._mevo_sequence_pooler = ClassificationFeaturePooler(dense)
    initial_head = nn.Linear(int(out_proj.in_features), 1)
    initial_head.load_state_dict(out_proj.state_dict())
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder, tokenizer, initial_head


def _load_adaptation_components(
    ranker: dict,
    settings: dict,
    device: torch.device,
    seed: int,
):
    initialization = str(settings.get("initialization", "global_checkpoint")).lower()
    if initialization == "pretrained":
        model_name = str(settings.get("base_model_name", ranker["model_name"]))
        encoder, tokenizer, initial_head = _load_pretrained_components(
            model_name,
            device,
            seed,
            bool(settings.get("local_files_only", True)),
        )
        metadata = {
            "initialization": initialization,
            "base_model_name": model_name,
            "backbone_architecture": "encoder_pooling",
            "global_model_dir": None,
        }
        return encoder, tokenizer, initial_head, metadata
    if initialization == "pretrained_reranker":
        model_name = str(settings.get("base_model_name", ranker["model_name"]))
        encoder, tokenizer, initial_head = _load_pretrained_reranker_components(
            model_name,
            device,
            bool(settings.get("local_files_only", True)),
        )
        metadata = {
            "initialization": initialization,
            "base_model_name": model_name,
            "backbone_architecture": "sequence_classifier",
            "global_model_dir": None,
        }
        return encoder, tokenizer, initial_head, metadata
    if initialization != "global_checkpoint":
        raise ValueError(
            "user_adaptation.initialization must be pretrained, "
            "pretrained_reranker, or global_checkpoint"
        )
    global_model_dir = resolve_path(settings["global_model_dir"])
    encoder, tokenizer, global_head = _load_global_components(global_model_dir, device)
    report = json.loads(
        (global_model_dir / "training_report.json").read_text(encoding="utf-8")
    )
    metadata = {
        "initialization": initialization,
        "base_model_name": str(report.get("model_name", "")),
        "backbone_architecture": str(
            report.get("model_architecture", "encoder_pooling")
        ),
        "global_model_dir": str(global_model_dir),
    }
    return encoder, tokenizer, global_head, metadata


def fit_user_head(
    initial_state: dict[str, torch.Tensor],
    features: torch.Tensor,
    weights: torch.Tensor,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    anchor_strength: float,
    batch_size: int,
    max_grad_norm: float,
    seed: int,
    scorer_factory=None,
) -> tuple[dict[str, torch.Tensor], dict]:
    if not len(features):
        raise ValueError("Cannot adapt a user head without preference pairs")
    hidden_size = int(features.shape[-1])
    scorer = scorer_factory() if scorer_factory is not None else nn.Linear(hidden_size, 1)
    scorer.load_state_dict(initial_state)
    initial = {key: value.detach().clone() for key, value in scorer.state_dict().items()}
    initial_loss, initial_accuracy = _pair_metrics(scorer, features, weights)
    optimizer = torch.optim.AdamW(
        scorer.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(features), generator=generator)
        loss_sum = 0.0
        pair_count = 0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            batch_features = features[indices]
            batch_weights = weights[indices]
            optimizer.zero_grad(set_to_none=True)
            scores = scorer(batch_features).squeeze(-1)
            pair_losses = F.softplus(-(scores[:, 0] - scores[:, 1])) * batch_weights
            anchor = sum(
                (parameter - initial[name]).pow(2).mean()
                for name, parameter in scorer.named_parameters()
            )
            loss = pair_losses.mean() + anchor_strength * anchor
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), max_grad_norm)
            optimizer.step()
            loss_sum += float(pair_losses.detach().sum().item())
            pair_count += len(indices)
        _, accuracy = _pair_metrics(scorer, features, weights)
        history.append(
            {
                "epoch": epoch,
                "pair_loss": loss_sum / max(pair_count, 1),
                "pair_accuracy": accuracy,
            }
        )
    final_loss, final_accuracy = _pair_metrics(scorer, features, weights)
    state = {key: value.detach().cpu() for key, value in scorer.state_dict().items()}
    return state, {
        "pairs": len(features),
        "initial_pair_loss": initial_loss,
        "initial_pair_accuracy": initial_accuracy,
        "final_pair_loss": final_loss,
        "final_pair_accuracy": final_accuracy,
        "history": history,
    }


def _adaptation_pairs(rows: list[dict], settings: dict, metric: str) -> dict[str, list[dict]]:
    by_user: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        user_id = str(row.get("parent_sample_id", ""))
        if not user_id:
            raise ValueError(f"Adaptation sample {row.get('id')} has no parent_sample_id")
        by_user[user_id].extend(
            _pair_rows(
                row,
                str(settings.get("pair_strategy", "all_pairs")),
                metric,
                float(settings.get("pair_minimum_margin", 0.02)),
                int(settings.get("max_pairs_per_sample", 45)),
            )
        )
    return dict(by_user)


@torch.no_grad()
def _encode_calibration_group(
    row: dict,
    encoder: nn.Module,
    tokenizer,
    ranker_settings: dict,
    metric: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    group = _listwise_view(row, metric)
    candidates = group["candidates"]
    input_mode = str(ranker_settings.get("input_mode", "task_only"))
    raw_text_input = bool(ranker_settings.get("raw_text_input", False))
    encoded = _global_ranker._encode_pairs(
        tokenizer,
        [_candidate_text(candidate, input_mode, raw_text_input) for candidate in candidates],
        [
            _context_text(group, input_mode, raw_text_input=raw_text_input)
            for _ in candidates
        ],
        ranker_settings,
        int(ranker_settings["max_length"]),
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    return _pooled(encoder, encoded).float().cpu(), torch.tensor(
        group["label_scores"], dtype=torch.float32
    )


def _select_blend_alpha(records: list[dict], grid: list[float]) -> tuple[float, list[dict]]:
    """Select a shared Global/User interpolation using held-out history."""
    if not records:
        return 1.0, []
    results = []
    for alpha in grid:
        regrets = []
        hits = 0
        for record in records:
            scores = (1.0 - alpha) * record["global_scores"] + alpha * record["user_scores"]
            labels = record["labels"]
            chosen = int(scores.argmax().item())
            regret = float(labels.max().item() - labels[chosen].item())
            regrets.append(regret)
            hits += int(regret <= 1.0e-8)
        results.append({
            "alpha": float(alpha),
            "mean_regret": sum(regrets) / len(regrets),
            "hit_at_1": hits / len(regrets),
            "groups": len(regrets),
        })
    best = min(
        results,
        key=lambda item: (item["mean_regret"], -item["hit_at_1"], item["alpha"]),
    )
    return float(best["alpha"]), results


def _select_per_user_blend_alphas(
    records: list[dict], grid: list[float]
) -> tuple[dict[str, float], dict[str, list[dict]]]:
    """Choose a target-blind residual strength from each user's held-out history."""
    records_by_user: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        records_by_user[str(record["user_id"])].append(record)
    alphas = {}
    results = {}
    for user_id, user_records in records_by_user.items():
        alpha, user_results = _select_blend_alpha(user_records, grid)
        alphas[user_id] = alpha
        results[user_id] = user_results
    return alphas, results


def _validation_user_id(row: dict, settings: dict) -> str:
    """Resolve the identity used to select a personalized head.

    Historical experiments used the current sample ID as the adaptation key.
    Per-Pcs can contain several current queries for one real user, so new
    experiments may set ``validation_user_field=user_id`` and share one head.
    """

    field = str(settings.get("validation_user_field", "")).strip()
    if field:
        value = row.get(field)
        if value is None or not str(value).strip():
            raise ValueError(
                f"Validation candidate group {row.get('sample_id')} has no {field}"
            )
        return str(value)
    return str(row["sample_id"]).split(":", 1)[-1]


def adapt(config: dict) -> dict:
    ranker = config["ranker"]
    settings = config["user_adaptation"]
    output_dir = resolve_path(settings["output_dir"])
    heads_dir = output_dir / "user_heads"
    heads_dir.mkdir(parents=True, exist_ok=True)
    device = _device(ranker)
    seed = int(config["project"]["seed"])

    encoder, tokenizer, global_head, initialization_metadata = (
        _load_adaptation_components(ranker, settings, device, seed)
    )
    backbone_architecture = initialization_metadata["backbone_architecture"]
    hidden_size = int(encoder.config.hidden_size)
    initial_state = {
        key: value.detach().clone()
        for key, value in _build_user_scorer(
            str(settings.get("mode", "linear_head")),
            hidden_size,
            global_head,
            int(settings.get("adapter_rank", 16)),
        ).state_dict().items()
    }

    mode = str(settings.get("mode", "linear_head"))
    adapter_rank = int(settings.get("adapter_rank", 16))

    def scorer_factory():
        return _build_user_scorer(mode, hidden_size, global_head, adapter_rank)

    adaptation_rows = read_jsonl(resolve_path(settings["adaptation_source"]))
    rows_by_user: dict[str, list[dict]] = defaultdict(list)
    pseudo_counts: dict[str, int] = defaultdict(int)
    for row in adaptation_rows:
        user_id = str(row["parent_sample_id"])
        pseudo_counts[user_id] += 1
        rows_by_user[user_id].append(row)
    configured_m = settings["profiles_per_user"]
    use_all_queries = str(configured_m).strip().lower() == "all"
    expected_m = "all" if use_all_queries else int(configured_m)
    wrong_counts = (
        {}
        if use_all_queries
        else {user: count for user, count in pseudo_counts.items() if count != expected_m}
    )
    if wrong_counts:
        raise ValueError(f"Expected M={expected_m} pseudo queries per user, got {wrong_counts}")

    calibration_queries = int(settings.get("calibration_queries", 0))
    if calibration_queries < 0:
        raise ValueError("user_adaptation.calibration_queries must be non-negative")
    if not use_all_queries and not calibration_queries < expected_m:
        raise ValueError("user_adaptation.calibration_queries must be in [0, M)")
    adaptation_train_rows = []
    calibration_rows_by_user = {}
    for user_id, rows in rows_by_user.items():
        ordered = sorted(rows, key=lambda row: str(row["id"]))
        if calibration_queries:
            calibration_rows_by_user[user_id] = ordered[-calibration_queries:]
            ordered = ordered[:-calibration_queries]
        adaptation_train_rows.extend(ordered)
    metric = str(config["metric"]["primary"])
    pairs_by_user = _adaptation_pairs(adaptation_train_rows, settings, metric)
    validation_groups = read_jsonl(resolve_path(settings["validation_candidates"]))
    validation_users = {
        _validation_user_id(row, settings) for row in validation_groups
    }
    if set(pairs_by_user) != validation_users:
        raise ValueError(
            "Per-user adaptation IDs do not match validation users: "
            f"adaptation={sorted(pairs_by_user)}, validation={sorted(validation_users)}"
        )

    # 断点重跑时清除旧协议按 Query 保存的 Head，避免目录中同时出现两套身份键。
    expected_head_names = {f"user_{user_id}.pt" for user_id in pairs_by_user}
    stale_head_paths = [
        path
        for path in heads_dir.glob("user_*.pt")
        if path.name not in expected_head_names
    ]
    for path in stale_head_paths:
        path.unlink()

    reports = {}
    calibration_records = []
    for index, user_id in enumerate(sorted(pairs_by_user), start=1):
        pairs = pairs_by_user[user_id]
        features, weights = _encode_pairs(
            pairs,
            encoder,
            tokenizer,
            ranker,
            int(settings.get("encoding_batch_size", 32)),
            device,
        )
        state, report = fit_user_head(
            initial_state,
            features,
            weights,
            int(settings["epochs"]),
            float(settings["learning_rate"]),
            float(settings.get("weight_decay", 0.0)),
            float(settings.get("anchor_strength", 0.1)),
            int(settings.get("batch_size", 64)),
            float(settings.get("max_grad_norm", 1.0)),
            _user_seed(seed, user_id),
            scorer_factory=scorer_factory,
        )
        torch.save(state, heads_dir / f"user_{user_id}.pt")
        if calibration_queries:
            calibrated_scorer = scorer_factory()
            calibrated_scorer.load_state_dict(state)
            calibrated_scorer.eval()
            global_scorer = nn.Linear(hidden_size, 1)
            global_scorer.load_state_dict(global_head.state_dict())
            global_scorer.eval()
            for calibration_row in calibration_rows_by_user[user_id]:
                group_features, labels = _encode_calibration_group(
                    calibration_row,
                    encoder,
                    tokenizer,
                    ranker,
                    metric,
                    device,
                )
                with torch.no_grad():
                    calibration_records.append({
                        "user_id": user_id,
                        "global_scores": global_scorer(group_features).squeeze(-1),
                        "user_scores": calibrated_scorer(group_features).squeeze(-1),
                        "labels": labels,
                    })
        reports[user_id] = {
            "pseudo_queries": pseudo_counts[user_id],
            "adaptation_queries": pseudo_counts[user_id] - calibration_queries,
            "calibration_queries": calibration_queries,
            "mode": mode,
            **report,
        }
        print(
            f"user-head {index}/{len(pairs_by_user)} user={user_id} pairs={report['pairs']} "
            f"accuracy={report['initial_pair_accuracy']:.4f}->{report['final_pair_accuracy']:.4f}",
            flush=True,
        )

    alpha_grid = [
        float(value)
        for value in settings.get("blend_alpha_grid", [index / 10 for index in range(11)])
    ]
    if not alpha_grid or any(not 0.0 <= value <= 1.0 for value in alpha_grid):
        raise ValueError("user_adaptation.blend_alpha_grid must contain values in [0, 1]")
    alpha_scope = str(settings.get("alpha_scope", "shared")).lower()
    if alpha_scope not in {"shared", "per_user"}:
        raise ValueError("user_adaptation.alpha_scope must be shared or per_user")
    per_user_alphas: dict[str, float] = {}
    if calibration_queries and alpha_scope == "per_user":
        per_user_alphas, blend_results = _select_per_user_blend_alphas(
            calibration_records, alpha_grid
        )
        selected_alpha = sum(per_user_alphas.values()) / len(per_user_alphas)
    elif calibration_queries:
        selected_alpha, blend_results = _select_blend_alpha(calibration_records, alpha_grid)
    else:
        selected_alpha = float(settings.get("blend_alpha", 1.0))
        blend_results = []
    for user_id, alpha in per_user_alphas.items():
        reports[user_id]["selected_user_score_alpha"] = alpha
    initialization = initialization_metadata["initialization"]
    if initialization == "pretrained_reranker":
        protocol_prefix = "pretrained_reranker_direct_per_user"
    elif initialization == "pretrained":
        protocol_prefix = "pretrained_backbone_direct_per_user"
    else:
        protocol_prefix = "global_backbone_frozen_per_user"
    manifest = {
        "protocol": f"{protocol_prefix}_{mode}",
        "initialization": initialization,
        "base_model_name": initialization_metadata["base_model_name"],
        "global_model_dir": initialization_metadata["global_model_dir"],
        "adaptation_source": str(resolve_path(settings["adaptation_source"])),
        "validation_candidates": str(resolve_path(settings["validation_candidates"])),
        "users": len(reports),
        "head_key": str(settings.get("validation_user_field", "sample_id")),
        "stale_heads_removed": len(stale_head_paths),
        "profiles_per_user": expected_m,
        "adaptation_queries_per_user": (
            "all" if use_all_queries else expected_m - calibration_queries
        ),
        "calibration_queries_per_user": calibration_queries,
        "selected_user_score_alpha": selected_alpha,
        "alpha_scope": alpha_scope,
        "per_user_score_alpha": per_user_alphas,
        "blend_calibration": blend_results,
        "encoder_frozen": True,
        "backbone_architecture": backbone_architecture,
        "feature_pooling": "sequence_pooler"
        if backbone_architecture == "sequence_classifier"
        else "mean_pool",
        "adaptation_mode": mode,
        "adapter_rank": adapter_rank if mode == "adapter" else None,
        "head_parameters_per_user": sum(
            parameter.numel() for parameter in scorer_factory().parameters()
        ),
        "current_query_target_used_for_adaptation": False,
        "settings": {
            key: settings[key]
            for key in (
                "epochs",
                "learning_rate",
                "weight_decay",
                "anchor_strength",
                "pair_strategy",
                "pair_minimum_margin",
                "max_pairs_per_sample",
            )
        },
        "per_user": reports,
    }
    (output_dir / "adaptation_report.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


@torch.no_grad()
def predict(config: dict) -> list[dict]:
    ranker = config["ranker"]
    settings = config["user_adaptation"]
    output_dir = resolve_path(settings["output_dir"])
    heads_dir = output_dir / "user_heads"
    device = _device(ranker)
    seed = int(config["project"]["seed"])
    encoder, tokenizer, global_head, initialization_metadata = (
        _load_adaptation_components(ranker, settings, device, seed)
    )
    input_mode = str(ranker.get("input_mode", "factor_aware"))
    raw_text_input = bool(ranker.get("raw_text_input", False))
    mode = str(settings.get("mode", "linear_head"))
    adapter_rank = int(settings.get("adapter_rank", 16))
    groups = read_jsonl(resolve_path(settings["validation_candidates"]))
    adaptation_report = json.loads(
        (output_dir / "adaptation_report.json").read_text(encoding="utf-8")
    )
    default_user_alpha = float(
        adaptation_report.get("selected_user_score_alpha", 1.0)
    )
    per_user_alphas = {
        str(key): float(value)
        for key, value in adaptation_report.get("per_user_score_alpha", {}).items()
    }
    predictions = []
    for group in groups:
        user_id = _validation_user_id(group, settings)
        user_alpha = per_user_alphas.get(user_id, default_user_alpha)
        state_path = heads_dir / f"user_{user_id}.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing adapted head for validation user={user_id}")
        # Reconstruct the per-user scorer architecture from the adaptation
        # manifest/config; no user-specific backbone weights are loaded.
        scorer = _build_user_scorer(
            mode, int(encoder.config.hidden_size), global_head, adapter_rank
        ).to(device)
        scorer.load_state_dict(torch.load(state_path, map_location=device))
        scorer.eval()
        candidates = group["candidates"]
        # Reuse the global Ranker's encoder order/truncation policy.  The old
        # user-head path always used candidate-first + only-second truncation,
        # which silently disagreed with the context-first global model.
        encoded = _global_ranker._encode_pairs(
            tokenizer,
            [_candidate_text(candidate, input_mode, raw_text_input) for candidate in candidates],
            [
                _context_text(group, input_mode, raw_text_input=raw_text_input)
                for _ in candidates
            ],
            ranker,
            int(ranker["max_length"]),
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        features = _pooled(encoder, encoded)
        user_scores = scorer(features).squeeze(-1).float().cpu()
        initial_scores = global_head.to(device)(features).squeeze(-1).float().cpu()
        scores = ((1.0 - user_alpha) * initial_scores + user_alpha * user_scores).tolist()
        ranked = [
            {
                **candidate,
                "ranker_score": float(score_value),
                "initial_score": float(initial_score),
                "user_score": float(user_score),
                "user_score_alpha": user_alpha,
            }
            for candidate, score_value, initial_score, user_score in zip(
                candidates, scores, initial_scores.tolist(), user_scores.tolist()
            )
        ]
        ranked.sort(key=lambda candidate: (-candidate["ranker_score"], candidate["candidate_id"]))
        predictions.append(
            {
                "sample_id": group["sample_id"],
                "adaptation_user_id": user_id,
                "selected_id": ranked[0]["candidate_id"],
                "prediction": ranked[0]["text"],
                "ranked_candidates": ranked,
                "protocol": (
                    f"pretrained_reranker_direct_per_user_{mode}"
                    if initialization_metadata["initialization"]
                    == "pretrained_reranker"
                    else (
                        f"pretrained_backbone_direct_per_user_{mode}"
                        if initialization_metadata["initialization"] == "pretrained"
                        else f"global_backbone_frozen_per_user_{mode}"
                    )
                ),
                "user_score_alpha": user_alpha,
            }
        )
    destination = output_dir / "validation_predictions.jsonl"
    write_jsonl(destination, predictions)
    print(
        f"per-user head predictions for {len(predictions)} candidate groups "
        f"using {len({_validation_user_id(group, settings) for group in groups})} user heads "
        f"-> {destination}"
    )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="10 - Adapt a frozen-encoder Linear Head per user")
    parser.add_argument("--config", default=USER_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    adapt(config)


if __name__ == "__main__":
    main()
