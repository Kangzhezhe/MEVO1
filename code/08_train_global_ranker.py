"""阶段 08：训练 Global Ranker，并在固定验证候选池上输出预测。

默认是 pairwise 目标；通过配置可切换冻结编码器、解冻最后 N 层、listwise 或
hybrid 目标。训练完成的 encoder/head 和报告均写入当前 experiment 的 result。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from common.utils import load_config, read_jsonl, resolve_path, write_jsonl
from common.runtime import GLOBAL_CONFIG


# ---------- 输入文本构造 ----------
# candidate_only：仅候选；task_only：候选+abstract；factor_aware：再加入自然语言
# 因子方向；profile_aware 还加入目标隔离后的历史标题；provenance_aware
# 加入因子 ID、类型和候选来源（当前正式路线不用）。
def _context_text(
    row: dict,
    input_mode: str = "provenance_aware",
    training: bool = False,
    shuffle_factors: bool = False,
    factor_dropout_probability: float = 0.0,
    raw_text_input: bool = False,
) -> str:
    if input_mode == "candidate_only":
        return ""
    if input_mode == "task_only":
        return str(row["source_text"]) if raw_text_input else f"CURRENT ABSTRACT: {row['source_text']}"
    if input_mode not in {"factor_aware", "profile_aware", "provenance_aware"}:
        raise ValueError(f"Unknown ranker input_mode={input_mode}")

    factors = list(row.get("factors", []))
    if training and factor_dropout_probability > 0 and factors:
        kept = [factor for factor in factors if random.random() >= factor_dropout_probability]
        factors = kept or [random.choice(factors)]
    if training and shuffle_factors:
        random.shuffle(factors)
    if input_mode in {"factor_aware", "profile_aware"}:
        # The clean input exposes only natural-language preference directions;
        # factor IDs/types and candidate provenance are deliberately absent.
        factor_text = " ".join(str(factor["direction"]) for factor in factors)
    else:
        factor_text = " ".join(
            f"[{factor['factor_id']}|{factor['type']}] {factor['direction']}" for factor in factors
        )
    # Preferences precede the abstract so long-context truncation preserves
    # personalization instructions before trimming the abstract tail.
    profile_text = ""
    if input_mode == "profile_aware":
        titles = [str(title) for title in row.get("profile_titles", []) if str(title).strip()]
        profile_text = f" USER HISTORY TITLES: {' [TITLE] '.join(titles) or 'none'}"
    return (
        f"USER PREFERENCES: {factor_text or 'none'}{profile_text} "
        f"CURRENT ABSTRACT: {row['source_text']}"
    )


def _candidate_text(
    candidate: dict,
    input_mode: str = "provenance_aware",
    raw_text_input: bool = False,
) -> str:
    if raw_text_input:
        return str(candidate["text"])
    if input_mode != "provenance_aware":
        return f"CANDIDATE: {candidate['text']}"
    provenance = [f"type={candidate.get('type', 'unknown')}"]
    if candidate.get("factor_id"):
        provenance.append(f"factor={candidate['factor_id']}")
    if candidate.get("used_factors"):
        provenance.append(f"factors={','.join(map(str, candidate['used_factors']))}")
    return f"CANDIDATE ({'; '.join(provenance)}): {candidate['text']}"


def _encode_pairs(
    tokenizer,
    candidate_texts: list[str],
    context_texts: list[str],
    settings: dict,
    max_length: int,
):
    """Encode context/candidate pairs with an explicit cross-encoder order."""
    order = str(settings.get("input_order", "candidate_first"))
    if order == "context_first":
        first, second = context_texts, candidate_texts
        # Context can exceed 512 tokens after adding factors and abstract;
        # allow the tokenizer to trim the longer side instead of forbidding
        # truncation of the first sequence.
        truncation = "longest_first"
    elif order == "candidate_first":
        first, second = candidate_texts, context_texts
        truncation = "only_second"
    else:
        raise ValueError("ranker.input_order must be candidate_first or context_first")
    return tokenizer(
        first,
        second,
        padding=True,
        truncation=truncation,
        max_length=max_length,
        return_tensors="pt",
    )


# ---------- Pairwise 数据管道 ----------
class PairDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = read_jsonl(path)
        if not self.rows:
            raise ValueError(f"Pair dataset is empty: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class PairCollator:
    def __init__(self, tokenizer, max_length: int, settings: dict, training: bool):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.settings = settings
        self.input_mode = str(settings.get("input_mode", "provenance_aware"))
        self.raw_text_input = bool(settings.get("raw_text_input", False))
        self.training = training
        self.shuffle_factors = bool(settings.get("shuffle_factor_directions", False))
        self.factor_dropout_probability = float(settings.get("factor_dropout_probability", 0.0))
        if not 0.0 <= self.factor_dropout_probability < 1.0:
            raise ValueError("ranker.factor_dropout_probability must be in [0, 1)")

    def __call__(self, rows: list[dict]) -> dict:
        candidate_texts = []
        context_texts = []
        for row in rows:
            context = _context_text(
                row,
                self.input_mode,
                self.training,
                self.shuffle_factors,
                self.factor_dropout_probability,
                self.raw_text_input,
            )
            candidate_texts.extend(
                (
                    _candidate_text(row["chosen"], self.input_mode, self.raw_text_input),
                    _candidate_text(row["rejected"], self.input_mode, self.raw_text_input),
                )
            )
            context_texts.extend((context, context))
        encoded = _encode_pairs(
            self.tokenizer, candidate_texts, context_texts, self.settings, self.max_length
        )
        encoded["pair_count"] = len(rows)
        encoded["weights"] = torch.tensor(
            [1.0 + min(float(row.get("margin", 0.0)), 1.0) for row in rows], dtype=torch.float32
        )
        return encoded


# ---------- Listwise / Hybrid 数据管道 ----------
class GroupDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = read_jsonl(path)
        if not self.rows:
            raise ValueError(f"Listwise group dataset is empty: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class GroupCollator:
    def __init__(self, tokenizer, max_length: int, settings: dict, training: bool):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_mode = str(settings.get("input_mode", "provenance_aware"))
        self.raw_text_input = bool(settings.get("raw_text_input", False))
        self.training = training
        self.shuffle_factors = bool(settings.get("shuffle_factor_directions", False))
        self.factor_dropout_probability = float(settings.get("factor_dropout_probability", 0.0))
        self.settings = settings

    def __call__(self, rows: list[dict]) -> dict:
        candidate_texts = []
        context_texts = []
        group_sizes = []
        label_scores = []
        for row in rows:
            candidates = row["candidates"]
            context = _context_text(
                row,
                self.input_mode,
                self.training,
                self.shuffle_factors,
                self.factor_dropout_probability,
                self.raw_text_input,
            )
            group_sizes.append(len(candidates))
            label_scores.extend(row["label_scores"])
            candidate_texts.extend(
                _candidate_text(candidate, self.input_mode, self.raw_text_input)
                for candidate in candidates
            )
            context_texts.extend(context for _ in candidates)
        encoded = _encode_pairs(
            self.tokenizer, candidate_texts, context_texts, self.settings, self.max_length
        )
        encoded["group_sizes"] = group_sizes
        encoded["label_scores"] = torch.tensor(label_scores, dtype=torch.float32)
        return encoded


# ---------- 编码器与打分头 ----------
class CandidateRanker(nn.Module):
    def __init__(self, encoder: nn.Module, architecture: str = "encoder_pooling"):
        super().__init__()
        if architecture not in {"encoder_pooling", "sequence_classifier"}:
            raise ValueError(f"Unknown ranker model_architecture={architecture}")
        self.encoder = encoder
        self.architecture = architecture
        if architecture == "encoder_pooling":
            hidden_size = int(encoder.config.hidden_size)
            dropout = float(getattr(encoder.config, "hidden_dropout_prob", 0.1))
            self.dropout = nn.Dropout(dropout)
            self.scorer = nn.Linear(hidden_size, 1)
        else:
            self.dropout = None
            self.scorer = None

    def forward(self, **encoded) -> torch.Tensor:
        outputs = self.encoder(**encoded)
        if self.architecture == "sequence_classifier":
            if outputs.logits.shape[-1] != 1:
                raise ValueError("sequence_classifier Ranker requires one output logit")
            return outputs.logits.squeeze(-1)
        hidden = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1.0)
        return self.scorer(self.dropout(pooled)).squeeze(-1)


def _transformer_layers(encoder: nn.Module):
    base_model = getattr(encoder, "base_model", encoder)
    layers = getattr(getattr(base_model, "encoder", None), "layer", None)
    if layers is None:
        layers = getattr(getattr(base_model, "transformer", None), "layer", None)
    return layers


def _configure_encoder_trainability(
    encoder: nn.Module,
    setting,
    keep_task_head_trainable: bool = False,
) -> dict:
    """冻结主干后仅解冻最后 N 层；N=0 表示完全冻结，'all' 表示全量微调。"""
    normalized = str(setting).lower()
    train_all = normalized == "all" or (
        normalized.lstrip("-").isdigit() and int(normalized) < 0
    )
    if train_all:
        for parameter in encoder.parameters():
            parameter.requires_grad = True
        mode = "all"
    else:
        trainable_layers = int(setting)
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        if trainable_layers > 0:
            layers = _transformer_layers(encoder)
            if layers is None:
                raise ValueError("Cannot locate transformer layers for partial encoder unfreezing")
            if trainable_layers > len(layers):
                raise ValueError(
                    f"Requested {trainable_layers} trainable layers, but encoder has only {len(layers)}"
                )
            for layer in layers[-trainable_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        mode = f"last_{trainable_layers}" if trainable_layers else "frozen"
    if keep_task_head_trainable:
        base_prefix = str(getattr(encoder, "base_model_prefix", ""))
        prefix = f"{base_prefix}." if base_prefix else ""
        for name, parameter in encoder.named_parameters():
            if not name.startswith(prefix):
                parameter.requires_grad = True
    total = sum(parameter.numel() for parameter in encoder.parameters())
    trainable = sum(parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad)
    return {"mode": mode, "encoder_trainable_parameters": trainable, "encoder_total_parameters": total}


def _move(batch: dict, device: torch.device) -> tuple[dict, int, torch.Tensor]:
    pair_count = int(batch.pop("pair_count"))
    weights = batch.pop("weights").to(device)
    return {key: value.to(device) for key, value in batch.items()}, pair_count, weights


@torch.no_grad()
# ---------- 损失函数和验证指标 ----------
def _pair_accuracy(model: CandidateRanker, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for batch in loader:
        encoded, pair_count, weights = _move(batch, device)
        scores = model(**encoded).view(pair_count, 2)
        deltas = scores[:, 0] - scores[:, 1]
        losses = F.softplus(-deltas) * weights
        loss_sum += float(losses.sum().item())
        correct += int((deltas > 0).sum().item())
        total += pair_count
    return correct / max(total, 1), loss_sum / max(total, 1)


def _group_batch(batch: dict, device: torch.device) -> tuple[dict, list[int], torch.Tensor]:
    group_sizes = list(batch.pop("group_sizes"))
    labels = batch.pop("label_scores").to(device)
    return {key: value.to(device) for key, value in batch.items()}, group_sizes, labels


def _split_groups(scores: torch.Tensor, group_sizes: list[int]) -> list[torch.Tensor]:
    return list(torch.split(scores, group_sizes))


def _listwise_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: list[int],
    label_temperature: float,
    score_temperature: float = 1.0,
) -> torch.Tensor:
    if label_temperature <= 0 or score_temperature <= 0:
        raise ValueError("Listwise temperatures must be positive")
    losses = []
    for score_group, label_group in zip(_split_groups(scores, group_sizes), _split_groups(labels, group_sizes)):
        target = F.softmax(label_group / label_temperature, dim=0)
        log_prediction = F.log_softmax(score_group / score_temperature, dim=0)
        losses.append(F.kl_div(log_prediction, target, reduction="sum"))
    return torch.stack(losses).mean()


def _listmle_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: list[int],
    score_temperature: float = 1.0,
    tie_tolerance: float = 1.0e-8,
) -> torch.Tensor:
    """Tie-aware Plackett-Luce loss over descending ROUGE utility levels.

    Candidates with the same label form one relevance tier. At each step the
    loss rewards assigning probability mass to the entire best remaining tier,
    so equal-ROUGE candidates never receive an arbitrary internal ordering.
    Every query contributes one normalized loss regardless of candidate count.
    """
    if score_temperature <= 0:
        raise ValueError("ranker.score_temperature must be positive")
    losses = []
    for score_group, label_group in zip(
        _split_groups(scores, group_sizes), _split_groups(labels, group_sizes)
    ):
        remaining = torch.ones_like(label_group, dtype=torch.bool)
        tier_losses = []
        while bool(remaining.any()):
            remaining_labels = label_group[remaining]
            best_label = remaining_labels.max()
            current_tier = remaining & (label_group >= best_label - tie_tolerance)
            scaled = score_group / score_temperature
            tier_losses.append(
                torch.logsumexp(scaled[remaining], dim=0)
                - torch.logsumexp(scaled[current_tier], dim=0)
            )
            remaining = remaining & ~current_tier
        losses.append(torch.stack(tier_losses).mean())
    return torch.stack(losses).mean()


def _pointwise_score_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: list[int],
) -> torch.Tensor:
    """Preserve each candidate's continuous ROUGE utility, not only order."""
    losses = []
    for score_group, label_group in zip(
        _split_groups(scores, group_sizes), _split_groups(labels, group_sizes)
    ):
        losses.append(F.smooth_l1_loss(torch.sigmoid(score_group), label_group))
    return torch.stack(losses).mean()


def _expected_regret_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: list[int],
    temperature: float = 1.0,
) -> torch.Tensor:
    """Minimize differentiable expected Top-1 ROUGE regret per query."""
    if temperature <= 0:
        raise ValueError("ranker top1 temperature must be positive")
    losses = []
    for score_group, label_group in zip(
        _split_groups(scores, group_sizes), _split_groups(labels, group_sizes)
    ):
        probabilities = F.softmax(score_group / temperature, dim=0)
        losses.append(label_group.max() - (probabilities * label_group).sum())
    return torch.stack(losses).mean()


def _top1_lambda_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: list[int],
) -> torch.Tensor:
    """Regret-weighted comparisons involving the model's current Top-1.

    Lower-rank pairs receive no gradient.  The selected candidate is detached
    only for choosing comparisons; score differences remain differentiable.
    """
    losses = []
    for score_group, label_group in zip(
        _split_groups(scores, group_sizes), _split_groups(labels, group_sizes)
    ):
        top_index = int(score_group.detach().argmax().item())
        top_score = score_group[top_index]
        top_label = label_group[top_index]
        pair_losses = []
        for index in range(len(score_group)):
            if index == top_index:
                continue
            gap = label_group[index] - top_label
            if abs(float(gap.item())) <= 1.0e-8:
                continue
            preferred_delta = score_group[index] - top_score if gap > 0 else top_score - score_group[index]
            pair_losses.append(F.softplus(-preferred_delta) * gap.abs())
        if pair_losses:
            losses.append(torch.stack(pair_losses).mean())
    return torch.stack(losses).mean() if losses else scores.sum() * 0.0


def _oracle_set_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: list[int],
    score_temperature: float = 1.0,
    tie_tolerance: float = 1.0e-8,
) -> torch.Tensor:
    """Put probability mass on any gold-tied oracle candidate.

    KL/ListNet supervises the entire ranking and can remain diffuse when ten
    candidates have similar ROUGE.  This auxiliary objective is deliberately
    top-heavy: it minimizes the negative log probability assigned to the set
    of candidates tied for the best label, without imposing an arbitrary
    order inside that set.
    """
    if score_temperature <= 0:
        raise ValueError("ranker.score_temperature must be positive")
    losses = []
    for score_group, label_group in zip(
        _split_groups(scores, group_sizes), _split_groups(labels, group_sizes)
    ):
        scaled = score_group / score_temperature
        oracle_mask = label_group >= label_group.max() - tie_tolerance
        losses.append(torch.logsumexp(scaled, dim=0) - torch.logsumexp(scaled[oracle_mask], dim=0))
    return torch.stack(losses).mean()


def _group_pairwise_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: list[int],
    minimum_margin: float,
    pair_strategy: str = "all_pairs",
    max_pairs_per_group: int = 0,
    margin_weight_scale: float = 1.0,
) -> torch.Tensor:
    """Compute the pairwise component inside each listwise group.

    ``hard_negative`` is selected dynamically from the current model scores:
    pairs with the smallest absolute score gap are the most uncertain ones.
    ``top_pairs`` instead keeps only gold-oracle versus non-oracle comparisons
    and prioritizes the nearest label gaps, directly matching Top-1 regret.
    """
    group_losses = []
    for score_group, label_group in zip(_split_groups(scores, group_sizes), _split_groups(labels, group_sizes)):
        candidates = []
        for left in range(len(score_group)):
            for right in range(left + 1, len(score_group)):
                gap = float(label_group[left] - label_group[right])
                if abs(gap) < minimum_margin:
                    continue
                delta = score_group[left] - score_group[right]
                candidates.append((left, right, gap, delta))
        if pair_strategy == "hard_negative":
            # Use detached scores only to choose the examples; gradients still
            # flow through the selected pair loss below.
            candidates.sort(
                key=lambda item: (
                    abs(float((item[3]).detach().item())),
                    abs(item[2]),
                    item[0],
                    item[1],
                )
            )
        elif pair_strategy == "top_pairs":
            oracle_label = float(label_group.max().item())
            candidates = [
                item
                for item in candidates
                if max(
                    float(label_group[item[0]].item()),
                    float(label_group[item[1]].item()),
                ) >= oracle_label - 1.0e-8
            ]
            candidates.sort(
                key=lambda item: (
                    abs(item[2]),
                    abs(float(item[3].detach().item())),
                    item[0],
                    item[1],
                )
            )
        if max_pairs_per_group > 0:
            candidates = candidates[:max_pairs_per_group]
        losses = []
        for _, _, gap, delta in candidates:
            if gap < 0:
                delta = -delta
            losses.append(
                F.softplus(-delta)
                * (1.0 + margin_weight_scale * min(abs(gap), 1.0))
            )
        if losses:
            # Every query is one independent training example.  Normalize
            # within the group so a query with more non-tied pairs does not
            # silently dominate another query.
            group_losses.append(torch.stack(losses).mean())
    if not group_losses:
        return scores.sum() * 0.0
    return torch.stack(group_losses).mean()


@torch.no_grad()
def _group_metrics(
    model: CandidateRanker,
    loader: DataLoader,
    device: torch.device,
    label_temperature: float,
) -> tuple[float, float, float]:
    model.eval()
    group_correct = 0
    group_count = 0
    loss_sum = 0.0
    regret_sum = 0.0
    for batch in loader:
        encoded, group_sizes, labels = _group_batch(batch, device)
        scores = model(**encoded)
        loss = _listwise_loss(scores, labels, group_sizes, label_temperature)
        loss_sum += float(loss.item()) * len(group_sizes)
        for score_group, label_group in zip(_split_groups(scores, group_sizes), _split_groups(labels, group_sizes)):
            group_correct += int(score_group.argmax().item() == label_group.argmax().item())
            regret_sum += float(label_group.max().item() - label_group[score_group.argmax()].item())
            group_count += 1
    return (
        group_correct / max(group_count, 1),
        loss_sum / max(group_count, 1),
        regret_sum / max(group_count, 1),
    )


def _checkpoint_selection_key(
    objective: str,
    requested: str,
    validation_accuracy: float,
    validation_loss: float,
    group_accuracy: float,
    group_loss: float,
    group_regret: float,
) -> tuple[tuple[float, ...], str]:
    """Return a maximization key for checkpoint selection.

    ``auto`` selects pair accuracy for a pure pairwise objective and mean
    regret for listwise/hybrid objectives.  Group accuracy is a useful
    diagnostic, but its single-argmax treatment of tied ROUGE labels is too
    coarse to be the default checkpoint key.
    """
    selection = str(requested or "auto").lower()
    if selection == "auto":
        selection = "pair_accuracy" if objective == "pairwise" else "mean_regret"
    if selection == "pair_accuracy":
        return (validation_accuracy, -validation_loss), "pair accuracy"
    if selection == "group_accuracy":
        return (group_accuracy, -group_regret, -group_loss), "group accuracy/regret"
    if selection == "mean_regret":
        return (-group_regret, group_accuracy, -group_loss), "mean regret"
    if selection == "listwise_kl":
        return (-group_loss, -group_regret, group_accuracy), "listwise KL"
    raise ValueError(
        "ranker.checkpoint_selection must be auto, pair_accuracy, "
        "group_accuracy, mean_regret, or listwise_kl"
    )


def _build_model(model_name: str, architecture: str, local_files_only: bool) -> CandidateRanker:
    if architecture == "sequence_classifier":
        encoder = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            local_files_only=local_files_only,
        )
    else:
        encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    return CandidateRanker(encoder, architecture)


def _save_model(model: CandidateRanker, tokenizer, output_dir: Path, metadata: dict) -> None:
    """保存所有可训练参数，以及足以恢复基础模型结构的 metadata。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    if metadata.get("checkpoint_format") == "trainable_delta":
        torch.save(
            {
                name: parameter.detach().cpu()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            },
            output_dir / "model_delta.pt",
        )
        metadata["base_model_name"] = metadata["model_name"]
    else:
        model.encoder.save_pretrained(output_dir / "encoder")
        tokenizer.save_pretrained(output_dir / "encoder")
        if model.scorer is not None:
            torch.save(model.scorer.state_dict(), output_dir / "ranker_head.pt")
    (output_dir / "training_report.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def train(config: dict, data_dir: Path, output_dir: Path) -> dict:
    """按 objective 选择 pair/group 数据，训练并用 early stopping 保留最佳轮次。"""
    settings = config["ranker"]
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    protocol = str(manifest.get("protocol", "grouped_ranker_pilot"))
    protocol_warning = str(manifest.get("warning", ""))
    seed = int(config["project"]["seed"])
    deterministic = bool(settings.get("deterministic", False))
    deterministic_warn_only = bool(settings.get("deterministic_warn_only", False))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=deterministic_warn_only)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    device_name = str(settings.get("device", "cuda"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("ranker warning: CUDA unavailable; falling back to CPU", flush=True)
        device_name = "cpu"
    device = torch.device(device_name)
    model_name = str(settings["model_name"])
    architecture = str(settings.get("model_architecture", "encoder_pooling"))
    local_files_only = bool(settings.get("local_files_only", True))
    initial_model_dir = settings.get("initial_model_dir")
    if initial_model_dir:
        initial_path = resolve_path(initial_model_dir)
        initial_report = json.loads(
            (initial_path / "training_report.json").read_text(encoding="utf-8")
        )
        if str(initial_report["model_name"]) != model_name:
            raise ValueError(
                "ranker.initial_model_dir model does not match ranker.model_name: "
                f"{initial_report['model_name']} != {model_name}"
            )
        if str(initial_report.get("model_architecture", "encoder_pooling")) != architecture:
            raise ValueError("ranker.initial_model_dir architecture does not match current config")
        model, tokenizer = _load_model(initial_path, device)
        print(f"ranker initialization: {initial_path}", flush=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        model = _build_model(model_name, architecture, local_files_only)
    trainability = _configure_encoder_trainability(
        model.encoder,
        settings.get("trainable_encoder_layers", "all"),
        keep_task_head_trainable=architecture == "sequence_classifier",
    )
    gradient_checkpointing = bool(settings.get("gradient_checkpointing", False))
    if gradient_checkpointing:
        enable_checkpointing = getattr(model.encoder, "gradient_checkpointing_enable", None)
        if not callable(enable_checkpointing):
            raise ValueError(
                f"ranker model {model_name} does not support gradient checkpointing"
            )
        enable_checkpointing()
    model.to(device)

    objective = str(settings.get("objective", "pairwise")).lower()
    if objective not in {"pairwise", "listwise", "hybrid"}:
        raise ValueError("ranker.objective must be pairwise, listwise, or hybrid")
    train_collator = PairCollator(tokenizer, int(settings["max_length"]), settings, training=True)
    validation_collator = PairCollator(tokenizer, int(settings["max_length"]), settings, training=False)
    group_train_collator = GroupCollator(tokenizer, int(settings["max_length"]), settings, training=True)
    group_validation_collator = GroupCollator(tokenizer, int(settings["max_length"]), settings, training=False)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        PairDataset(data_dir / "train_pairs.jsonl"),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        collate_fn=train_collator,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        PairDataset(data_dir / "validation_pairs.jsonl"),
        batch_size=int(settings["inference_batch_size"]),
        shuffle=False,
        collate_fn=validation_collator,
        num_workers=0,
    )
    group_train_loader = DataLoader(
        GroupDataset(data_dir / "train_listwise.jsonl"),
        batch_size=int(settings.get("group_batch_size", 2)),
        shuffle=True,
        collate_fn=group_train_collator,
        generator=generator,
        num_workers=0,
    )
    group_validation_loader = DataLoader(
        GroupDataset(data_dir / "validation_listwise.jsonl"),
        batch_size=int(settings.get("group_inference_batch_size", 2)),
        shuffle=False,
        collate_fn=group_validation_collator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    epochs = int(settings["epochs"])
    active_loader_length = len(train_loader) if objective == "pairwise" else len(group_train_loader)
    gradient_accumulation_steps = max(
        1, int(settings.get("gradient_accumulation_steps", 1))
    )
    optimizer_steps_per_epoch = (
        active_loader_length + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    total_steps = max(1, epochs * optimizer_steps_per_epoch)
    warmup_steps = round(total_steps * float(settings["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    mixed_precision = str(settings.get("mixed_precision", "fp16")).lower()
    if mixed_precision not in {"none", "fp16", "bf16"}:
        raise ValueError("ranker.mixed_precision must be none, fp16, or bf16")
    use_amp = device.type == "cuda" and mixed_precision != "none"
    amp_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp and mixed_precision == "fp16"
    )
    history = []
    best_selection = None
    best_epoch = 0
    stale_epochs = 0
    patience = int(settings.get("early_stopping_patience", 0))
    label_temperature = float(settings.get("listwise_temperature", 0.1))
    score_temperature = float(settings.get("score_temperature", 1.0))
    pairwise_weight = float(settings.get("pairwise_weight", 0.3))
    listwise_weight = float(settings.get("listwise_weight", 1.0))
    listmle_weight = float(settings.get("listmle_weight", 0.0))
    pointwise_weight = float(settings.get("pointwise_weight", 0.0))
    expected_regret_weight = float(settings.get("expected_regret_weight", 0.0))
    lambda_top1_weight = float(settings.get("lambda_top1_weight", 0.0))
    pair_margin_weight_scale = float(settings.get("pair_margin_weight_scale", 1.0))
    oracle_set_weight = float(settings.get("oracle_set_weight", 0.0))
    oracle_tie_tolerance = float(settings.get("oracle_tie_tolerance", 1.0e-8))
    temperature_start = float(settings.get("top1_temperature_start", score_temperature))
    temperature_end = float(settings.get("top1_temperature_end", temperature_start))
    if min(
        oracle_set_weight,
        pointwise_weight,
        pair_margin_weight_scale,
        listwise_weight,
        listmle_weight,
        expected_regret_weight,
        lambda_top1_weight,
        oracle_tie_tolerance,
        temperature_start,
        temperature_end,
    ) < 0 or temperature_start == 0 or temperature_end == 0:
        raise ValueError("Ranker auxiliary loss weights must be non-negative")
    checkpoint_selection = str(settings.get("checkpoint_selection", "auto")).lower()

    print(
        f"ranker train: model={model_name}, device={device}, train_pairs={len(train_loader.dataset)}, "
        f"validation_pairs={len(validation_loader.dataset)}, epochs={epochs}, "
        f"trainability={trainability['mode']}, objective={objective}, "
        f"pair_strategy={settings.get('pair_strategy', 'all_pairs')}, "
        f"deterministic={deterministic}, mixed_precision={mixed_precision}, "
        f"gradient_accumulation={gradient_accumulation_steps}, "
        f"gradient_checkpointing={gradient_checkpointing}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        temperature_progress = (epoch - 1) / max(epochs - 1, 1)
        top1_temperature = (
            temperature_start
            + temperature_progress * (temperature_end - temperature_start)
        )
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        if objective == "pairwise":
            optimizer.zero_grad(set_to_none=True)
            for batch_index, batch in enumerate(train_loader):
                encoded, pair_count, weights = _move(batch, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    scores = model(**encoded).view(pair_count, 2)
                    losses = F.softplus(-(scores[:, 0] - scores[:, 1])) * weights
                    loss = losses.mean()
                window_start = (
                    batch_index // gradient_accumulation_steps
                ) * gradient_accumulation_steps
                window_size = min(
                    gradient_accumulation_steps, len(train_loader) - window_start
                )
                scaler.scale(loss / window_size).backward()
                should_step = (
                    (batch_index + 1) % gradient_accumulation_steps == 0
                    or batch_index + 1 == len(train_loader)
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(settings["max_grad_norm"])
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                train_loss_sum += float(losses.detach().sum().item())
                train_count += pair_count
        else:
            optimizer.zero_grad(set_to_none=True)
            for batch_index, batch in enumerate(group_train_loader):
                encoded, group_sizes, labels = _group_batch(batch, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    scores = model(**encoded)
                    list_loss = _listwise_loss(
                        scores,
                        labels,
                        group_sizes,
                        label_temperature,
                        top1_temperature,
                    )
                    listmle_loss = (
                        _listmle_loss(
                            scores,
                            labels,
                            group_sizes,
                            top1_temperature,
                            oracle_tie_tolerance,
                        )
                        if listmle_weight > 0
                        else scores.sum() * 0.0
                    )
                    oracle_loss = (
                        _oracle_set_loss(
                            scores,
                            labels,
                            group_sizes,
                            top1_temperature,
                            oracle_tie_tolerance,
                        )
                        if oracle_set_weight > 0
                        else scores.sum() * 0.0
                    )
                    point_loss = (
                        _pointwise_score_loss(scores, labels, group_sizes)
                        if pointwise_weight > 0
                        else scores.sum() * 0.0
                    )
                    regret_loss = (
                        _expected_regret_loss(
                            scores, labels, group_sizes, top1_temperature
                        )
                        if expected_regret_weight > 0
                        else scores.sum() * 0.0
                    )
                    lambda_loss = (
                        _top1_lambda_loss(scores, labels, group_sizes)
                        if lambda_top1_weight > 0
                        else scores.sum() * 0.0
                    )
                    if objective == "hybrid":
                        pair_loss = _group_pairwise_loss(
                            scores,
                            labels,
                            group_sizes,
                            float(settings.get("pair_minimum_margin", 0.02)),
                            str(settings.get("pair_strategy", "all_pairs")),
                            int(settings.get("max_pairs_per_sample", 0)),
                            pair_margin_weight_scale,
                        )
                        loss = (
                            listwise_weight * list_loss
                            + listmle_weight * listmle_loss
                            + pairwise_weight * pair_loss
                            + oracle_set_weight * oracle_loss
                            + pointwise_weight * point_loss
                            + expected_regret_weight * regret_loss
                            + lambda_top1_weight * lambda_loss
                        )
                    else:
                        loss = (
                            listwise_weight * list_loss
                            + listmle_weight * listmle_loss
                            + oracle_set_weight * oracle_loss
                            + pointwise_weight * point_loss
                            + expected_regret_weight * regret_loss
                            + lambda_top1_weight * lambda_loss
                        )
                window_start = (
                    batch_index // gradient_accumulation_steps
                ) * gradient_accumulation_steps
                window_size = min(
                    gradient_accumulation_steps,
                    len(group_train_loader) - window_start,
                )
                scaler.scale(loss / window_size).backward()
                should_step = (
                    (batch_index + 1) % gradient_accumulation_steps == 0
                    or batch_index + 1 == len(group_train_loader)
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(settings["max_grad_norm"])
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                train_loss_sum += float(loss.detach().item()) * len(group_sizes)
                train_count += len(group_sizes)

        validation_accuracy, validation_loss = _pair_accuracy(model, validation_loader, device)
        group_accuracy, group_loss, group_regret = _group_metrics(
            model, group_validation_loader, device, label_temperature
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_count, 1),
            "validation_loss": validation_loss,
            "validation_pair_accuracy": validation_accuracy,
            "validation_group_accuracy": group_accuracy,
            "validation_listwise_kl": group_loss,
            "validation_mean_regret": group_regret,
            "top1_temperature": top1_temperature,
        }
        history.append(epoch_metrics)
        print(f"ranker epoch {epoch}/{epochs}: {epoch_metrics}", flush=True)
        selection_key, selection_name = _checkpoint_selection_key(
            objective,
            checkpoint_selection,
            validation_accuracy,
            validation_loss,
            group_accuracy,
            group_loss,
            group_regret,
        )
        if best_selection is None or selection_key > best_selection:
            best_selection = selection_key
            best_epoch = epoch
            stale_epochs = 0
            _save_model(
                model,
                tokenizer,
                output_dir,
                {
                    "protocol": protocol,
                    "model_name": model_name,
                    "model_architecture": architecture,
                    "seed": seed,
                    "trainability": trainability,
                    "input_mode": str(settings.get("input_mode", "provenance_aware")),
                    "input_order": str(settings.get("input_order", "candidate_first")),
                    "shuffle_factor_directions": bool(
                        settings.get("shuffle_factor_directions", False)
                    ),
                    "factor_dropout_probability": float(
                        settings.get("factor_dropout_probability", 0.0)
                    ),
                    "deterministic": deterministic,
                    "deterministic_warn_only": deterministic_warn_only,
                    "mixed_precision": mixed_precision,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "gradient_checkpointing": gradient_checkpointing,
                    "best_epoch": epoch,
                    "best_validation_pair_accuracy": validation_accuracy,
                    "best_validation_group_accuracy": group_accuracy,
                    "best_validation_mean_regret": group_regret,
                    "best_validation_listwise_kl": group_loss,
                    "checkpoint_selection": checkpoint_selection,
                    "objective": objective,
                    "listwise_temperature": label_temperature,
                    "score_temperature": score_temperature,
                    "pairwise_weight": pairwise_weight,
                    "listwise_weight": listwise_weight,
                    "listmle_weight": listmle_weight,
                    "pointwise_weight": pointwise_weight,
                    "expected_regret_weight": expected_regret_weight,
                    "lambda_top1_weight": lambda_top1_weight,
                    "oracle_tie_tolerance": oracle_tie_tolerance,
                    "top1_temperature_start": temperature_start,
                    "top1_temperature_end": temperature_end,
                    "pair_margin_weight_scale": pair_margin_weight_scale,
                    "oracle_set_weight": oracle_set_weight,
                    "pair_strategy": str(settings.get("pair_strategy", "all_pairs")),
                    "max_pairs_per_sample": int(settings.get("max_pairs_per_sample", 0)),
                    "checkpoint_format": str(settings.get("checkpoint_format", "full")),
                    "initial_model_dir": str(initial_model_dir or ""),
                    "history": history,
                    "warning": protocol_warning,
                },
            )
        else:
            stale_epochs += 1
            if patience > 0 and stale_epochs >= patience:
                print(
                    f"ranker early stop at epoch={epoch}; no {selection_name} improvement "
                    f"for {patience} epochs",
                    flush=True,
                )
                break
    selected_metrics = next(
        (item for item in history if item["epoch"] == best_epoch), {}
    )
    final_report = {
        "protocol": protocol,
        "model_name": model_name,
        "model_architecture": architecture,
        "seed": seed,
        "trainability": trainability,
        "input_mode": str(settings.get("input_mode", "provenance_aware")),
        "input_order": str(settings.get("input_order", "candidate_first")),
        "shuffle_factor_directions": bool(settings.get("shuffle_factor_directions", False)),
        "factor_dropout_probability": float(settings.get("factor_dropout_probability", 0.0)),
        "deterministic": deterministic,
        "deterministic_warn_only": deterministic_warn_only,
        "mixed_precision": mixed_precision,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": gradient_checkpointing,
        "checkpoint_selection": checkpoint_selection,
        "best_epoch": best_epoch,
        "best_validation_pair_accuracy": selected_metrics.get(
            "validation_pair_accuracy", 0.0
        ),
        "best_validation_group_accuracy": selected_metrics.get(
            "validation_group_accuracy", 0.0
        ),
        "best_validation_mean_regret": selected_metrics.get(
            "validation_mean_regret", 0.0
        ),
        "best_validation_listwise_kl": selected_metrics.get(
            "validation_listwise_kl", 0.0
        ),
        "objective": objective,
        "listwise_temperature": label_temperature,
        "score_temperature": score_temperature,
        "pairwise_weight": pairwise_weight,
        "listwise_weight": listwise_weight,
        "listmle_weight": listmle_weight,
        "pointwise_weight": pointwise_weight,
        "expected_regret_weight": expected_regret_weight,
        "lambda_top1_weight": lambda_top1_weight,
        "oracle_tie_tolerance": oracle_tie_tolerance,
        "top1_temperature_start": temperature_start,
        "top1_temperature_end": temperature_end,
        "pair_margin_weight_scale": pair_margin_weight_scale,
        "oracle_set_weight": oracle_set_weight,
        "pair_strategy": str(settings.get("pair_strategy", "all_pairs")),
        "max_pairs_per_sample": int(settings.get("max_pairs_per_sample", 0)),
        "checkpoint_format": str(settings.get("checkpoint_format", "full")),
        "initial_model_dir": str(initial_model_dir or ""),
        "history": history,
        "warning": protocol_warning,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return final_report


def _load_model(output_dir: Path, device: torch.device) -> tuple[CandidateRanker, object]:
    """从 result 目录恢复 encoder 和 head，供独立预测/评估使用。"""
    report = json.loads((output_dir / "training_report.json").read_text(encoding="utf-8"))
    architecture = str(report.get("model_architecture", "encoder_pooling"))
    if (output_dir / "model_delta.pt").exists():
        model_name = str(report["model_name"])
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = _build_model(model_name, architecture, local_files_only=True)
    elif (output_dir / "encoder_delta.pt").exists():
        # Compatibility with early Contriever-only checkpoints.
        model_name = str(report["model_name"])
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = _build_model(model_name, "encoder_pooling", local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(output_dir / "encoder", local_files_only=True)
        if architecture == "sequence_classifier":
            encoder = AutoModelForSequenceClassification.from_pretrained(
                output_dir / "encoder", local_files_only=True
            )
        else:
            encoder = AutoModel.from_pretrained(output_dir / "encoder", local_files_only=True)
        model = CandidateRanker(encoder, architecture)
    model_delta_path = output_dir / "model_delta.pt"
    encoder_delta_path = output_dir / "encoder_delta.pt"
    if model_delta_path.exists():
        model.load_state_dict(torch.load(model_delta_path, map_location="cpu"), strict=False)
    elif encoder_delta_path.exists():
        model.encoder.load_state_dict(
            torch.load(encoder_delta_path, map_location="cpu"), strict=False
        )
    head_path = output_dir / "ranker_head.pt"
    if head_path.exists() and model.scorer is not None:
        model.scorer.load_state_dict(torch.load(head_path, map_location="cpu"))
    model.to(device).eval()
    return model, tokenizer


@torch.no_grad()
def predict(config: dict, data_dir: Path, output_dir: Path, split: str, destination: Path) -> None:
    """逐候选打分并选择 Top-1；预测阶段不读取 label_scores 或 target。"""
    settings = config["ranker"]
    device_name = str(settings.get("device", "cuda"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    model, tokenizer = _load_model(output_dir, device)
    input_mode = str(settings.get("input_mode", "provenance_aware"))
    raw_text_input = bool(settings.get("raw_text_input", False))
    groups = read_jsonl(data_dir / f"{split}_candidates.jsonl")
    flat = [(group, candidate) for group in groups for candidate in group["candidates"]]
    batch_size = int(settings["inference_batch_size"])
    scores = []
    for start in range(0, len(flat), batch_size):
        batch = flat[start : start + batch_size]
        encoded = _encode_pairs(
            tokenizer,
            [_candidate_text(candidate, input_mode, raw_text_input) for _, candidate in batch],
            [_context_text(group, input_mode, raw_text_input=raw_text_input) for group, _ in batch],
            settings,
            int(settings["max_length"]),
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        scores.extend(model(**encoded).float().cpu().tolist())

    cursor = 0
    predictions = []
    for group in groups:
        ranked = []
        for candidate in group["candidates"]:
            ranked.append({**candidate, "ranker_score": float(scores[cursor])})
            cursor += 1
        ranked.sort(key=lambda candidate: (-candidate["ranker_score"], candidate["candidate_id"]))
        predictions.append(
            {
                "sample_id": group["sample_id"],
                "selected_id": ranked[0]["candidate_id"],
                "prediction": ranked[0]["text"],
                "ranked_candidates": ranked,
                "protocol": "target_blind_pairwise_ranker",
            }
        )
    write_jsonl(destination, predictions)
    print(f"ranker predictions for {len(predictions)} samples -> {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="08 - Train the global pairwise candidate Ranker")
    parser.add_argument("--config", default=GLOBAL_CONFIG)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["ranker"]
    data_dir = args.data_dir or resolve_path(settings["data_dir"])
    output_dir = args.output_dir or resolve_path(settings["output_dir"])
    train(config, data_dir, output_dir)


if __name__ == "__main__":
    main()
