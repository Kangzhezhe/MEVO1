"""阶段 20：以阶段一 SFT Adapter 为 reference，训练每用户 Editor IDPO Adapter。"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from pipeline_common import load_config, read_jsonl, resolve_path, write_json  # noqa: E402
from idpo_common import group_pairs_by_user, idpo_path  # noqa: E402


def _truncate_prompt(ids: list[int], maximum: int) -> list[int]:
    if len(ids) <= maximum:
        return ids
    head = max(1, maximum // 2)
    return ids[:head] + ids[-(maximum - head) :]


def _sequence_batch(
    pairs: list[dict[str, Any]],
    tokenizer,
    response_key: str,
    max_length: int,
    response_weighting: str = "sequence",
    trace_weight: float = 0.2,
    output_weight: float = 1.0,
) -> dict[str, Any]:
    sequences: list[list[int]] = []
    response_masks: list[list[int]] = []
    score_weights: list[list[float]] = []
    for pair in pairs:
        prompt_ids = tokenizer.encode(str(pair["prompt"]), add_special_tokens=True)
        if response_weighting in {"trace_output", "trace_output_optional"}:
            side = "chosen" if response_key == "chosen" else "rejected"
            trace_ids = tokenizer.encode(
                str(pair.get(f"{side}_trace_text", "")), add_special_tokens=False
            )
            output_ids = tokenizer.encode(
                str(pair[f"{side}_output_text"]), add_special_tokens=False
            )
            if tokenizer.eos_token_id is not None:
                output_ids.append(int(tokenizer.eos_token_id))
            if not trace_ids and response_weighting == "trace_output":
                raise ValueError("trace_output weighting要求非空Trace")
            response_ids = trace_ids + output_ids
            response_score_weights = (
                ([float(trace_weight) / len(trace_ids)] * len(trace_ids) if trace_ids else [])
                + [float(output_weight) / len(output_ids)] * len(output_ids)
            )
        elif response_weighting == "sequence":
            response_ids = tokenizer.encode(str(pair[response_key]), add_special_tokens=False)
            if tokenizer.eos_token_id is not None:
                response_ids.append(int(tokenizer.eos_token_id))
            response_score_weights = [1.0] * len(response_ids)
        else:
            raise ValueError(f"未知 idpo.response_logp_weighting={response_weighting}")
        if len(response_ids) >= max_length:
            raise ValueError("IDPO response exceeds max_length")
        prompt_ids = _truncate_prompt(prompt_ids, max_length - len(response_ids))
        sequence = prompt_ids + response_ids
        sequences.append(sequence)
        response_masks.append([0] * len(prompt_ids) + [1] * len(response_ids))
        score_weights.append([0.0] * len(prompt_ids) + response_score_weights)
    maximum = max(len(value) for value in sequences)
    pad_id = int(tokenizer.pad_token_id)
    return {
        "input_ids": __import__("torch").tensor(
            [value + [pad_id] * (maximum - len(value)) for value in sequences], dtype=__import__("torch").long
        ),
        "attention_mask": __import__("torch").tensor(
            [[1] * len(value) + [0] * (maximum - len(value)) for value in sequences], dtype=__import__("torch").long
        ),
        "response_mask": __import__("torch").tensor(
            [value + [0] * (maximum - len(value)) for value in response_masks], dtype=__import__("torch").float32
        ),
        "score_weights": __import__("torch").tensor(
            [value + [0.0] * (maximum - len(value)) for value in score_weights],
            dtype=__import__("torch").float32,
        ),
    }


def _sequence_logps(model, batch: dict[str, Any], device, normalize: bool):
    import torch
    import torch.nn.functional as F

    encoded = {
        key: value.to(device)
        for key, value in batch.items()
        if key not in {"response_mask", "score_weights"}
    }
    outputs = model(**encoded)
    logits = outputs.logits[:, :-1, :].float()
    labels = encoded["input_ids"][:, 1:]
    mask = batch["response_mask"][:, 1:].to(device)
    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    weights = batch.get("score_weights")
    shifted_weights = None if weights is None else weights[:, 1:]
    weighted_spans = bool(
        shifted_weights is not None and (shifted_weights != mask.cpu()).any().item()
    )
    if weighted_spans:
        values = (token_logps * shifted_weights.to(device)).sum(dim=1)
    else:
        values = (token_logps * mask).sum(dim=1)
    if normalize and not weighted_spans:
        values = values / mask.sum(dim=1).clamp_min(1.0)
    return values


def dpo_loss(
    policy_chosen,
    policy_rejected,
    reference_chosen,
    reference_rejected,
    beta: float,
):
    """标准 DPO 偏好损失，单独暴露以便 CPU 契约测试。"""

    import torch.nn.functional as F

    logits = beta * (
        (policy_chosen - policy_rejected)
        - (reference_chosen - reference_rejected)
    )
    return -F.logsigmoid(logits).mean()


def _load_models(config: dict):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = config["idpo"]
    model_path = resolve_path(config["model"]["path"])
    adapter_path = resolve_path(
        settings.get("reference_adapter_path")
        or (resolve_path(config["paths"]["editor_output_dir"]) / "final_adapter")
    )
    if not adapter_path.exists():
        raise FileNotFoundError(f"IDPO reference adapter 不存在: {adapter_path}")
    device = torch.device(
        str(settings.get("device", "cuda"))
        if torch.cuda.is_available() or str(settings.get("device", "cuda")) == "cpu"
        else "cpu"
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base_policy = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    )
    base_reference = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    )
    policy = PeftModel.from_pretrained(base_policy, adapter_path, is_trainable=True).to(device)
    reference = PeftModel.from_pretrained(base_reference, adapter_path).to(device).eval()
    policy.config.use_cache = False
    reference.config.use_cache = False
    policy.gradient_checkpointing_enable()
    if hasattr(policy, "enable_input_require_grads"):
        policy.enable_input_require_grads()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    return policy, reference, tokenizer, device, adapter_path


def _train_user(
    user_id: str,
    pairs: list[dict[str, Any]],
    policy,
    reference,
    tokenizer,
    device,
    global_state: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    if len(pairs) < int(settings["minimum_pairs_per_user"]):
        return {"user_id": user_id, "status": "skipped", "pairs": len(pairs)}
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(policy, global_state)
    policy.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings.get("weight_decay", 0.0)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    beta = float(settings["beta"])
    max_length = int(settings["max_length"])
    batch_size = max(1, int(settings["batch_size"]))
    accumulation = max(1, int(settings.get("gradient_accumulation_steps", 1)))
    normalize = bool(settings.get("length_normalized_logp", True))
    response_weighting = str(settings.get("response_logp_weighting", "sequence"))
    trace_weight = float(settings.get("trace_logp_weight", 0.2))
    output_weight = float(settings.get("output_logp_weight", 1.0))
    generator = random.Random(f"{settings['seed']}:{user_id}")
    history = []
    optimizer.zero_grad(set_to_none=True)
    update_count = 0
    for epoch in range(1, int(settings["epochs"]) + 1):
        ordered = list(pairs)
        generator.shuffle(ordered)
        epoch_losses = []
        for start in range(0, len(ordered), batch_size):
            batch_pairs = ordered[start : start + batch_size]
            chosen_batch = _sequence_batch(
                batch_pairs, tokenizer, "chosen", max_length,
                response_weighting, trace_weight, output_weight,
            )
            rejected_batch = _sequence_batch(
                batch_pairs, tokenizer, "rejected", max_length,
                response_weighting, trace_weight, output_weight,
            )
            autocast = torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.float16)
            with autocast:
                policy_chosen = _sequence_logps(policy, chosen_batch, device, normalize)
                policy_rejected = _sequence_logps(policy, rejected_batch, device, normalize)
                with torch.no_grad():
                    reference_chosen = _sequence_logps(reference, chosen_batch, device, normalize)
                    reference_rejected = _sequence_logps(reference, rejected_batch, device, normalize)
                loss = dpo_loss(
                    policy_chosen,
                    policy_rejected,
                    reference_chosen,
                    reference_rejected,
                    beta,
                )
            scaler.scale(loss / accumulation).backward()
            epoch_losses.append(float(loss.detach().float().item()))
            if ((start // batch_size) + 1) % accumulation == 0 or start + batch_size >= len(ordered):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in policy.parameters() if parameter.requires_grad],
                    float(settings.get("max_grad_norm", 1.0)),
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_count += 1
        history.append({"epoch": epoch, "loss": sum(epoch_losses) / max(1, len(epoch_losses))})
    from peft import get_peft_model_state_dict

    final_state = {
        key: value.detach().cpu().clone()
        for key, value in get_peft_model_state_dict(policy).items()
    }
    policy.eval()
    return {
        "user_id": user_id,
        "status": "trained",
        "pairs": len(pairs),
        "epochs": int(settings["epochs"]),
        "updates": update_count,
        "final_loss": history[-1]["loss"] if history else None,
        "history": history,
        "state": final_state,
    }


def train(config: dict, split: str = "validation") -> dict[str, Any]:
    import torch

    settings = config["idpo"]
    round_index = int(settings["round"])
    source = read_jsonl(idpo_path(config, round_index, f"{split}_pairs.jsonl"))
    grouped = group_pairs_by_user(source)
    user_limit = int(settings.get("user_limit", 0))
    if user_limit > 0:
        grouped = {key: grouped[key] for key in sorted(grouped)[:user_limit]}
    output_root = idpo_path(config, round_index, "user_adapters")
    output_root.mkdir(parents=True, exist_ok=True)
    if bool(settings.get("mock_training", False)):
        reports = {
            user: {"user_id": user, "status": "mock_trained", "pairs": len(values)}
            for user, values in sorted(grouped.items())
        }
        for user in reports:
            (output_root / f"user_{user}").mkdir(parents=True, exist_ok=True)
            write_json(output_root / f"user_{user}" / "mock_adapter.json", reports[user])
        manifest = {"round": round_index, "split": split, "users": len(reports), "per_user": reports, "mock_training": True}
        write_json(idpo_path(config, round_index, f"{split}_editor_idpo_report.json"), manifest)
        return manifest

    policy, reference, tokenizer, device, adapter_path = _load_models(config)
    from peft import get_peft_model_state_dict

    global_state = {
        key: value.detach().cpu().clone()
        for key, value in get_peft_model_state_dict(policy).items()
    }
    reports = {}
    try:
        for index, user_id in enumerate(sorted(grouped), 1):
            result = _train_user(
                user_id,
                grouped[user_id],
                policy,
                reference,
                tokenizer,
                device,
                global_state,
                settings,
            )
            state = result.pop("state", None)
            if state is not None:
                destination = output_root / f"user_{user_id}"
                destination.mkdir(parents=True, exist_ok=True)
                # 保存标准 PEFT Adapter；同一用户的全部伪 Query 共享这一目录。
                from peft import set_peft_model_state_dict

                set_peft_model_state_dict(policy, state)
                policy.save_pretrained(destination)
                tokenizer.save_pretrained(destination)
                result["adapter_path"] = str(destination)
            reports[user_id] = result
            print(
                f"IDPO user {index}/{len(grouped)} user={user_id} status={result['status']} "
                f"pairs={result['pairs']}",
                flush=True,
            )
    finally:
        del policy, reference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    manifest = {
        "protocol": "per_user_iterative_dpo_v1",
        "round": round_index,
        "split": split,
        "reference_adapter": str(adapter_path),
        "users": len(reports),
        "trained_users": sum(value["status"] == "trained" for value in reports.values()),
        "skipped_users": sum(value["status"] == "skipped" for value in reports.values()),
        "preference_source": str(settings.get("preference_source", "teacher_judge")),
        "teacher_sees_hidden_target": False,
        "hidden_target_used_for_label_only": str(settings.get("preference_source")) == "loo_gold",
        "gold_visible_during_rollout": False,
        "backbone_frozen": True,
        "settings": {
            key: settings[key]
            for key in ("beta", "epochs", "learning_rate", "batch_size", "gradient_accumulation_steps", "minimum_pairs_per_user")
        },
        "per_user": reports,
    }
    manifest["settings"].update(
        {
            "response_logp_weighting": str(
                settings.get("response_logp_weighting", "sequence")
            ),
            "trace_logp_weight": float(settings.get("trace_logp_weight", 0.2)),
            "output_logp_weight": float(settings.get("output_logp_weight", 1.0)),
        }
    )
    write_json(idpo_path(config, round_index, f"{split}_editor_idpo_report.json"), manifest)
    print(f"per-user Editor IDPO -> {output_root}; users={len(reports)}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="20 - Train per-user Editor IDPO adapters")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    train(load_config(args.config), args.split)


if __name__ == "__main__":
    main()
