"""训练一个共享的全局 IDPO LoRA Adapter。

所有训练用户的 LOO pairs 合并后连续优化同一 policy；不按 user 重置参数，
也不创建 per-user adapter。Reference 始终冻结为 Shared SFT Adapter。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from idpo_common import idpo_path  # noqa: E402
from pipeline_common import load_config, read_jsonl, resolve_path, write_json, write_jsonl  # noqa: E402


def _truncate(ids: list[int], maximum: int) -> list[int]:
    if len(ids) <= maximum:
        return ids
    head = max(1, maximum // 2)
    return ids[:head] + ids[-(maximum - head):]


def _batch(pairs: list[dict[str, Any]], tokenizer, side: str, maximum: int, trace_weight: float, output_weight: float):
    import torch
    sequences, masks, weights = [], [], []
    for pair in pairs:
        prompt = tokenizer.encode(str(pair["prompt"]), add_special_tokens=True)
        trace_ids = tokenizer.encode(str(pair.get(f"{side}_trace_text", "")), add_special_tokens=False)
        output_ids = tokenizer.encode(str(pair[f"{side}_output_text"]), add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            output_ids.append(int(tokenizer.eos_token_id))
        response = trace_ids + output_ids
        if len(response) >= maximum:
            raise ValueError(f"{pair['pair_id']} response exceeds max_length")
        prompt = _truncate(prompt, maximum - len(response))
        seq = prompt + response
        sequences.append(seq)
        masks.append([0] * len(prompt) + [1] * len(response))
        weights.append([0.0] * len(prompt) + ([trace_weight / max(1, len(trace_ids))] * len(trace_ids)) + ([output_weight / max(1, len(output_ids))] * len(output_ids)))
    width = max(map(len, sequences))
    pad = int(tokenizer.pad_token_id)
    return {
        "input_ids": torch.tensor([x + [pad] * (width - len(x)) for x in sequences], dtype=torch.long),
        "attention_mask": torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in sequences], dtype=torch.long),
        "response_mask": torch.tensor([x + [0] * (width - len(x)) for x in masks], dtype=torch.float32),
        "score_weights": torch.tensor([x + [0.0] * (width - len(x)) for x in weights], dtype=torch.float32),
    }


def _logps(model, batch: dict[str, Any], device):
    import torch
    import torch.nn.functional as F
    encoded = {key: value.to(device) for key, value in batch.items() if key not in {"response_mask", "score_weights"}}
    # 不 materialize FP32 的完整 [batch, seq, vocab] log_softmax 张量。
    # Qwen vocab 较大，原实现的 float 转换在连续 forward 时会制造数 GB 临时显存峰值。
    logits = model(**encoded).logits[:, :-1, :]
    labels = encoded["input_ids"][:, 1:]
    token_logits = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    token = token_logits - torch.logsumexp(logits, dim=-1)
    token = token.float()
    mask = batch["response_mask"][:, 1:].to(device)
    weights = batch["score_weights"][:, 1:].to(device)
    return (token * weights).sum(dim=1)


def _load_tokenizer(config: dict[str, Any], adapter_path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _load_model(config: dict[str, Any], adapter_path, trainable: bool):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base_path = resolve_path(config["model"]["path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        base, adapter_path, is_trainable=trainable
    ).to(device)
    if not trainable:
        model.eval()
    model.config.use_cache = False
    if trainable:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        for parameter in model.parameters():
            parameter.requires_grad = False
    return model, device


def _load(config: dict[str, Any]):
    adapter_path = resolve_path(config["paths"]["editor_output_dir"]) / "final_adapter"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Shared SFT Adapter 不存在: {adapter_path}")
    tokenizer = _load_tokenizer(config, adapter_path)
    policy, device = _load_model(config, adapter_path, trainable=True)
    return policy, tokenizer, device, adapter_path


def _reference_cache_path(config: dict[str, Any]):
    return idpo_path(config, 0, "reference_logps.jsonl")


def _compute_reference_logps(config: dict[str, Any], pairs: list[dict[str, Any]], tokenizer):
    """阶段 A：单独计算 Reference 标量 log-prob，然后释放 7B Reference。

    缓存的只有每个 pair 的两个序列分数，不缓存 logits，避免磁盘和内存膨胀。
    Reference 与 SFT Adapter 完全冻结，因此该阶段不会改变 DPO 数学目标。
    """
    import gc
    import torch

    cache_path = _reference_cache_path(config)
    if cache_path.exists():
        cached = read_jsonl(cache_path)
        by_id = {str(row.get("pair_id")): row for row in cached}
        if all(str(pair["pair_id"]) in by_id for pair in pairs):
            return [by_id[str(pair["pair_id"])] for pair in pairs]

    adapter_path = resolve_path(config["paths"]["editor_output_dir"]) / "final_adapter"
    reference, device = _load_model(config, adapter_path, trainable=False)
    settings = config["global_idpo"]
    batch_size = max(1, int(settings.get("reference_batch_size", settings.get("batch_size", 1))))
    max_length = int(settings.get("max_length", 2048))
    trace_weight = float(settings.get("trace_weight", 0.2))
    output_weight = float(settings.get("output_weight", 1.0))
    values = []
    try:
        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                chunk = pairs[start : start + batch_size]
                chosen = _batch(chunk, tokenizer, "chosen", max_length, trace_weight, output_weight)
                rejected = _batch(chunk, tokenizer, "rejected", max_length, trace_weight, output_weight)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.float16):
                    chosen_logp = _logps(reference, chosen, device).detach().cpu().tolist()
                    rejected_logp = _logps(reference, rejected, device).detach().cpu().tolist()
                values.extend(
                    {
                        "pair_id": str(pair["pair_id"]),
                        "reference_chosen_logp": float(chosen_value),
                        "reference_rejected_logp": float(rejected_value),
                    }
                    for pair, chosen_value, rejected_value in zip(chunk, chosen_logp, rejected_logp)
                )
                print(f"reference logp {min(start + len(chunk), len(pairs))}/{len(pairs)}", flush=True)
    finally:
        del reference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    write_jsonl(cache_path, values)
    return values


def train(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    settings = config["global_idpo"]
    pairs = read_jsonl(idpo_path(config, 0, "train_pairs.jsonl"))
    if not pairs:
        raise ValueError("train_pairs.jsonl 为空")
    tokenizer = _load_tokenizer(
        config, resolve_path(config["paths"]["editor_output_dir"]) / "final_adapter"
    )
    reference_values = _compute_reference_logps(config, pairs, tokenizer)
    reference_by_id = {str(row["pair_id"]): row for row in reference_values}
    policy, _, device, _ = _load(config)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=float(settings["learning_rate"]), weight_decay=float(settings.get("weight_decay", 0.0)))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    batch_size = max(1, int(settings.get("batch_size", 2)))
    accumulation = max(1, int(settings.get("gradient_accumulation_steps", 1)))
    max_length = int(settings.get("max_length", 2048))
    beta = float(settings.get("beta", 0.1))
    trace_weight = float(settings.get("trace_weight", 0.2))
    output_weight = float(settings.get("output_weight", 1.0))
    rng = random.Random(int(config["project"]["seed"]))
    history = []
    optimizer.zero_grad(set_to_none=True)
    updates = 0
    for epoch in range(1, int(settings.get("epochs", 1)) + 1):
        ordered = list(pairs)
        rng.shuffle(ordered)
        losses = []
        for start in range(0, len(ordered), batch_size):
            chunk = ordered[start:start + batch_size]
            chosen = _batch(chunk, tokenizer, "chosen", max_length, trace_weight, output_weight)
            rejected = _batch(chunk, tokenizer, "rejected", max_length, trace_weight, output_weight)
            reference_chunk = [reference_by_id[str(item["pair_id"])] for item in chunk]
            rc = torch.tensor(
                [float(item["reference_chosen_logp"]) for item in reference_chunk],
                dtype=torch.float32,
                device=device,
            )
            rr = torch.tensor(
                [float(item["reference_rejected_logp"]) for item in reference_chunk],
                dtype=torch.float32,
                device=device,
            )
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.float16):
                pc = _logps(policy, chosen, device)
                pr = _logps(policy, rejected, device)
                loss = -torch.nn.functional.logsigmoid(beta * ((pc - pr) - (rc - rr))).mean()
            scaler.scale(loss / accumulation).backward()
            losses.append(float(loss.detach().float().item()))
            step = start // batch_size + 1
            if step % accumulation == 0 or start + batch_size >= len(ordered):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], float(settings.get("max_grad_norm", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
        history.append({"epoch": epoch, "loss": sum(losses) / max(1, len(losses)), "updates": updates})
        print(f"global IDPO epoch={epoch} loss={history[-1]['loss']:.6f} updates={updates}", flush=True)
    output_dir = resolve_path(config["paths"]["editor_output_dir"]) / "global_idpo"
    adapter_dir = output_dir / "final_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    report = {"protocol": "global_shared_idpo_lora", "pairs": len(pairs), "users": len({str(p["user_id"]) for p in pairs}), "epochs": int(settings.get("epochs", 1)), "updates": updates, "history": history, "reference_adapter": str(resolve_path(config["paths"]["editor_output_dir"]) / "final_adapter"), "final_adapter": str(adapter_dir), "trace_weight": trace_weight, "output_weight": output_weight, "teacher_crossover_in_pairs": bool(settings.get("dpo_include_teacher_crossover", False))}
    write_json(output_dir / "training_report.json", report)
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="27 - train global IDPO LoRA")
    parser.add_argument("--config", default=str(HERE.parent / "config_global.yaml"))
    args = parser.parse_args()
    train(load_config(args.config))
