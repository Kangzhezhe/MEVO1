"""阶段 06：使用 FP16 LoRA 训练无 Factor 的全局 Editor。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline_common import (  # noqa: E402
    load_config,
    read_jsonl,
    resolve_path,
    write_json,
)


class TraceOutputDataset(Dataset):
    """将 Trace 和 Output 分开归一化，使权重具有稳定含义。"""

    def __init__(self, rows: list[dict[str, Any]], tokenizer, config: dict):
        self.features = []
        self.truncated_prompts = 0
        settings = config["training"]
        max_length = int(settings["max_length"])
        trace_weight = float(settings["trace_loss_weight"])
        output_weight = float(settings["output_loss_weight"])
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer 必须定义 eos_token_id")
        for row in rows:
            sample_weight = float(row.get("sample_weight", 1.0))
            if sample_weight <= 0:
                raise ValueError(f"example={row['example_id']} 的 sample_weight 必须大于 0")
            prompt_ids = tokenizer.encode(str(row["prompt"]), add_special_tokens=True)
            trace_ids = tokenizer.encode(str(row["trace_text"]), add_special_tokens=False)
            output_ids = tokenizer.encode(str(row["output_text"]), add_special_tokens=False) + [
                eos_id
            ]
            response_length = len(trace_ids) + len(output_ids)
            if response_length >= max_length:
                raise ValueError(f"example={row['example_id']} 的响应超过 max_length")
            maximum_prompt = max_length - response_length
            if len(prompt_ids) > maximum_prompt:
                # 同时保留输入开头和靠近末尾的 Parent/Profile 控制信息。
                head = max(1, maximum_prompt // 2)
                prompt_ids = prompt_ids[:head] + prompt_ids[-(maximum_prompt - head) :]
                self.truncated_prompts += 1
            input_ids = prompt_ids + trace_ids + output_ids
            labels = [-100] * len(prompt_ids) + trace_ids + output_ids
            # 每个 span 的总权重固定，不因 Trace 更长而压过最终输出。
            trace_token_weight = sample_weight * trace_weight / max(len(trace_ids), 1)
            output_token_weight = sample_weight * output_weight / max(len(output_ids), 1)
            weights = (
                [0.0] * len(prompt_ids)
                + [trace_token_weight] * len(trace_ids)
                + [output_token_weight] * len(output_ids)
            )
            self.features.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "loss_weights": weights,
                    "length": len(input_ids),
                }
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        maximum = max(len(item["input_ids"]) for item in features)
        result = {"input_ids": [], "attention_mask": [], "labels": [], "loss_weights": []}
        for item in features:
            padding = maximum - len(item["input_ids"])
            result["input_ids"].append(item["input_ids"] + [self.pad_token_id] * padding)
            result["attention_mask"].append([1] * len(item["input_ids"]) + [0] * padding)
            result["labels"].append(item["labels"] + [-100] * padding)
            result["loss_weights"].append(item["loss_weights"] + [0.0] * padding)
        return {
            "input_ids": torch.tensor(result["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(result["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(result["labels"], dtype=torch.long),
            "loss_weights": torch.tensor(result["loss_weights"], dtype=torch.float32),
        }


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        weights = inputs.pop("loss_weights")
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        shifted_weights = weights[:, 1:].to(logits.device)
        token_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shifted_labels)
        active_weights = shifted_weights * shifted_labels.ne(-100)
        # 每个样本的 Trace/Output 已分别归一化，再对 batch 求均值。
        sample_loss = (token_loss * active_weights).sum(dim=1)
        loss = sample_loss.mean()
        return (loss, outputs) if return_outputs else loss


def train(config: dict, max_steps_override: int | None = None) -> dict[str, Any]:
    model_path = resolve_path(config["model"]["path"])
    data_dir = resolve_path(config["paths"]["sft_dir"])
    output_dir = resolve_path(config["paths"]["editor_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        raise FileNotFoundError(f"缺少基础模型: {model_path}")
    train_rows = read_jsonl(data_dir / "train_sft.jsonl")
    validation_rows = read_jsonl(data_dir / "validation_sft.jsonl")
    settings = config["training"]
    set_seed(int(settings["seed"]))
    if torch.cuda.is_available() and float(settings.get("cuda_memory_fraction", 0)) > 0:
        torch.cuda.set_per_process_memory_fraction(
            float(settings["cuda_memory_fraction"]), device=0
        )
    initial_adapter_value = str(settings.get("initial_adapter_path", "")).strip()
    initial_adapter = resolve_path(initial_adapter_value) if initial_adapter_value else None
    if initial_adapter is not None and not initial_adapter.exists():
        raise FileNotFoundError(f"Warm-up初始Adapter不存在: {initial_adapter}")
    tokenizer_source = initial_adapter if initial_adapter is not None else model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, use_fast=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_data = TraceOutputDataset(train_rows, tokenizer, config)
    validation_data = TraceOutputDataset(validation_rows, tokenizer, config)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if initial_adapter is not None:
        from peft import PeftModel

        # 在阶段一共享 LoRA 上继续训练，而不是重新随机初始化第二个 LoRA。
        # 保存结果仍是一个标准 PEFT Adapter，可直接作为 IDPO reference/policy。
        model = PeftModel.from_pretrained(
            model, initial_adapter, is_trainable=True
        )
    else:
        lora = config["lora"]
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(lora["rank"]),
                lora_alpha=int(lora["alpha"]),
                lora_dropout=float(lora["dropout"]),
                target_modules=list(lora["target_modules"]),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    max_steps = (
        int(max_steps_override)
        if max_steps_override is not None
        else int(settings.get("max_steps", -1))
    )
    arguments = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=float(settings["epochs"]),
        max_steps=max_steps,
        per_device_train_batch_size=int(settings["batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(settings["gradient_accumulation_steps"]),
        learning_rate=float(settings["learning_rate"]),
        warmup_ratio=float(settings["warmup_ratio"]),
        weight_decay=float(settings.get("weight_decay", 0.0)),
        max_grad_norm=float(settings.get("max_grad_norm", 1.0)),
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=int(settings.get("logging_steps", 10)),
        evaluation_strategy="steps" if max_steps > 0 else "epoch",
        eval_steps=1 if max_steps > 0 else None,
        save_strategy=str(settings.get("save_strategy", "steps" if max_steps > 0 else "epoch")),
        save_steps=int(settings.get("save_steps", 1 if max_steps > 0 else 500)),
        save_total_limit=int(settings.get("save_total_limit", 2)),
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=int(settings["seed"]),
        data_seed=int(settings["seed"]),
    )
    trainer = WeightedTrainer(
        model=model,
        args=arguments,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=Collator(tokenizer.pad_token_id),
        tokenizer=tokenizer,
    )
    # 外部终止后自动从最近 checkpoint 继续，避免 SFT 进度丢失。
    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = get_last_checkpoint(str(output_dir / "checkpoints"))
    result = trainer.train(resume_from_checkpoint=last_checkpoint) if last_checkpoint else trainer.train()
    evaluation = trainer.evaluate()
    adapter_dir = output_dir / "final_adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    report = {
        "protocol": {
            "output_only": "s0_output_only_global_editor_lora_v1",
            "gold_aware_trace": "s1_free_trace_global_editor_lora_v1",
            "atomic_trace": "s2_atomic_trace_global_editor_lora_v1",
            "conditional_preference_trace": "conditional_preference_global_editor_lora_v1",
            "simple_conditional_trace": "top8_simple_conditional_global_editor_lora_v1",
        }.get(str(config["sft_data"].get("supervision_mode")), "global_editor_lora_v1"),
        "supervision_mode": str(
            config["sft_data"].get("supervision_mode", "gold_aware_trace")
        ),
        "model": str(model_path),
        "initial_adapter": None if initial_adapter is None else str(initial_adapter),
        "train_examples": len(train_data),
        "validation_examples": len(validation_data),
        "trace_loss_weight": float(settings["trace_loss_weight"]),
        "output_loss_weight": float(settings["output_loss_weight"]),
        "span_length_normalized": True,
        "query_normalized_sample_weight": True,
        "train_loss": float(result.training_loss),
        "eval_loss": float(evaluation["eval_loss"]),
        "global_step": int(trainer.state.global_step),
        "final_adapter": str(adapter_dir),
        "explicit_user_factors": False,
    }
    write_json(output_dir / "training_report.json", report)
    print(f"factor-free Editor adapter -> {adapter_dir}; report={report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="06 - 训练无 Factor Qwen LoRA Editor")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    train(load_config(args.config), args.max_steps)


if __name__ == "__main__":
    main()
