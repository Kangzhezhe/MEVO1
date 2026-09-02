"""生成不挂载任何 LoRA 的 Llama2 基础模型文本基线。

基础模型未经过 JSON 指令微调，因此基线使用纯文本标题协议；JSON 解析
只属于 Editor 的工程接口，不能作为 Base 的质量指标。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline_common import (  # noqa: E402
    load_config,
    read_jsonl,
    resolve_path,
    score,
    stage_path,
    visible_history,
    write_jsonl,
)


def _prompt(row: dict, parent: dict, config: dict) -> str:
    history = visible_history(row, 8)
    examples = "\n".join(
        f"Example input: {item['input'][:300]}\nExample title: {item['output'][:180]}"
        for item in history
    )
    return (
        "Write one concise academic paper title. Use the abstract and the user title "
        "examples as guidance. Preserve factual content. Output only the title on one line.\n\n"
        f"ABSTRACT:\n{row['source_text']}\n\n"
        f"USER TITLE EXAMPLES:\n{examples}\n\n"
        f"PARENT TITLE:\n{parent['text']}\n\nTITLE:\n"
    )


def _clean(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:text|title)?\s*|\s*```$", "", text, flags=re.I).strip()
    text = re.sub(r"^(?:title|output)\s*:\s*", "", text, flags=re.I).strip()
    # A base completion may add an explanation after the first line. The
    # baseline protocol evaluates the first non-empty title line only.
    return next((line.strip().strip('"') for line in text.splitlines() if line.strip()), "")


def run(config: dict) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = read_jsonl(stage_path(config, "test", "seeds"))
    model_path = resolve_path(config["model"]["path"])
    adapter = config.get("model", {}).get("base_text_adapter")
    adapter_path = resolve_path(adapter) if adapter else None
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or model_path, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model = model.to(device).eval()
    name = str(config.get("model", {}).get("base_text_output_name", "")) or (
        "base_text" if adapter_path is None else adapter_path.parent.name
    )
    destination = resolve_path(config["paths"]["prediction_dir"]) / name / "test_predictions.jsonl"
    existing = read_jsonl(destination) if destination.exists() else []
    done = {str(row["id"]): row for row in existing}
    output = list(done.values())
    if done:
        print(f"resume text predictions: {len(done)}/{len(rows)}", flush=True)
    for index, row in enumerate(rows, 1):
        parents = list(row.get("candidates", []))
        if not parents:
            continue
        if str(row["id"]) in done:
            continue
        parent = parents[0]
        prompt = _prompt(row, parent, config)
        encoded = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        prediction = _clean(raw)
        output.append(
            {
                "id": str(row["id"]),
                "user_id": str(row.get("user_id", "")),
                "source_text": str(row["source_text"]),
                "target": str(row["target"]),
                "parent": str(parent["text"]),
                "prediction": prediction,
                "raw_response": raw,
                "error": None,
            }
        )
        if index % 10 == 0:
            write_jsonl(destination, output)
            print(f"base text progress {index}/{len(rows)}", flush=True)
    write_jsonl(destination, output)
    print(f"Base text predictions -> {destination}; rows={len(output)}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="30 - generate raw-text Base predictions")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default="", help="optional LoRA adapter for matched text protocol")
    parser.add_argument("--output-name", default="", help="output subdirectory name")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.adapter:
        config.setdefault("model", {})["base_text_adapter"] = args.adapter
    if args.output_name:
        config.setdefault("model", {})["base_text_output_name"] = args.output_name
    run(config)
