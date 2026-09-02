"""用全局 IDPO Adapter 对 test Query 生成单个标题。"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline_common import build_configured_editor_prompt, load_config, read_jsonl, resolve_path, stage_path, write_jsonl  # noqa: E402


def run(config: dict[str, Any]) -> Path:
    module_path = HERE / "07_generate_editor_pool.py"
    spec = importlib.util.spec_from_file_location("global_prediction_editor", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    local = copy.deepcopy(config)
    # 默认使用 Global IDPO adapter；评估 SFT 基线时可通过
    # model.prediction_adapter_path 指定共享 SFT adapter。
    configured = config.get("model", {}).get("prediction_adapter_path")
    base_only = bool(config.get("model", {}).get("prediction_base_only", False))
    local.setdefault("model", {})["base_only"] = base_only
    local.setdefault("model", {})["adapter_path"] = str(
        resolve_path(configured)
        if configured
        else resolve_path(config["paths"]["editor_output_dir"]) / "global_idpo" / "final_adapter"
    )
    prediction_mode = str(config.get("sft_data", {}).get("supervision_mode", "output_only"))
    if prediction_mode not in {"output_only", "plain_output_only"}:
        prediction_mode = "output_only"
    local.setdefault("sft_data", {})["supervision_mode"] = prediction_mode
    local.setdefault("inference", {})["batch_size"] = int(config.get("evaluation", {}).get("prediction_batch_size", 8))
    local["inference"]["do_sample"] = False
    local["inference"]["max_new_tokens"] = int(config.get("evaluation", {}).get("max_new_tokens", 128))
    editor = module.LocalEditor(local)
    rows = read_jsonl(stage_path(config, "test", "seeds"))
    prompts = []
    metadata = []
    for row in rows:
        parents = list(row.get("candidates", []))
        if not parents:
            continue
        parent = parents[0]
        prompts.append(build_configured_editor_prompt(
            config,
            row, "mutation", parent, None,
            int(config["generation"].get("maximum_history_records", 8)),
            supervision_mode=prediction_mode,
            history_input_max_chars=int(config["simple_conditional_trace"].get("history_input_max_chars", 500)),
            history_output_max_chars=int(config["simple_conditional_trace"].get("history_output_max_chars", 300)),
        ))
        metadata.append((row, parent))
    raw = editor.generate_many(prompts)
    output = []
    for (row, parent), (payload, raw_text, error) in zip(metadata, raw):
        title = str(payload.get("output", "")).strip() if isinstance(payload, dict) else ""
        output.append({"id": str(row["id"]), "user_id": str(row.get("user_id", "")), "source_text": str(row["source_text"]), "target": str(row["target"]), "parent": str(parent["text"]), "prediction": title, "raw_response": raw_text, "error": None if error is None else str(error)})
    destination = resolve_path(config["paths"]["prediction_dir"]) / "test_predictions.jsonl"
    write_jsonl(destination, output)
    model_stage = "Base" if base_only else ("SFT" if configured else "Global IDPO")
    print(f"{model_stage} predictions -> {destination}; rows={len(output)}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="28 - generate global test predictions")
    parser.add_argument("--config", default=str(HERE.parent / "config_global.yaml"))
    args = parser.parse_args()
    run(load_config(args.config))
