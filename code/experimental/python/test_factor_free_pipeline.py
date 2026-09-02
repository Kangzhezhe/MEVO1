"""Gold-aware 阶段一的数据隔离与流水线契约测试。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from pipeline_common import build_editor_prompt, candidate, response_parts


HERE = Path(__file__).resolve().parent


def _stage(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(identifier: str = "q1") -> dict:
    return {
        "id": identifier,
        "user_id": "u1",
        "source_text": "We present a memory reclamation method for app launch latency.",
        "target": "SmartLMK: A Memory Reclamation Scheme for Improving App Launch Time",
        "retrieved_profile": [
            {"id": "h2", "abstract": "Prior system paper", "title": "SmartOld: A Memory System"},
            {"id": "h7", "abstract": "Prior latency paper", "title": "FastApp: Improving Launch Time"},
        ],
        "candidates": [
            candidate(f"{identifier}_task_0", "task_seed", "Memory Reclamation for Faster Apps"),
            candidate(f"{identifier}_task_1", "task_seed", "Improving App Launch with Memory Reclamation"),
            candidate(f"{identifier}_profile_0", "profile_seed", "SmartLMK: Improving App Launch Time"),
            candidate(f"{identifier}_profile_1", "profile_seed", "FastApp Memory Reclamation"),
        ],
    }


def _config() -> dict:
    return {
        "generation": {"task_seeds": 2, "profile_seeds": 2},
        "gold_aware_sft": {
            "maximum_history_records": 2,
            "schema_retries": 0,
        },
        "sft_data": {
            "maximum_history_records": 2,
            "maximum_examples_per_query": 4,
            "validation_fraction": 0.5,
        },
        "training": {"trace_loss_weight": 0.2, "output_loss_weight": 1.0},
    }


def test_simple_trace_teacher_accepts_one_visible_evidence_and_partial_fallback() -> None:
    module = _stage(
        "31_build_simple_conditional_traces.py", "simple_trace_teacher_contract"
    )
    row = _row()
    history = [
        {"id": "h2", "input": "Prior system paper", "output": "SmartOld"},
        {"id": "h7", "input": "Prior latency paper", "output": "FastApp"},
    ]
    value = {
        "traces": [
            {
                "parent_id": row["candidates"][0]["candidate_id"],
                "evidence_ids": ["h2"],
                "edit_reason": "History h2 supports placing the system name first here.",
                "edit_action": "Move the system name to the beginning.",
            }
        ]
    }
    traces = module._validate_response(value, row, history)
    assert traces[0]["personalized"] is True
    assert traces[0]["evidence_ids"] == ["h2"]
    assert all(item["personalized"] is False for item in traces[1:])


def test_teacher_gold_aware_prompt_and_program_owned_output() -> None:
    module = _stage("04_teacher_evolve.py", "gold_aware_stage04_prompt_test")

    class Client:
        config = {"provider": "openai_compatible", "model": "mock-qwen"}
        prompt = ""

        def json(self, task, prompt, context):
            self.prompt = prompt
            assert context["reference_output"] == _row()["target"]
            return {
                "traces": [
                    {
                        "parent_id": parent["candidate_id"],
                        "task_correction": "Retain the method and the app launch objective.",
                        "profile_signal": {
                            "evidence_ids": ["h2"],
                            "observation": "Visible history places a named system before its purpose.",
                        },
                        "edit_action": "Use the demonstrated system-first structure and precise task wording.",
                        # 即使 Teacher 越权返回 output，程序也必须忽略它。
                        "output": "WRONG TEACHER OUTPUT",
                    }
                    for parent in _row()["candidates"]
                ]
            }, ""

        def invalidate(self, task, prompt):
            raise AssertionError("valid response must not be invalidated")

    client = Client()
    result = module._build_one(_row(), _config(), client)
    assert _row()["target"] in client.prompt
    assert len(result["gold_aware_traces"]) == 4
    assert all(item["output"] == _row()["target"] for item in result["gold_aware_traces"])
    assert result["gold_aware_metadata"]["teacher_generates_output"] is False


def test_student_prompt_is_target_blind_and_sft_output_is_exact_gold() -> None:
    stage04 = _stage("04_teacher_evolve.py", "gold_aware_stage04_mock_test")
    stage05 = _stage("05_build_editor_sft.py", "gold_aware_stage05_test")

    class Mock:
        config = {"provider": "mock", "model": "mock"}

    row = stage04._build_one(_row(), _config(), Mock())
    examples = [stage05._example(row, trace, _config()) for trace in row["gold_aware_traces"]]
    assert len(examples) == 4
    for example in examples:
        payload = json.loads(
            example["prompt"].split("PAYLOAD:\n", 1)[1].rsplit("\n\nOUTPUT:\n", 1)[0]
        )
        assert "target" not in payload
        assert "reference_output" not in payload
        assert example["output"] == row["target"]
        assert json.loads(example["trace_text"] + example["output_text"])["output"] == row["target"]


def test_s0_output_only_has_no_teacher_trace_and_exact_gold() -> None:
    stage05 = _stage("05_build_editor_sft.py", "s0_stage05_test")
    config = _config()
    config["sft_data"]["supervision_mode"] = "output_only"
    example = stage05._output_only_example(_row(), _row()["candidates"][0], config)
    payload = json.loads(
        example["prompt"].split("PAYLOAD:\n", 1)[1].rsplit("\n\nOUTPUT:\n", 1)[0]
    )
    assert "target" not in payload
    assert "reference_output" not in payload
    assert example["trace_text"] == ""
    assert json.loads(example["output_text"]) == {"output": _row()["target"]}


def test_s0_output_only_inference_schema_is_minimal() -> None:
    prompt = build_editor_prompt(
        _row(), "mutation", _row()["candidates"][0], None, 2, "output_only"
    )
    assert 'REQUIRED_SCHEMA:\n{"output":"..."}' in prompt
    assert "task_correction" not in prompt


def test_s0_can_treat_a_single_line_generation_as_output() -> None:
    module = _stage("07_generate_editor_pool.py", "s0_raw_output_test")
    assert module.extract_json(
        "A Memory Reclamation Scheme for Improving App Launch Time", output_only=True
    ) == {"output": "A Memory Reclamation Scheme for Improving App Launch Time"}
    with pytest.raises(ValueError):
        module.extract_json("A plain title", output_only=False)


def test_editor_prompt_truncation_keeps_head_and_parent_tail() -> None:
    module = _stage("07_generate_editor_pool.py", "editor_truncation_test")
    values = list(range(20))
    assert module.truncate_prompt_ids(values, 8) == [0, 1, 2, 3, 16, 17, 18, 19]
    assert module.truncate_prompt_ids(values[:5], 8) == values[:5]


def test_response_spans_include_task_correction() -> None:
    example = {
        "operation_type": "mutation",
        "decision": "revise",
        "task_correction": "Keep the main technical contribution.",
        "profile_signal": {"evidence_ids": ["h2"], "observation": "System name first."},
        "edit_action": "Move the name.",
        "output": "SmartLMK: Faster Apps",
    }
    trace, output = response_parts(example)
    value = json.loads(trace + output)
    assert value["task_correction"] == example["task_correction"]
    assert value["output"] == example["output"]


def test_editor_prompt_never_serializes_gold() -> None:
    row = _row()
    prompt = build_editor_prompt(row, "mutation", row["candidates"][0], None, 2)
    payload = json.loads(prompt.split("PAYLOAD:\n", 1)[1].rsplit("\n\nOUTPUT:\n", 1)[0])
    assert set(payload) == {"operation_type", "current_input", "retrieved_history", "parent_a"}
    assert row["target"] not in json.dumps(payload, ensure_ascii=False)


def test_seed_parser_and_retry_do_not_insert_previous_answers() -> None:
    module = _stage("03_generate_seeds.py", "gold_aware_stage03_test")
    assert module._candidate_values({"titles": ["First", "Second"]}) == ["First", "Second"]

    class Client:
        prompts: list[str] = []

        def json(self, task, prompt, context):
            self.prompts.append(prompt)
            return ({"title": "First"}, "") if len(self.prompts) == 1 else (["First", "Second"], "")

        def invalidate(self, task, prompt):
            pass

    client = Client()
    assert module._request_group(client, "seed", "PROMPT", {}, 2, 2) == ["First", "Second"]
    assert "First" not in client.prompts[1]


def test_business_retry_does_not_include_previous_response() -> None:
    module = _stage("04_teacher_evolve.py", "gold_aware_stage04_retry_test")

    class Client:
        prompts: list[str] = []

        def json(self, task, prompt, context):
            self.prompts.append(prompt)
            value = "bad response text" if len(self.prompts) == 1 else "valid"
            return {"traces": [{"value": value}]}, ""

        def invalidate(self, task, prompt):
            pass

    def validator(values):
        if values[0]["value"] != "valid":
            raise ValueError("invalid business field")

    client = Client()
    values = module._request_list(client, "trace", "PROMPT", {}, "traces", 1, 1, validator)
    assert values == [{"value": "valid"}]
    assert "bad response text" not in client.prompts[1]


def test_mock_local_editor_builds_ten_target_blind_candidates() -> None:
    module = _stage("07_generate_editor_pool.py", "gold_aware_stage07_test")
    config = {
        "evolution": {"crossovers_per_query": 2, "candidate_budget": 10, "maximum_history_records": 2},
        "inference": {"batch_size": 4, "fallback_on_invalid_json": True},
        "metric": {"primary": "rouge_l", "preference_margin": 0.02},
    }
    result = module.generate_one(_row(), config, module.MockLocalEditor())
    assert len(result["candidates"] + result["mutations"]) == 10
    assert result["editor_metadata"]["gold_visible_during_generation"] is False


def test_mock_local_editor_batches_multiple_queries() -> None:
    module = _stage("07_generate_editor_pool.py", "gold_aware_stage07_batch_test")
    config = {
        "evolution": {"crossovers_per_query": 2, "candidate_budget": 10, "maximum_history_records": 2},
        "inference": {"batch_size": 8, "fallback_on_invalid_json": True},
        "metric": {"primary": "rouge_l", "preference_margin": 0.02},
    }
    rows = [ _row("q1"), _row("q2") ]
    results = module.generate_many_rows(rows, config, module.MockLocalEditor())
    assert [item["id"] for item in results] == ["q1", "q2"]
    assert all(len(item["candidates"] + item["mutations"]) == 10 for item in results)
    assert all(item["editor_metadata"]["cross_query_batch"] is True for item in results)


def test_ranker_training_source_is_local_editor_pool() -> None:
    source = (HERE / "08_build_scorer_data.py").read_text(encoding="utf-8")
    assert 'train_source=stage_path(config, "train", "editor")' in source
    assert 'train_source=stage_path(config, "train", "evolved")' not in source


def test_pipeline_generates_local_train_pool_before_ranker() -> None:
    source = (HERE / "run_pipeline.sh").read_text(encoding="utf-8")
    assert "for split in train validation test" in source
    assert source.index("for split in train validation test") < source.index("08_build_scorer_data.py")


def test_gold_aware_stage_rejects_non_train_split() -> None:
    module = _stage("04_teacher_evolve.py", "gold_aware_stage04_split_test")
    with pytest.raises(ValueError, match="只能在 train"):
        module.build(_config(), "validation")
