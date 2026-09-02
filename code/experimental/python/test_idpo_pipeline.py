"""IDPO 阶段二的数据隔离与 DPO 契约测试。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


def _stage(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row() -> dict:
    return {
        "id": "u1:profile:h1",
        "user_id": "u1",
        "source_text": "We study a memory method for app launch latency.",
        "target": "HIDDEN TITLE MUST NEVER ENTER IDPO PROMPTS",
        "retrieved_profile": [
            {"id": "h2", "abstract": "A prior memory paper.", "title": "Memory Systems"},
            {"id": "h3", "abstract": "A prior latency paper.", "title": "Fast App Launch"},
        ],
        "candidates": [
            {"candidate_id": "p1", "type": "task_seed", "text": "Memory for App Launch"},
            {"candidate_id": "p2", "type": "profile_seed", "text": "Fast Memory Method"},
        ],
    }


def _config() -> dict:
    return {
        "idpo": {
            "round": 0,
            "maximum_history_records": 2,
            "rollout_samples": 3,
            "minimum_valid_responses": 2,
            "mock_editor": True,
            "minimum_confidence": 0.65,
            "judge_retries": 1,
            "judge_concurrency": 1,
            "checkpoint_every": 1,
            "resume_existing": False,
            "mock_training": True,
        },
        "paths": {"idpo_dir": "/tmp/mevo_idpo_contract"},
        "teacher": {"provider": "mock", "model": "mock", "cache_dir": "/tmp/mevo_idpo_contract_teacher"},
    }


def test_mock_rollout_is_same_prompt_and_target_blind() -> None:
    module = _stage("17_idpo_rollout.py", "idpo_rollout_contract")
    config = _config()
    rollouts = module._build_rollout(_row(), config, None)
    assert len(rollouts) == 3
    for rollout in rollouts:
        assert len(rollout["responses"]) == 3
        assert rollout["minimum_responses_met"] is True
        assert _row()["target"] not in json.dumps(rollout, ensure_ascii=False)
        assert len({item["response_text"] for item in rollout["responses"]}) == 3


def test_judge_prompt_excludes_hidden_target() -> None:
    rollout_module = _stage("17_idpo_rollout.py", "idpo_rollout_prompt_contract")
    judge_module = _stage("18_idpo_teacher_judge.py", "idpo_judge_prompt_contract")
    rollout = rollout_module._build_rollout(_row(), _config(), None)[0]
    prompt = judge_module._prompt(rollout)
    assert _row()["target"] not in prompt
    assert "HIDDEN TITLE MUST NEVER ENTER IDPO PROMPTS" not in prompt


def test_judge_and_pair_conversion_preserve_shared_prompt() -> None:
    rollout_module = _stage("17_idpo_rollout.py", "idpo_rollout_pair_contract")
    judge_module = _stage("18_idpo_teacher_judge.py", "idpo_judge_pair_contract")
    pair_module = _stage("19_build_idpo_pairs.py", "idpo_pair_contract")
    config = _config()
    rollout = rollout_module._build_rollout(_row(), config, None)[0]
    judged = judge_module._judge_one(rollout, config)
    assert judged["judge_status"] == "accepted"
    config["idpo"]["round"] = 0
    # The pair builder operates on JSONL files; use its conversion logic through
    # a temporary file to ensure the exact model prompt is retained.
    root = Path(config["paths"]["idpo_dir"]) / "round_0"
    root.mkdir(parents=True, exist_ok=True)
    (root / "validation_judged.jsonl").write_text(
        json.dumps(judged, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = pair_module.build(config, "validation")
    assert report["pairs"] == 1
    pair = json.loads((root / "validation_pairs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert pair["prompt"] == rollout["prompt"]
    assert pair["chosen"] != pair["rejected"]


def test_gold_preference_uses_target_only_after_rollout() -> None:
    rollout_module = _stage("17_idpo_rollout.py", "idpo_gold_rollout_contract")
    gold_module = _stage("18_idpo_gold_score.py", "idpo_gold_score_contract")
    config = _config()
    config["idpo"].update(
        {
            "rollout_operations": ["mutation"],
            "rollout_mutation_parents": 1,
            "gold_reward_metric": "rouge_l",
            "minimum_reward_margin": 0.03,
        }
    )
    rollout = rollout_module._build_rollout(_row(), config, None)[0]
    # 构造确定的高低质量候选；Gold 只交给 scorer，不在 rollout 内出现。
    rollout["responses"][0]["output"] = "Completely Unrelated"
    rollout["responses"][1]["output"] = "Exact Gold Label"
    result = gold_module._score_one(rollout, "Exact Gold Label", config["idpo"])
    assert result["preference_status"] == "accepted"
    assert result["preference"]["chosen_id"] == rollout["responses"][1]["response_id"]
    assert result["preference"]["margin"] >= 0.03
    assert "Exact Gold Label" not in result["prompt"]


def test_gold_preference_filters_small_margin() -> None:
    rollout_module = _stage("17_idpo_rollout.py", "idpo_gold_margin_rollout")
    gold_module = _stage("18_idpo_gold_score.py", "idpo_gold_margin_contract")
    config = _config()
    config["idpo"].update(
        {
            "rollout_operations": ["mutation"],
            "rollout_mutation_parents": 1,
            "gold_reward_metric": "rouge_l",
            "minimum_reward_margin": 0.03,
        }
    )
    rollout = rollout_module._build_rollout(_row(), config, None)[0]
    for response in rollout["responses"]:
        response["output"] = "Same Candidate"
    result = gold_module._score_one(rollout, "Different Gold", config["idpo"])
    assert result["preference_status"] == "low_margin"
    assert result["preference"] is None


def test_trace_aware_rollout_preserves_first_stage_trace_and_hides_gold() -> None:
    module = _stage("17_idpo_rollout.py", "idpo_trace_rollout_contract")
    config = _config()
    config["sft_data"] = {"supervision_mode": "conditional_preference_trace"}
    config["idpo"].update(
        {
            "rollout_operations": ["mutation"],
            "rollout_mutation_parents": 1,
            "rollout_response_mode": "conditional_preference_trace",
        }
    )
    rollout = module._build_rollout(_row(), config, None)[0]
    assert rollout["trace_aware"] is True
    assert len(rollout["responses"]) == 3
    for response in rollout["responses"]:
        assert response["trace_text"].startswith('{"history_analysis"')
        assert '"preference"' in response["trace_text"]
        assert '"edit_plan"' in response["trace_text"]
        assert response["response_text"] == response["trace_text"] + response["output_text"]
        assert _row()["target"] not in response["response_text"]


def test_simple_trace_rollout_accepts_one_top8_evidence_and_hides_gold() -> None:
    module = _stage("17_idpo_rollout.py", "idpo_simple_trace_rollout_contract")
    config = _config()
    config["sft_data"] = {"supervision_mode": "simple_conditional_trace"}
    config["idpo"].update(
        {
            "rollout_operations": ["mutation"],
            "rollout_mutation_parents": 1,
            "rollout_response_mode": "simple_conditional_trace",
            "trace_maximum_history_records": 8,
        }
    )
    rollout = module._build_rollout(_row(), config, None)[0]
    assert rollout["trace_aware"] is True
    assert rollout["trace_mode"] == "simple"
    assert len(rollout["responses"]) == 3
    for response in rollout["responses"]:
        parsed = json.loads(response["response_text"])
        assert parsed["evidence_ids"] == ["h2"]
        assert set(parsed) == {
            "evidence_ids", "edit_reason", "edit_action", "output"
        }
        assert _row()["target"] not in response["response_text"]


def test_trace_output_sequence_weights_are_span_normalized() -> None:
    torch = pytest.importorskip("torch")
    module = _stage("20_train_user_editor_idpo.py", "idpo_trace_weight_contract")

    class Tokenizer:
        eos_token_id = 99
        pad_token_id = 0

        @staticmethod
        def encode(text, add_special_tokens=False):
            values = list(range(1, len(str(text).split()) + 1))
            return ([98] + values) if add_special_tokens else values

    pair = {
        "prompt": "prompt words",
        "chosen": "unused",
        "chosen_trace_text": "trace has four tokens",
        "chosen_output_text": "title two",
    }
    batch = module._sequence_batch(
        [pair], Tokenizer(), "chosen", 64, "trace_output", 0.2, 1.0
    )
    weights = batch["score_weights"][0]
    assert torch.isclose(weights.sum(), torch.tensor(1.2))
    assert int((weights > 0).sum()) == 7  # 4 Trace + 2 output + EOS


def test_trace_rollout_falls_back_as_one_output_only_group() -> None:
    module = _stage("17_idpo_rollout.py", "idpo_trace_fallback_contract")
    config = _config()
    config["sft_data"] = {"supervision_mode": "conditional_preference_trace"}
    config["idpo"].update(
        {
            "rollout_operations": ["mutation"],
            "rollout_mutation_parents": 1,
            "rollout_response_mode": "conditional_preference_trace",
            "trace_fallback_to_output_only": True,
        }
    )
    spec = module._build_operation_specs(_row(), config)[0]
    invalid = [({"output": f"Title {i}"}, f'{{"output":"Title {i}"}}', None) for i in range(3)]
    first = module._finalize_rollout(spec, invalid, config)
    assert first["minimum_responses_met"] is False
    fallback = module._fallback_spec(spec)
    second = module._finalize_rollout(fallback, invalid, config)
    assert second["minimum_responses_met"] is True
    assert second["trace_fallback"] is True
    assert all(not item["trace_text"] for item in second["responses"])


def test_ranker_adaptation_scores_seed_parent_with_loo_gold() -> None:
    module = _stage("idpo_common.py", "idpo_ranker_seed_score_contract")
    seed = {"candidate_id": "p1", "type": "task_seed", "text": "Exact Gold Title"}
    scored = module.seed_with_score(seed, "Exact Gold Title")
    assert scored["scores"] == {"rouge_1": 1.0, "rouge_l": 1.0}
    assert scored["candidate_id"] == seed["candidate_id"]
    assert "scores" not in seed


def test_dpo_loss_is_log_two_when_policy_equals_reference() -> None:
    torch = pytest.importorskip("torch")

    module = _stage("20_train_user_editor_idpo.py", "idpo_dpo_math_contract")
    values = torch.tensor([[-2.0, -3.0]])
    loss = module.dpo_loss(values[:, 0], values[:, 1], values[:, 0], values[:, 1], 0.1)
    assert abs(float(loss.item()) - 0.693147) < 1.0e-5


def test_sft_adapter_path_can_be_overridden_for_next_idpo_round() -> None:
    source = (HERE / "07_generate_editor_pool.py").read_text(encoding="utf-8")
    assert 'config.get("model", {}).get("adapter_path")' in source


def test_idpo_rollout_forces_stochastic_sampling(monkeypatch) -> None:
    module = _stage("17_idpo_rollout.py", "idpo_sampling_contract")
    captured = {}

    class FakeEditor:
        def __init__(self, config):
            captured.update(config["inference"])

    class FakeModule:
        LocalEditor = FakeEditor

    monkeypatch.setattr(module, "load_project_stage", lambda *args: FakeModule)
    config = {
        "idpo": {
            "policy_adapter_root": "",
            "rollout_samples": 4,
            "rollout_do_sample": True,
            "rollout_temperature": 0.75,
            "rollout_top_p": 0.9,
            "rollout_batch_size": 4,
        },
        "inference": {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "batch_size": 1,
        },
    }
    module._editor_for_user(config, "u1")
    assert captured["do_sample"] is True
    assert captured["temperature"] == 0.75
    assert captured["top_p"] == 0.9
    assert captured["batch_size"] == 4


def test_idpo_editor_batch_size_includes_query_batch(monkeypatch) -> None:
    module = _stage("17_idpo_rollout.py", "idpo_query_batch_contract")
    captured = {}

    class FakeEditor:
        def __init__(self, config):
            captured.update(config["inference"])

    class FakeModule:
        LocalEditor = FakeEditor

    monkeypatch.setattr(module, "load_project_stage", lambda *args: FakeModule)
    config = {
        "idpo": {
            "policy_adapter_root": "",
            "rollout_samples": 8,
            "rollout_do_sample": True,
            "rollout_temperature": 0.8,
            "rollout_top_p": 0.95,
            "rollout_batch_size": 8,
            "rollout_query_batch_size": 4,
        },
        "inference": {"batch_size": 1},
        "training": {},
        "model": {},
    }
    module._editor_for_user(config, "u1")
    assert captured["batch_size"] == 32


def test_idpo_pair_grouping_uses_real_user_id() -> None:
    module = _stage("idpo_common.py", "idpo_group_contract")
    rows = [
        {"pair_id": "u1-a", "user_id": "u1"},
        {"pair_id": "u1-b", "user_id": "u1"},
        {"pair_id": "u2-a", "user_id": "u2"},
    ]
    grouped = module.group_pairs_by_user(rows)
    assert sorted(grouped) == ["u1", "u2"]
    assert [item["pair_id"] for item in grouped["u1"]] == ["u1-a", "u1-b"]


def test_next_round_loads_adapter_by_user_id(monkeypatch, tmp_path) -> None:
    module = _stage("17_idpo_rollout.py", "idpo_round_one_contract")
    adapter = tmp_path / "round_0" / "user_adapters" / "user_u7"
    adapter.mkdir(parents=True)
    captured = {}

    class FakeEditor:
        def __init__(self, config):
            captured.update(config)

    class FakeModule:
        LocalEditor = FakeEditor

    monkeypatch.setattr(module, "load_project_stage", lambda *args: FakeModule)
    config = {
        "idpo": {
            "policy_adapter_root": str(tmp_path / "round_0" / "user_adapters"),
            "rollout_samples": 2,
            "rollout_do_sample": True,
        },
        "inference": {},
    }
    module._editor_for_user(config, "u7")
    assert captured["model"]["adapter_path"] == str(adapter)
