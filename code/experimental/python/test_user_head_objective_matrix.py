from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "user_head_objective_matrix", HERE / "24_compare_user_head_objectives.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _settings() -> dict:
    return {
        "pair_minimum_margin": 0.02,
        "max_pairs_per_group": 12,
        "label_temperature": 0.05,
        "score_temperature": 1.0,
        "regret_temperature": 1.0,
    }


def test_oracle_set_and_regret_reward_the_gold_top1() -> None:
    labels = torch.tensor([1.0, 0.4, 0.1])
    good = torch.tensor([4.0, 0.0, -1.0])
    bad = torch.tensor([-1.0, 0.0, 4.0])
    for objective in ("oracle_set", "expected_regret", "hybrid_top1"):
        assert MODULE.objective_loss(objective, good, labels, [3], _settings()) < MODULE.objective_loss(
            objective, bad, labels, [3], _settings()
        )


def test_user_split_is_deterministic_and_disjoint() -> None:
    rows = [{"sample_id": f"q{i}"} for i in range(100)]
    first = MODULE.split_user_records(rows, 0.2, 42, "u1")
    second = MODULE.split_user_records(rows, 0.2, 42, "u1")
    assert first == second
    assert len(first[0]) == 80
    assert len(first[1]) == 20
    assert {row["sample_id"] for row in first[0]}.isdisjoint(
        row["sample_id"] for row in first[1]
    )
