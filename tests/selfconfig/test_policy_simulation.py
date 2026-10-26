import pandas as pd

from src.selfconfig.policy_simulation import (
    simulate_v11,
    simulate_v12,
    three_anchors,
)


def _frames():
    proposed = pd.DataFrame(
        [
            {"model": "a", "context_steps": 10, "context_hours": 1.0, "mean_entity_mae": 5.0},
            {"model": "a", "context_steps": 20, "context_hours": 2.0, "mean_entity_mae": 4.0},
            {"model": "a", "context_steps": 40, "context_hours": 4.0, "mean_entity_mae": 3.0},
            {"model": "b", "context_steps": 10, "context_hours": 1.0, "mean_entity_mae": 5.2},
            {"model": "b", "context_steps": 20, "context_hours": 2.0, "mean_entity_mae": 4.1},
            {"model": "b", "context_steps": 40, "context_hours": 4.0, "mean_entity_mae": 3.03},
        ]
    )

    full = pd.DataFrame(
        [
            {"model": "a", "context_steps": 10, "context_hours": 1.0, "mean_entity_mae": 5.0},
            {"model": "a", "context_steps": 20, "context_hours": 2.0, "mean_entity_mae": 4.0},
            {"model": "a", "context_steps": 40, "context_hours": 4.0, "mean_entity_mae": 3.0},
            {"model": "a", "context_steps": 80, "context_hours": 8.0, "mean_entity_mae": 3.1},
            {"model": "b", "context_steps": 10, "context_hours": 1.0, "mean_entity_mae": 5.2},
            {"model": "b", "context_steps": 20, "context_hours": 2.0, "mean_entity_mae": 4.1},
            {"model": "b", "context_steps": 40, "context_hours": 4.0, "mean_entity_mae": 3.03},
            {"model": "b", "context_steps": 80, "context_hours": 8.0, "mean_entity_mae": 2.8},
        ]
    )
    return proposed, full


def test_three_anchors_adds_capacity():
    proposed, full = _frames()
    a = three_anchors(proposed, full, "a")
    assert a["short"] == 10
    assert a["structural"] == 40
    assert a["capacity"] == 80
    assert a["selected"] == [10, 40, 80]


def test_v11_can_recover_capacity_winner():
    proposed, full = _frames()
    evaluated, best, _ = simulate_v11(proposed, full)
    assert len(evaluated) == 6
    assert best["model"] == "b"
    assert int(best["context_steps"]) == 80


def test_v12_extends_competitive_models_only():
    proposed, full = _frames()
    evaluated, best, _, competitive = simulate_v12(
        proposed,
        full,
        tie_band=0.02,
    )
    assert competitive == ["a", "b"]
    assert len(evaluated) == 6
    assert best["model"] == "b"
    assert int(best["context_steps"]) == 80
