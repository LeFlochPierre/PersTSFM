import pandas as pd
import pytest
from src.evaluation.case_sampling import balanced_subset


def frame():
    return pd.DataFrame([
        {"entity": ent, "eval_index": j, "eval_origin": f"2026-01-{j+1:02d}"}
        for ent in ["a", "b", "c", "d", "e"] for j in range(12)
    ])


def test_determinism_and_spread():
    p = frame()
    a = balanced_subset(p, "entity", 3, 4, seed=42)
    b = balanced_subset(p.sample(frac=1, random_state=123), "entity", 3, 4, seed=42)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 12
    assert a.entity.nunique() == 3
    for _, group in a.groupby("entity"):
        assert group.eval_index.tolist() == [0, 4, 7, 11]


def test_one_origin_most_recent():
    a = balanced_subset(frame(), "entity", 2, 1)
    assert len(a) == 2
    assert set(a.eval_index) == {11}


def test_no_truncation_when_small():
    a = balanced_subset(frame(), "entity", 10, 20)
    assert len(a) == 60


def test_invalid_counts():
    with pytest.raises(ValueError):
        balanced_subset(frame(), "entity", n_entities=0)
    with pytest.raises(ValueError):
        balanced_subset(frame(), "entity", origins_per_entity=0)
