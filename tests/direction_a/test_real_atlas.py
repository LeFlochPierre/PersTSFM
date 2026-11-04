"""CPU-only contract and integration checks for Direction A B1.1."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.budgeted_forecasting.cache import FIELDS, load_cache
from src.budgeted_forecasting.real_atlas import (
    OriginCase, auto_budgets, build_cache_rows, seasonal_naive,
    select_origins, write_csv_atomic,
)


def _protocol(n=12, entities=("A", "B"), base="2026-01-01"):
    records = []
    for entity in entities:
        for i in range(n):
            origin = pd.Timestamp(base) + pd.Timedelta(days=i)
            records.append({"entity": entity, "eval_index": i,
                            "eval_origin": origin,
                            "target_end": origin + pd.Timedelta(hours=23, minutes=45)})
    return pd.DataFrame(records)


def test_temporal_split_and_pooled_embargo():
    result = select_origins(_protocol(), "entity", entities=2,
                            calibration_origins=3, evaluation_origins=3, seed=42)
    assert len(result) == 12
    cal = result[result.split == "calibration"]
    ev = result[result.split == "evaluation"]
    assert cal.target_end.max() < ev.eval_origin.min()
    assert set(cal.entity) == set(ev.entity) == {"A", "B"}


def test_entity_sampling_is_seed_reproducible():
    protocol = _protocol(12, ("A", "B", "C", "D"))
    x = select_origins(protocol, "entity", entities=2, calibration_origins=2,
                       evaluation_origins=2, seed=11)
    y = select_origins(protocol, "entity", entities=2, calibration_origins=2,
                       evaluation_origins=2, seed=11)
    pd.testing.assert_frame_equal(x, y)


@pytest.mark.parametrize("entities,cal,ev", [(0, 2, 2), (1, 0, 2), (1, 2, 0)])
def test_invalid_selection(entities, cal, ev):
    with pytest.raises(ValueError):
        select_origins(_protocol(), "entity", entities=entities,
                       calibration_origins=cal, evaluation_origins=ev, seed=1)


def test_insufficient_history_dates():
    with pytest.raises(ValueError, match="distinct origin"):
        select_origins(_protocol(3), "entity", entities=1,
                       calibration_origins=2, evaluation_origins=2, seed=1)


def test_missing_entity_in_one_block_rejected():
    protocol = _protocol(12)
    protocol = protocol[~((protocol.entity == "B") & (protocol.eval_index >= 9))]
    with pytest.raises(ValueError, match="missing calibration/evaluation"):
        select_origins(protocol, "entity", entities=2,
                       calibration_origins=2, evaluation_origins=2, seed=1)


def test_duplicate_origins_rejected():
    protocol = pd.concat((_protocol(), _protocol().iloc[:1]), ignore_index=True)
    with pytest.raises(ValueError, match="Repeated evaluation origin"):
        select_origins(protocol, "entity", entities=2,
                       calibration_origins=2, evaluation_origins=2, seed=1)


def test_seasonal_naive_preserves_daily_pattern():
    history = np.arange(384, dtype=float)
    pred = seasonal_naive(history, 96)
    np.testing.assert_array_equal(pred, np.arange(288, 384))
    np.testing.assert_array_equal(seasonal_naive(history, 100)[96:], np.arange(288, 292))


def test_seasonal_naive_too_short():
    with pytest.raises(ValueError):
        seasonal_naive(np.arange(32), 96)


@dataclass
class StubForecastInput:
    case_id: str
    timestamps: pd.DatetimeIndex
    values: np.ndarray


class StubModel:
    def __init__(self):
        self.calls = []

    def predict_batch(self, inputs, horizon, resolution_minutes, context_steps, batch_size):
        assert len(inputs) == 1 and batch_size == 1 and resolution_minutes == 15
        assert len(inputs[0].values) == context_steps
        self.calls.append((inputs[0].case_id, context_steps))
        return [SimpleNamespace(case_id=inputs[0].case_id,
                                prediction=np.full(horizon, np.mean(inputs[0].values)))]


def _cases():
    origin_base = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    ret = []
    for i, split in enumerate(("calibration", "calibration", "evaluation", "evaluation")):
        origin = origin_base + pd.Timedelta(days=5 * i)
        index = pd.date_range(end=origin.tz_localize(None), periods=14 * 96, freq="15min")
        ret.append(OriginCase(case_id=f"elec::{i}", entity_id="A", split=split,
                              origin=origin, label_available_at=origin + pd.Timedelta(days=1),
                              context=pd.Series(np.arange(len(index), dtype=float) + i, index=index),
                              truth=np.full(96, 8.0 + i)))
    return ret


def test_end_to_end_stub_model_and_b0_cache(tmp_path):
    model = StubModel()
    rows, costs = build_cache_rows(_cases(), model, forecast_input_factory=StubForecastInput)
    assert len(rows) == 4 * 3 * 96
    assert len(costs) == 4 * 3
    assert len(model.calls) == 4 * 2
    assert {n for _, n in model.calls} == {4 * 96, 14 * 96}
    path = tmp_path / "cache.csv"
    write_csv_atomic(rows, path, FIELDS)
    validated = load_cache(path)
    assert len(validated) == 4
    assert len(auto_budgets(validated)) >= 1
    with pytest.raises(FileExistsError):
        write_csv_atomic(rows, path, FIELDS)


def test_bad_model_output_is_rejected():
    class BadModel(StubModel):
        def predict_batch(self, inputs, horizon, resolution_minutes, context_steps, batch_size):
            return [SimpleNamespace(case_id=inputs[0].case_id,
                                    prediction=np.full(horizon - 1, 1.0))]
    with pytest.raises(ValueError, match="Invalid chronos2"):
        build_cache_rows(_cases(), BadModel(), forecast_input_factory=StubForecastInput)


def test_bad_model_case_id_is_rejected():
    class BadModel(StubModel):
        def predict_batch(self, inputs, horizon, resolution_minutes, context_steps, batch_size):
            return [SimpleNamespace(case_id="WRONG", prediction=np.ones(horizon))]
    with pytest.raises(RuntimeError, match="Adapter response mismatch"):
        build_cache_rows(_cases(), BadModel(), forecast_input_factory=StubForecastInput)


def test_prepare_cases_uses_only_past_and_validates_grid(monkeypatch):
    import sys
    import types
    from src.budgeted_forecasting.real_atlas import prepare_cases

    fake_parent = types.ModuleType("src.data")
    fake_parent.__path__ = []
    fake_registry = types.ModuleType("src.data.registry")
    fake_registry.context_steps = lambda hours, resolution: int(hours * 60 / resolution)

    def fake_extract(loaded, row, hours):
        assert hours == 336
        origin = row["eval_origin"]
        context_index = pd.date_range(end=origin, periods=336 * 4, freq="15min")
        target_index = pd.date_range(start=origin + pd.Timedelta(minutes=15),
                                     periods=96, freq="15min")
        return {"case_id": "fake", "entity_id": "A", "eval_origin": origin,
                "context": pd.Series(np.ones(len(context_index)), index=context_index),
                "truth": pd.Series(np.ones(96), index=target_index)}

    fake_registry.extract_case = fake_extract
    monkeypatch.setitem(sys.modules, "src.data", fake_parent)
    monkeypatch.setitem(sys.modules, "src.data.registry", fake_registry)
    fake = SimpleNamespace(spec=SimpleNamespace(key="electricity", resolution_minutes=15,
                                                forecast_steps=96))
    selected = pd.DataFrame([
        {"eval_origin": pd.Timestamp("2026-02-01"), "split": "calibration"},
        {"eval_origin": pd.Timestamp("2026-02-04"), "split": "evaluation"},
    ])
    cases = prepare_cases(fake, selected)
    assert len(cases) == 2
    assert cases[0].label_available_at <= cases[1].origin
    assert cases[0].origin.tzinfo is not None
    assert cases[0].context.index[-1] == selected.iloc[0].eval_origin
