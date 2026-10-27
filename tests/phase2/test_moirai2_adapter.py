"""No-checkpoint unit tests; exercise adapter I/O and context changes."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.adapters import ForecastInput
from src.models.moirai2_adapter import Moirai2ForecastModel


class FakeModule:
    pass


class FakeForecast:
    builds = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.builds.append(kwargs)
        self.prediction_length = kwargs["prediction_length"]
        self.context_length = kwargs["context_length"]
        self.seen = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def predict(self, values):
        self.seen = values
        b = len(values)
        h = self.prediction_length
        return np.broadcast_to(np.arange(9)[None, :, None], (b, 9, h)).copy()


def adapter():
    model = Moirai2ForecastModel.__new__(Moirai2ForecastModel)
    model.model_id = "dummy"
    model.device = torch.device("cpu")
    model._forecast_cls = FakeForecast
    model.module = FakeModule()
    model.load_seconds = 0.0
    model._forecast = None
    model._shape = None
    model.batch_timings = []
    FakeForecast.builds.clear()
    return model


def case(key, v):
    return ForecastInput(
        case_id=key,
        timestamps=pd.date_range("2026-01-01", periods=len(v), freq="h"),
        values=np.asarray(v, dtype=np.float32),
    )


def test_shapes_quantiles_case_order():
    m = adapter()
    result = m.predict_batch(
        [case("a", [1, 2, 3]), case("b", [4, 5, 6])],
        horizon=4, resolution_minutes=60, context_steps=3, batch_size=2,
    )
    assert [x.case_id for x in result] == ["a", "b"]
    assert all(x.prediction.shape == (4,) for x in result)
    assert all(np.all(x.q10 == 0) and np.all(x.q50 == 4) and
               np.all(x.q90 == 8) for x in result)
    assert len(m.batch_timings) == 1


def test_context_rebuild_is_lazy():
    m = adapter()
    for size in (3, 3, 5):
        m.predict_batch([case("a", np.arange(size))],
                        horizon=2, resolution_minutes=60, context_steps=size,
                        batch_size=1)
    assert [x["context_length"] for x in FakeForecast.builds] == [3, 5]


def test_missing_history_preserved_for_internal_causal_imputation():
    m = adapter()
    m.predict_batch([case("a", [1, np.nan, 3])],
                    horizon=2, resolution_minutes=60, context_steps=3,
                    batch_size=1)
    assert np.isnan(m._forecast.seen[0][1])
    assert m.batch_timings[0]["has_missing_values"]


def test_invalid_histories():
    m = adapter()
    with pytest.raises(ValueError, match="no observed history"):
        m.predict_batch([case("a", [np.nan, np.nan])],
                        horizon=2, resolution_minutes=60, context_steps=2,
                        batch_size=1)
    with pytest.raises(ValueError, match="context length mismatch"):
        m.predict_batch([case("a", [1, 2])],
                        horizon=2, resolution_minutes=60, context_steps=3,
                        batch_size=1)
