"""Fast offline adapter contract checks; no checkpoint download required."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.adapters import ForecastInput
from src.models.toto2_adapter import Toto2ForecastModel


class FakeModel:
    def __init__(self):
        self.last = None

    def forecast(self, data, horizon, decode_block_size, has_missing_values):
        self.last = (data, horizon, decode_block_size, has_missing_values)
        b = data["target"].shape[0]
        # Expected quantile order: 0.1, 0.2, ... 0.9.
        return torch.arange(9, dtype=torch.float32).view(9, 1, 1, 1).expand(9, b, 1, horizon)


def stub():
    adapter = Toto2ForecastModel.__new__(Toto2ForecastModel)
    adapter.name = "toto2_22m"
    adapter.model_id = "Datadog/Toto-2.0-22m"
    adapter.device = torch.device("cpu")
    adapter.model = FakeModel()
    adapter.batch_timings = []
    return adapter


def case(key, values):
    return ForecastInput(
        case_id=key,
        timestamps=pd.date_range("2026-01-01", periods=len(values), freq="h"),
        values=np.array(values, dtype=np.float32),
    )


def test_quantile_mapping_and_case_order():
    m = stub()
    out = m.predict_batch([case("a", [1, 2, 3]), case("b", [4, 5, 6])],
                          horizon=4, resolution_minutes=60, context_steps=3, batch_size=2)
    assert [x.case_id for x in out] == ["a", "b"]
    assert all(x.prediction.shape == (4,) for x in out)
    assert all(np.all(x.q10 == 0) and np.all(x.q50 == 4)
               and np.all(x.q90 == 8) and np.all(x.prediction == 4) for x in out)
    assert m.model.last[2] is None
    assert m.model.last[3] is True  # 3 real observations, 29 masked left-pad
    assert tuple(m.model.last[0]["target"].shape) == (2, 1, 32)
    assert len(m.batch_timings) == 1


def test_mask_missing_values_and_zero_fill():
    m = stub()
    m.predict_batch([case("a", [1, np.nan, 3])],
                    horizon=2, resolution_minutes=60, context_steps=3, batch_size=1)
    data = m.model.last[0]
    assert m.model.last[3] is True
    assert not bool(data["target_mask"][0, 0, 1])
    assert float(data["target"][0, 0, 1]) == 0.0


def test_no_history_rejected():
    m = stub()
    with pytest.raises(ValueError, match="no observed history"):
        m.predict_batch([case("a", [np.nan, np.nan])],
                        horizon=2, resolution_minutes=60, context_steps=2, batch_size=1)


def test_context_mismatch_rejected():
    m = stub()
    with pytest.raises(ValueError, match="context length"):
        m.predict_batch([case("a", [1, 2])],
                        horizon=2, resolution_minutes=60, context_steps=3, batch_size=1)
