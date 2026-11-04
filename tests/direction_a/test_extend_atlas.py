"""B1.2 offline invariants; no GPU or raw electricity data required."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.budgeted_forecasting.cache import FIELDS, load_cache
from src.budgeted_forecasting.extend_atlas import (
    _assert_matched_cache, _assert_matched_cases, _predict, action_name,
    make_shard_rows, merge, sha256,
)
from src.budgeted_forecasting.real_atlas import OriginCase


def _rows(action="seasonal_naive", gap=1., *, missing_case=False):
    data = []
    for split, day, series, offset in (
        ("calibration", 1, "e1", 1), ("calibration", 1, "e2", 2),
        ("evaluation", 4, "e1", 3), ("evaluation", 4, "e2", 4),
    ):
        if missing_case and split == "evaluation" and series == "e2":
            continue
        origin = f"2026-01-{day:02d}T00:00:00+00:00"
        available = f"2026-01-{day+1:02d}T00:00:00+00:00"
        for step in (1, 2):
            truth = float(offset + step)
            data.append(dict(dataset_id="electricity", series_id=series,
                             case_id=f"electricity::{series}::{origin}", split=split,
                             origin_time=origin, label_available_at=available,
                             action_id=action, step=step, truth=truth,
                             prediction=truth + gap, latency_ms=1. if action == "seasonal_naive" else 10.))
    return data


def _save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _shard(base, path, rows, action):
    _save(path, rows)
    manifest = dict(schema_version="direction_a_shard_v1", action_id=action,
                    base_cache_sha256=sha256(base), shard_sha256=sha256(path))
    path.with_name(path.stem + "_manifest.json").write_text(json.dumps(manifest))
    return path


def test_valid_action_names():
    assert action_name("tirex2", 96) == "tirex2_4d"
    assert action_name("moirai2_small", 336) == "moirai2_small_14d"


@pytest.mark.parametrize("model,hours", [("unknown", 96), ("chronos2", 12), ("tirex2", -1)])
def test_reject_unknown_action(model, hours):
    with pytest.raises(ValueError):
        action_name(model, hours)


def test_merge_success(tmp_path):
    base = _save(tmp_path / "base.csv", _rows())
    new = _shard(base, tmp_path / "tirex2.csv", _rows("tirex2_4d", .4), "tirex2_4d")
    report = merge(base, [new], tmp_path / "joined")
    cases = load_cache(tmp_path / "joined/forecast_cache.csv")
    assert len(cases) == 4
    assert set(cases[0].actions) == {"tirex2_4d", "seasonal_naive"}
    assert len(report["budgets"]) == 2
    assert (tmp_path / "joined/decisions.csv").is_file()
    # Neither the original cache nor its shard has been modified.
    assert sha256(base) == json.loads((tmp_path / "tirex2_manifest.json").read_text())["base_cache_sha256"]


def test_merge_rejects_wrong_truth(tmp_path):
    base = _save(tmp_path / "base.csv", _rows())
    wrong = _rows("tirex2_4d")
    wrong[0]["truth"] += 1.
    shard = _shard(base, tmp_path / "shard.csv", wrong, "tirex2_4d")
    with pytest.raises(ValueError, match="truth"):
        merge(base, [shard], tmp_path / "out")


def test_merge_rejects_missing_case(tmp_path):
    base = _save(tmp_path / "base.csv", _rows())
    shard = _shard(base, tmp_path / "shard.csv", _rows("tirex2_4d", missing_case=True), "tirex2_4d")
    with pytest.raises(ValueError, match="case IDs"):
        merge(base, [shard], tmp_path / "out")


def test_merge_rejects_tampered_shard(tmp_path):
    base = _save(tmp_path / "base.csv", _rows())
    shard = _shard(base, tmp_path / "shard.csv", _rows("tirex2_4d"), "tirex2_4d")
    shard.write_text(shard.read_text() + "tampered\n")
    with pytest.raises(ValueError, match="provenance or checksum"):
        merge(base, [shard], tmp_path / "out")


def test_merge_rejects_duplicate_action(tmp_path):
    base = _save(tmp_path / "base.csv", _rows())
    shard = _shard(base, tmp_path / "shard.csv", _rows("seasonal_naive"), "seasonal_naive")
    with pytest.raises(ValueError, match="Duplicate action"):
        merge(base, [shard], tmp_path / "out")


def test_merge_refuses_overwrite(tmp_path):
    base = _save(tmp_path / "base.csv", _rows())
    shard = _shard(base, tmp_path / "shard.csv", _rows("tirex2_4d"), "tirex2_4d")
    merge(base, [shard], tmp_path / "out")
    with pytest.raises(FileExistsError, match="overwrite"):
        merge(base, [shard], tmp_path / "out")


def _real_case():
    idx = pd.date_range("2026-01-01", periods=384, freq="15min", tz="UTC")
    origin = idx[-1]
    return OriginCase(case_id="electricity::e1::2026-01-04T23:45:00+00:00",
                      entity_id="e1", split="calibration", origin=origin,
                      label_available_at=origin + pd.Timedelta(hours=24),
                      context=pd.Series(np.ones(384), index=idx),
                      truth=np.ones(96))


@dataclass
class Input:
    case_id: str
    timestamps: object
    values: object


class Model:
    def predict_batch(self, inputs, horizon, resolution_minutes, context_steps, batch_size):
        assert len(inputs) == 1 and len(inputs[0].values) == 384
        assert horizon == 96 and resolution_minutes == 15
        assert context_steps == 384 and batch_size == 1
        return [SimpleNamespace(case_id=inputs[0].case_id, prediction=np.ones(horizon))]


def test_action_forecasting_mock_model():
    case = _real_case()
    pred, duration = _predict(Model(), case, 96, Input, lambda: None)
    assert len(pred) == 96 and duration > 0
    rows, costs = make_shard_rows([case], Model(), 96, "tirex2", Input, lambda: None)
    assert len(rows) == 96 and len(costs) == 1
    assert rows[0]["action_id"] == "tirex2_4d"
    assert all(x["truth"] == x["prediction"] for x in rows)


def test_context_mismatch_rejected():
    with pytest.raises(ValueError, match="Invalid context"):
        _predict(Model(), _real_case(), 336, Input, lambda: None)


def test_bad_response_case_id_rejected():
    class Bad(Model):
        def predict_batch(self, *args, **kwargs):
            return [SimpleNamespace(case_id="wrong", prediction=np.ones(96))]
    with pytest.raises(RuntimeError, match="mismatch"):
        _predict(Bad(), _real_case(), 96, Input, lambda: None)


def test_origin_split_check(tmp_path):
    base = load_cache(_save(tmp_path / "base.csv", _rows()))
    # This case tests the exact-match invariant, without loading the real data.
    wrong = _real_case()
    with pytest.raises(ValueError, match="Case mismatch"):
        _assert_matched_cases(base, [wrong])
