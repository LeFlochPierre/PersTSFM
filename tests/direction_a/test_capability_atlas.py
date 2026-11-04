"""B3.0 regression tests: frozen cohort integrity, complete matched actions, leakage guards."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.budgeted_forecasting import capability_atlas as b3


@dataclass
class DummySpec:
    key: str
    id_col: str
    protocol_path: Path
    raw_path: Path
    loader_kind: str = "dummy"
    resolution_minutes: int = 60
    forecast_steps: int = 2
    context_hours: tuple = (24, 48)


@pytest.fixture
def environment(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol.csv"
    raw = tmp_path / "data.csv"
    raw.write_text("dummy data marker\n")
    times = pd.to_datetime(["2026-01-04", "2026-01-05", "2026-01-11", "2026-01-12"])
    frame = pd.DataFrame([
        dict(entity=e, eval_origin=t, target_start=t+pd.Timedelta(hours=1),
             target_end=t+pd.Timedelta(hours=2), split="validation", eval_index=i)
        for e in ("A", "B") for i,t in enumerate(times)
    ])
    frame.to_csv(protocol, index=False)
    index = pd.date_range("2026-01-01", periods=24 * 20, freq="h")
    data = {e: pd.Series(np.arange(len(index), dtype=float) + (20 if e == "B" else 0), index=index)
            for e in ("A", "B")}
    spec = DummySpec("dummy", "entity", protocol, raw)

    def load_protocol(_spec, split):
        assert split == "validation"
        df = pd.read_csv(protocol, parse_dates=["eval_origin", "target_start", "target_end"])
        return df[df["split"] == split].copy()

    def extract_case(loaded, row, hours):
        origin = pd.Timestamp(row["eval_origin"])
        series = loaded.series[row["entity"]]
        n = int(hours)
        return dict(case_id=f"{row['entity']}::{origin}", entity_id=row["entity"],
                    eval_origin=origin, eval_index=int(row["eval_index"]),
                    context=series.loc[origin - pd.Timedelta(hours=n-1):origin],
                    truth=series.loc[origin+pd.Timedelta(hours=1):origin+pd.Timedelta(hours=2)])

    dummy = SimpleNamespace(get_dataset_spec=lambda k: spec if k == "dummy" else None,
                            load_protocol=load_protocol, extract_case=extract_case,
                            LoadedDataset=lambda **kwargs: SimpleNamespace(**kwargs),
                            _load_dummy=lambda _spec, ids: {k:data[k] for k in ids})
    monkeypatch.setattr(b3, "_registry", lambda: dummy)
    return tmp_path, frame, data


def freeze(env, hours=48):
    root, _, _ = env
    path = root / "frozen" / "cohort.json"
    b3.freeze_cohort("dummy", "validation", 2, 2, 2, hours, 12, path.parent)
    return path


def test_action_names():
    assert b3.action_name("chronos2", 96) == "chronos2_4d"
    assert b3.action_name("toto2_22m", 0.5) == "toto2_22m_0p5h"
    assert b3.action_name("seasonal_naive") == "seasonal_naive"
    with pytest.raises(ValueError):
        b3.action_name("not_a_model", 24)


def test_context_rejects_noninteger_steps():
    with pytest.raises(ValueError, match="not divisible"):
        b3.context_steps(1.1, 60)
    assert b3.context_steps(96, 15) == 384


def test_freeze_reconstruct_and_temporal_split(environment):
    path = freeze(environment)
    meta, loaded = b3.load_frozen_cases(path)
    assert len(loaded) == 8
    assert len({x[0]["case_id"] for x in loaded}) == 8
    assert max(x["label_available_at"] for x in meta["cases"] if x["split"] == "calibration") < min(
        x["origin_time"] for x in meta["cases"] if x["split"] == "evaluation")
    assert all(len(y) == 2 and len(ctx) == 48 for _,ctx,y in loaded)


def test_freeze_no_overwrite(environment):
    freeze(environment)
    with pytest.raises(FileExistsError):
        freeze(environment)


def test_mutated_truth_rejected(environment):
    path = freeze(environment)
    root, _, data = environment
    data["A"].loc[pd.Timestamp("2026-01-12 01:00")] += 1
    with pytest.raises(ValueError, match="truth_sha256"):
        b3.load_frozen_cases(path)


def test_mutated_history_rejected(environment):
    path = freeze(environment)
    root, _, data = environment
    data["B"].loc[pd.Timestamp("2026-01-11 00:00")] += 9
    with pytest.raises(ValueError, match="context_sha256"):
        b3.load_frozen_cases(path)


def test_changed_protocol_rejected(environment):
    path = freeze(environment)
    root, *_ = environment
    (root / "protocol.csv").write_text("tampered\n")
    with pytest.raises(ValueError, match="Protocol changed"):
        b3.load_frozen_cases(path)


def test_duplicate_frozen_case_rejected(environment):
    path = freeze(environment)
    cohort = json.loads(path.read_text())
    cohort["cases"].append(cohort["cases"][0])
    path.write_text(json.dumps(cohort))
    with pytest.raises(ValueError, match="Duplicate case"):
        b3.load_frozen_cases(path)


def test_models_run_and_report(environment):
    path = freeze(environment)
    root, *_ = environment
    a = root / "shards" / "last_value.csv"
    b = root / "shards" / "seasonal_naive.csv"
    b3.run_single_action(path, "last_value", None, a)
    b3.run_single_action(path, "seasonal_naive", None, b)
    out = root / "report"
    result = b3.analyze_shards(path, [a,b], out)
    assert result["case_count"] == 8
    assert len(result["actions"]) == 2
    assert len(pd.read_csv(out / "forecast_cache.csv")) == 8 * 2 * 2
    assert len(pd.read_csv(out / "per_case.csv")) == 8 * 2
    assert result["evaluation"]["last_value"]["cases"] == 4


def test_dry_run_does_not_create_outputs(environment):
    path = freeze(environment)
    root, *_ = environment
    shard = root / "shards" / "baseline.csv"
    b3.run_single_action(path, "last_value", None, shard, dry_run=True)
    assert not shard.exists()


def test_run_action_no_overwrite(environment):
    path = freeze(environment)
    root, *_ = environment
    shard = root / "shards" / "baseline.csv"
    b3.run_single_action(path, "last_value", None, shard)
    with pytest.raises(FileExistsError):
        b3.run_single_action(path, "last_value", None, shard)


def test_model_context_exceeds_frozen(environment):
    path = freeze(environment, 24)
    root, *_ = environment
    with pytest.raises(ValueError, match="exceeds frozen maximum"):
        b3.run_single_action(path, "chronos2", 48, root / "no.csv", dry_run=True)


def test_missing_shard_rejected(environment):
    path = freeze(environment)
    root, *_ = environment
    with pytest.raises(FileNotFoundError, match="Missing action manifest"):
        b3.analyze_shards(path, [root / "missing.csv"], root / "out")


def test_tampered_shard_rejected(environment):
    path = freeze(environment)
    root, *_ = environment
    shard = root / "shards" / "last_value.csv"
    b3.run_single_action(path, "last_value", None, shard)
    shard.write_text(shard.read_text() + "tamper\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        b3.analyze_shards(path, [shard], root / "out")


def test_duplicate_action_rejected(environment):
    path = freeze(environment)
    root, *_ = environment
    shard = root / "shards" / "last_value.csv"
    b3.run_single_action(path, "last_value", None, shard)
    with pytest.raises(ValueError, match="Duplicate model/context"):
        b3.analyze_shards(path, [shard, shard], root / "out")


def test_report_no_overwrite(environment):
    path = freeze(environment)
    root, *_ = environment
    shard = root / "shards" / "last_value.csv"
    b3.run_single_action(path, "last_value", None, shard)
    out = root / "out"
    b3.analyze_shards(path, [shard], out)
    with pytest.raises(FileExistsError):
        b3.analyze_shards(path, [shard], out)


def test_aggregate_reports_normalized(environment):
    path = freeze(environment)
    root, *_ = environment
    a = root / "shards" / "last_value.csv"
    b = root / "shards" / "seasonal_naive.csv"
    b3.run_single_action(path, "last_value", None, a)
    b3.run_single_action(path, "seasonal_naive", None, b)
    output = root / "report"
    b3.analyze_shards(path, [a,b], output)
    result = b3.aggregate_reports([output / "report.json"], root / "aggregate")
    assert len(result) == 2
    season = next(x for x in result if x["action_id"] == "seasonal_naive")
    assert season["relative_mae_to_reference"] == 1.0
    assert (root / "aggregate" / "cross_dataset.csv").is_file()


def test_aggregate_rejects_duplicate_datasets(environment):
    path = freeze(environment)
    root, *_ = environment
    a = root / "shards" / "seasonal_naive.csv"
    b3.run_single_action(path, "seasonal_naive", None, a)
    output = root / "report"
    b3.analyze_shards(path, [a], output)
    with pytest.raises(ValueError, match="multiple reports"):
        b3.aggregate_reports([output / "report.json", output / "report.json"], root / "aggregate")
