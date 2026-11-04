"""Portable PRE-OFFICIAL time-series snapshots and causal historical windows.

Compatible with the corrected Phase 3.3C snapshot schema (nanosecond UTC).
No FEV dependency is imported unless export_fev_snapshot is explicitly called.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

SCHEMA = 'pertsfm_3_3c_history_ns_v2'
FEV_YAML = ('https://raw.githubusercontent.com/autogluon/fev/refs/tags/'
            'v0.10.0/benchmarks/fev_bench/tasks.yaml')

@dataclass(frozen=True)
class Series:
    id: str
    y: np.ndarray
    time: pd.DatetimeIndex

@dataclass(frozen=True)
class Snapshot:
    task: str
    horizon: int
    records: tuple[Series, ...]


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def timestamp_ns(stamps) -> np.ndarray:
    ix = pd.DatetimeIndex(stamps)
    if ix.tz is not None:
        ix = ix.tz_convert('UTC').tz_localize(None)
    arr = np.asarray(ix.as_unit('ns').asi8, dtype=np.int64)
    if np.any(arr == np.iinfo(np.int64).min):
        raise ValueError('NaT timestamp')
    return arr


def make_series(eid, values, stamps) -> Series:
    values = np.asarray(values, dtype=np.float64)
    ix = pd.DatetimeIndex(pd.to_datetime(stamps, utc=True)).tz_localize(None).as_unit('ns')
    t = timestamp_ns(ix)
    if values.ndim != 1 or len(values) != len(ix) or len(values) < 10:
        raise ValueError(f'{eid}: malformed series')
    if np.any(np.diff(t) <= 0):
        raise ValueError(f'{eid}: timestamps are not strictly increasing')
    return Series(str(eid), values, ix)


def history_hash(records) -> str:
    h = hashlib.sha256()
    for r in records:
        h.update(r.id.encode())
        h.update(np.asarray(timestamp_ns(r.time), dtype='<i8').tobytes())
        h.update(np.asarray(r.y, dtype='<f8').tobytes())
    return h.hexdigest()


def load_snapshot(path: Path) -> Snapshot:
    with np.load(path, allow_pickle=False) as z:
        if ('snapshot_schema' not in z.files or 'timestamp_unit' not in z.files or
            str(z['snapshot_schema'].item()) != SCHEMA or str(z['timestamp_unit'].item()) != 'ns'):
            raise RuntimeError('Unsupported/legacy timestamp snapshot. Re-export with corrected ns exporter.')
        ids = z['ids'].astype(str).tolist()
        offsets = np.asarray(z['offsets'], dtype=np.int64)
        vals = np.asarray(z['values'], dtype=np.float64)
        ns = np.asarray(z['timestamps_ns'], dtype=np.int64)
        task = str(z['dataset_config'].item())
        horizon = int(z['horizon'].item())
        saved_hash = str(z['history_sha256'].item())
    if (not ids or ids != sorted(ids) or len(ids) != len(set(ids)) or horizon < 1 or
        offsets.shape != (len(ids)+1,) or offsets[0] != 0 or
        np.any(np.diff(offsets) <= 0) or offsets[-1] != len(vals) or len(vals) != len(ns)):
        raise RuntimeError('Invalid/corrupt historical snapshot')
    records = []
    for i, eid in enumerate(ids):
        lo, hi = map(int, offsets[i:i+2])
        times = pd.to_datetime(ns[lo:hi], utc=True, unit='ns')
        records.append(make_series(eid, vals[lo:hi], times))
    if history_hash(records) != saved_hash:
        raise RuntimeError('Snapshot historical hash mismatch')
    return Snapshot(task, horizon, tuple(records))


def save_snapshot(snapshot: Snapshot, dest: Path):
    """Export only the historical input; never overwrite existing snapshots."""
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    if not snapshot.records or snapshot.horizon < 1:
        raise ValueError('Empty snapshot or bad horizon')
    dest.parent.mkdir(parents=True, exist_ok=True)
    lengths = [len(r.y) for r in snapshot.records]
    np.savez_compressed(
        dest, ids=np.asarray([r.id for r in snapshot.records], dtype=np.str_),
        offsets=np.r_[0, np.cumsum(lengths, dtype=np.int64)],
        values=np.concatenate([r.y for r in snapshot.records]),
        timestamps_ns=np.concatenate([timestamp_ns(r.time) for r in snapshot.records]),
        dataset_config=np.array(snapshot.task), horizon=np.array(snapshot.horizon),
        snapshot_schema=np.array(SCHEMA), timestamp_unit=np.array('ns'),
        history_sha256=np.array(history_hash(snapshot.records)),
    )


def export_fev_snapshot(task_name: str, dest: Path):
    """Call ONLY from PERTSFM_DATA_ENV. No calls to get_test_data or official labels."""
    import fev
    bench = fev.Benchmark.from_yaml(FEV_YAML)
    matched = [task for task in bench.tasks if task.dataset_config == task_name]
    if len(matched) != 1:
        raise ValueError(f'Expected one FEV task {task_name}, found {len(matched)}')
    task = matched[0]
    if not isinstance(task.target, str):
        raise TypeError('Only univariate FEV tasks supported')
    past, _future_known = task.get_window(0).get_input_data()
    records = [make_series(row[task.id_column], row[task.target], row[task.timestamp_column])
               for row in past]
    records.sort(key=lambda x: x.id)
    if len({r.id for r in records}) != len(records):
        raise RuntimeError('Duplicated entity IDs')
    snap = Snapshot(task_name, int(task.horizon), tuple(records))
    save_snapshot(snap, dest)
    return snap


def case_window(record: Series, horizon: int, context: int, lag: int, *, reveal_later=False):
    """lag=2: validation; lag=1: future diagnostic. lag1 truth withheld by default."""
    if lag not in (1, 2) or context < 2 or horizon < 1 or len(record.y) < context+2*horizon:
        raise ValueError(f'{record.id}: infeasible lag/context/horizon')
    end = len(record.y) - lag*horizon
    x = np.asarray(record.y[end-context:end], dtype=np.float32)
    target = record.y[end:end+horizon] if (lag == 2 or reveal_later) else None
    if len(x) != context or not np.isfinite(x).all():
        raise ValueError(f'{record.id}: invalid lag{lag} context at C={context}')
    if target is not None and (len(target) != horizon or not np.isfinite(target).all()):
        raise ValueError(f'{record.id}: invalid lag{lag} target at C={context}')
    if timestamp_ns(record.time)[end-1] >= timestamp_ns(record.time)[end]:
        raise ValueError(f'{record.id}: history overlaps target')
    return x, record.time[end-context:end], None if target is None else np.asarray(target, dtype=np.float64).copy()


def training_pools(records, horizon: int, history: int):
    """All training observations precede lag2 labels by construction."""
    pools = []
    for r in records:
        prefix = r.y[:-2*horizon]
        if history > len(prefix) or history < 1:
            raise ValueError(f'{r.id}: training history extends beyond strictly pre-lag2 pool')
        y = np.asarray(prefix[-history:], dtype=np.float32).copy()
        if not np.isfinite(y).all():
            raise ValueError(f'{r.id}: nonfinite LoRA training pool')
        pools.append(y)
    return pools


def cadence_minutes(records) -> int:
    vals = set()
    for r in records:
        diffs = np.diff(timestamp_ns(r.time))
        if not len(diffs) or np.any(diffs != diffs[0]) or diffs[0] <= 0 or diffs[0] % (60*10**9):
            raise ValueError(f'{r.id}: irregular timestamps; cannot infer fixed cadence')
        vals.add(int(diffs[0] // (60*10**9)))
    if len(vals) != 1:
        raise ValueError(f'Multiple sampling cadences: {vals}')
    return vals.pop()
