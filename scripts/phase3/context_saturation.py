#!/usr/bin/env python3
"""PersTSFM Phase 3.3C: Chronos-2 full-history pre-official context sweep.

Selection: historical lag-2 targets only. Later: lag-1 targets only in --score-later.
Uses only task.get_window(0).get_input_data(), NEVER official FEV test labels.
This is retrospective exploratory validation, not held-out family generalization.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

FEV_YAML = ("https://raw.githubusercontent.com/autogluon/fev/refs/tags/"
            "v0.10.0/benchmarks/fev_bench/tasks.yaml")
DEFAULT_CONTEXTS = (96, 384, 768, 1536, 3072, 4096, 6144, 8192)
OUTPUT = Path("results/phase3/context_saturation")
PRED_NAMES = ("point", "q10", "q50", "q90")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def timestamp_ns(stamps) -> np.ndarray:
    """Canonical int64 nanoseconds, regardless of pandas source datetime unit.

    Pandas DatetimeIndex.asi8 uses the index's own unit (e.g., us, not
    necessarily ns). Snapshot I/O and cadence checks must not assume ns.
    """
    index = pd.DatetimeIndex(stamps)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    ns = index.as_unit("ns").asi8
    if (ns == np.iinfo(np.int64).min).any():
        raise ValueError("NaT timestamp is not permitted")
    return ns


def sha256_histories(records: list[dict]) -> str:
    """Canonical identity of *pre-official inputs*, independent of datetime unit."""
    h = hashlib.sha256()
    for r in records:
        h.update(r["id"].encode())
        h.update(np.asarray(timestamp_ns(r["time"]), dtype="<i8").tobytes())
        h.update(np.asarray(r["y"], dtype="<f8").tobytes())
    return h.hexdigest()


def take_record(row, *, entity_key: str, time_key: str, target_key: str):
    eid = str(row[entity_key])
    values = np.asarray(row[target_key], dtype=np.float64)
    stamps = pd.DatetimeIndex(pd.to_datetime(row[time_key], utc=True)).tz_localize(None)
    if values.ndim != 1 or len(values) != len(stamps) or len(values) < 10:
        raise ValueError(f"{eid}: malformed or short series")
    stamps = stamps.as_unit("ns")
    ns = timestamp_ns(stamps)
    if not np.all(np.diff(ns) > 0):
        raise ValueError(f"{eid}: timestamps not strictly increasing")
    return dict(id=eid, y=values, time=stamps)


def case_windows(record: dict, horizon: int, context: int, *, reveal_lag1: bool = False) -> dict:
    """Strict chronological splits from ONLY pre-official input history.

      train: x[:-2H] (never scored)
      lag2: x[-2H-C:-2H] -> x[-2H:-H]  (selection)
      lag1: x[-H-C:-H]   -> x[-H:]      (future evaluation)
    """
    y, ts = record["y"], record["time"]
    if horizon < 1 or context < 2 or len(y) < context + 2 * horizon:
        raise ValueError(f"{record['id']}: insufficient history for C={context}, H={horizon}")
    out = {}
    for lag, end in ((2, len(y) - 2 * horizon), (1, len(y) - horizon)):
        xs, ys = slice(end-context, end), slice(end, end+horizon)
        past = y[xs]
        truth = y[ys] if (lag == 2 or reveal_lag1) else None
        if len(past) != context or not np.isfinite(past).all():
            raise ValueError(f"{record['id']}: incomplete/nonfinite context at lag{lag}, C={context}")
        if truth is not None and (len(truth) != horizon or not np.isfinite(truth).all()):
            raise ValueError(f"{record['id']}: incomplete/nonfinite historical target at lag{lag}")
        if not ts[xs][-1] < ts[ys][0]:
            raise ValueError(f"{record['id']}: chronological split failed")
        out[lag] = dict(values=past.astype(np.float32,copy=True), times=ts[xs],
                        truth=truth.astype(np.float64,copy=True) if truth is not None else None,
                        future_times=ts[ys])
    return out


def score(pred: np.ndarray, target: np.ndarray) -> dict:
    """pred shape (entities, horizon, 4), q: .1, .5, .9."""
    if pred.ndim != 3 or pred.shape[-1] != 4 or target.shape != pred.shape[:2]:
        raise ValueError("Prediction/target shape mismatch")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("Prediction/target nonfinite")
    mae_by_entity = np.abs(pred[:, :, 0] - target).mean(axis=1)
    q = np.array([.1, .5, .9])
    residual = target[..., None] - pred[:, :, 1:]
    pinball = np.maximum(q * residual, (q-1)*residual).mean(axis=(1,2))
    return {"mae": float(mae_by_entity.mean()), "pinball3": float(pinball.mean()),
            "entity_mae": mae_by_entity.tolist()}


def select_context(rows: list[dict]) -> int:
    if not rows:
        raise ValueError("No validation rows")
    if len(set(int(row["context"]) for row in rows)) != len(rows):
        raise ValueError("Repeated context results")
    # Favor shorter context in an exact tie.
    return min(rows, key=lambda r: (float(r["validation_mae"]), int(r["context"])))["context"]



def save_snapshot(records: list[dict], name: str, horizon: int, dest: Path) -> None:
    """Portable non-pickle FEV historical input snapshot; no official test labels."""
    if not records:
        raise ValueError("No historical records")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite snapshot: {dest}")
    lengths = [len(r['y']) for r in records]
    offsets = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
    all_y = np.concatenate([np.asarray(r['y'], dtype=np.float64) for r in records])
    all_ns = np.concatenate([timestamp_ns(r['time']) for r in records]).astype(np.int64)
    np.savez_compressed(dest, ids=np.array([r['id'] for r in records], dtype=np.str_),
                        offsets=offsets, values=all_y, timestamps_ns=all_ns,
                        dataset_config=np.array(name), horizon=np.array(horizon),
                        snapshot_schema=np.array("pertsfm_3_3c_history_ns_v2"),
                        timestamp_unit=np.array("ns"),
                        history_sha256=np.array(sha256_histories(records)))


def read_snapshot(path: Path):
    """Read FEV preofficial histories in Chronos model env, without importing fev."""
    with np.load(path, allow_pickle=False) as z:
        # Previous exports incorrectly wrote microseconds as `timestamps_ns`.
        # Refuse all unversioned snapshots rather than silently reading them as 1970 dates.
        if ('snapshot_schema' not in z.files or
            'timestamp_unit' not in z.files or
            str(z['snapshot_schema'].item()) != "pertsfm_3_3c_history_ns_v2" or
            str(z['timestamp_unit'].item()) != 'ns'):
            raise RuntimeError(
                "Legacy/unknown snapshot timestamp encoding. Re-export the full "
                "pre-official history to a NEW .npz filename with the patched script."
            )
        ids=z['ids'].astype(str).tolist()
        offsets=np.asarray(z['offsets'],dtype=np.int64)
        values=z['values']
        time_ns=z['timestamps_ns']
        task=str(z['dataset_config'].item())
        horizon=int(z['horizon'].item())
        expected_hash=str(z['history_sha256'].item())
    if (horizon < 1 or len(ids)==0 or len(set(ids))!=len(ids) or
        len(offsets)!=len(ids)+1 or offsets[0]!=0 or
        (np.diff(offsets)<=0).any() or offsets[-1]!=len(values) or
        len(values)!=len(time_ns)):
        raise RuntimeError('Corrupt / malformed full-history snapshot')
    records=[]
    for i,eid in enumerate(ids):
        lo,hi=int(offsets[i]),int(offsets[i+1])
        stamps=pd.DatetimeIndex(pd.to_datetime(time_ns[lo:hi],unit='ns',utc=True)).tz_localize(None)
        record=take_record({'id':eid,'ts':stamps,'y':values[lo:hi]},
                           entity_key='id',time_key='ts',target_key='y')
        records.append(record)
    if sha256_histories(records)!=expected_hash:
        raise RuntimeError('Snapshot historical content hash mismatch')
    if [r['id'] for r in records]!=sorted(ids):
        raise RuntimeError('Snapshot IDs not in deterministic order')
    return task,horizon,records


def find_task(name: str):
    import fev
    registry = fev.Benchmark.from_yaml(FEV_YAML)
    matching = [task for task in registry.tasks if task.dataset_config == name]
    if len(matching) != 1:
        raise ValueError(f"Expected unique dataset config '{name}' in pinned FEV, found {len(matching)}")
    task = matching[0]
    if not isinstance(task.target, str):
        raise TypeError("This experiment accepts one univariate target only")
    return task


def load_records(task, max_entities: int) -> list[dict]:
    """get_input_data reads ONLY data known before FEV official evaluation."""
    past, _future_known = task.get_window(0).get_input_data()
    rows = [take_record(row, entity_key=task.id_column,
                        time_key=task.timestamp_column, target_key=task.target)
            for row in past]
    if len({r['id'] for r in rows}) != len(rows):
        raise RuntimeError("Duplicated FEV entity identifiers")
    rows.sort(key=lambda r: r["id"])
    return rows[:max_entities] if max_entities else rows


def check_and_plan(records, horizon: int, contexts: list[int]) -> list[dict]:
    if not records:
        raise RuntimeError("No entities")
    if len(contexts) != len(set(contexts)) or not contexts or any(c < 2 or c > 8192 for c in contexts):
        raise ValueError("Contexts must be distinct integers in [2,8192]")
    if any(c > min(len(r['y']) - 2*horizon for r in records) for c in contexts):
        raise RuntimeError("Insufficient history for one of the context lengths on the matched cohort")
    for c in contexts:
        for r in records:
            case_windows(r,horizon,c)
    return [dict(context=int(c), entities=len(records), cases_per_lag=len(records)) for c in contexts]


def cadence_minutes(records) -> int:
    resolutions = set()
    for r in records:
        t = timestamp_ns(r['time'])
        diff = np.diff(t)
        if not np.all(diff == diff[0]) or diff[0] <= 0 or diff[0] % (60*10**9):
            raise RuntimeError(f"{r['id']}: chronology is irregular or not minute-aligned")
        resolutions.add(int(diff[0] // (60*10**9)))
    if len(resolutions) != 1:
        raise RuntimeError(f"Non-uniform cadence: {resolutions}")
    return resolutions.pop()


def infer(adapter, records, horizon, context, lag, batch_size, cadence):
    from src.models.adapters import ForecastInput
    import torch
    result = np.empty((len(records),horizon,4), dtype=np.float64)
    crossings = 0
    tic = time.perf_counter()
    for start in range(0, len(records), batch_size):
        subset=records[start:start+batch_size]
        windows=[case_windows(r,horizon,context)[lag] for r in subset]
        batch=[ForecastInput(case_id=r['id'],timestamps=w['times'],values=w['values'])
               for r,w in zip(subset,windows)]
        outputs=adapter.predict_batch(batch, horizon=horizon,
                                      resolution_minutes=cadence, context_steps=context,
                                      batch_size=batch_size)
        mapping={str(p.case_id):p for p in outputs}
        if len(mapping)!=len(batch) or set(mapping)!={r['id'] for r in subset}:
            raise RuntimeError("Model forecasts do not match entity identifiers")
        for j,r in enumerate(subset):
            p=mapping[r['id']]
            pred=np.asarray(p.prediction,dtype=float).reshape(-1)
            quant=np.stack([np.asarray(p.q10,dtype=float), np.asarray(p.q50,dtype=float),
                            np.asarray(p.q90,dtype=float)],axis=-1)
            if pred.shape!=(horizon,) or quant.shape!=(horizon,3) or not np.isfinite(pred).all() or not np.isfinite(quant).all():
                raise RuntimeError(f"Bad output shape/values for {r['id']}")
            crossings += int(np.any(np.diff(quant,axis=-1)<-1e-6,axis=1).sum())
            # Phase 2 diagnostic pinball3 policy: quantile rearrangement, point unchanged.
            result[start+j,:,0]=pred
            result[start+j,:,1:]=np.sort(quant,axis=-1)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return result, crossings, float(time.perf_counter()-tic)


def run(args):
    if args.batch_size < 1 or args.max_entities < 0:
        raise ValueError("Invalid batch size / entity limit")
    contexts=sorted(args.contexts)
    snapshot_path = Path(args.snapshot).resolve() if args.snapshot else None
    if snapshot_path is not None:
        taskname,horizon,records=read_snapshot(snapshot_path)
        if taskname!=args.task:
            raise RuntimeError(f"Snapshot task mismatch: {taskname} vs {args.task}")
        if args.max_entities:
            records=records[:args.max_entities]
    else:
        task=find_task(args.task)
        horizon=int(task.horizon)
        records=load_records(task,args.max_entities)
    grid=check_and_plan(records,horizon,contexts)
    hashes=sha256_histories(records)
    print("="*102)
    print("PERTSFM PHASE 3.3C | FULL-HISTORY CHRONOS-2 CONTEXT SATURATION | FEV PREOFFICIAL ONLY")
    print("="*102)
    print(f"Dataset={args.task} | H={horizon} | entities={len(records)} | context grid={contexts}")
    print(f"Full input histories: min={min(len(r['y']) for r in records)} max={max(len(r['y']) for r in records)}")
    print(f"Historical input SHA256: {hashes}")
    print("Lag2 selection; lag1 scores unavailable until --score-later. No official test labels accessed.")
    for g in grid:
        print(f"  c={g['context']:5d}: {g['entities']} entities | lag2+lag1 finite/chronological: PASS")
    if not args.execute:
        print("PASS: DRY RUN; no model loaded, no scores evaluated.")
        return
    import torch
    from src.models.adapters import Chronos2Model
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for consistent GPU benchmark")
    cadence=cadence_minutes(records)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dest=Path(args.output_root)/f"{stamp}_{args.task}_n{len(records)}"
    dest.mkdir(parents=True,exist_ok=False)
    manifest={"schema":"pertsfm_3_3c_v1", "task":args.task,"model":"amazon/chronos-2",
              "horizon":horizon,"contexts":contexts,"ids":[r['id'] for r in records],
              "entities":len(records), "cadence_minutes":cadence,"batch_size":args.batch_size,
              "history_sha256":hashes,"fev_registry":FEV_YAML,
              "snapshot_path":str(snapshot_path) if snapshot_path else None,
              "snapshot_sha256":sha256_file(snapshot_path) if snapshot_path else None,
              "protocol":"lag2 validation, lag1 future diagnostic, no official labels",
              "timestamp_utc":stamp, "gpu":torch.cuda.get_device_name(0)}
    (dest/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    adapter=Chronos2Model()
    if max(contexts)>adapter.pipeline.model_context_length:
        raise RuntimeError("Request exceeds loaded model's actual supported context")
    output=np.empty((len(contexts),2,len(records),horizon,4),dtype=np.float32)
    rows=[]
    for ci,c in enumerate(contexts):
        for li,lag in enumerate((2,1)):
            pred,cross,elapsed=infer(adapter,records,horizon,c,lag,args.batch_size,cadence)
            output[ci,li]=pred
            if lag==2:
                truth=np.stack([case_windows(r,horizon,c)[2]['truth'] for r in records])
                scored=score(pred,truth)
                rows.append(dict(context=c,validation_mae=scored['mae'],
                                 validation_pinball3=scored['pinball3'],
                                 quantile_crossing_steps=cross,
                                 lag2_inference_seconds=elapsed))
                print(f"c={c:5d} | lag2 MAE={scored['mae']:.5f} | pinball3={scored['pinball3']:.5f} | crossings={cross} | elapsed={elapsed:.2f}s",flush=True)
            else:
                print(f"c={c:5d} | lag1 forecasts SAVED, NOT SCORED | elapsed={elapsed:.2f}s",flush=True)
    # Persist immutable predictions and locked context before reading later targets.
    np.savez_compressed(dest/'predictions.npz',forecast=output, contexts=np.array(contexts,dtype=np.int32))
    df=pd.DataFrame(rows).sort_values('context')
    df.to_csv(dest/'validation.csv',index=False)
    chosen=int(select_context(rows))
    selection={"schema":"pertsfm_3_3c_selection_v1","chosen_context":chosen,
               "selection_metric":"lag2_entity_macro_mae",
               "predictions_sha256":sha256_file(dest/'predictions.npz'),
               "validation_sha256":sha256_file(dest/'validation.csv'),
               "manifest_sha256":sha256_file(dest/'manifest.json'),
               "locked_before_lag1_scoring":True}
    (dest/'selection.json').write_text(json.dumps(selection,indent=2)+"\n")
    print("\nVALIDATION ONLY:\n"+df.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print(f"LOCKED CONTEXT ON LAG2: {chosen}")
    print(f"PASS: {dest}")
    print(f"NEXT: python scripts/phase3/context_saturation.py --score-later {dest}")


def later(args):
    d=Path(args.score_later)
    selection=json.loads((d/'selection.json').read_text())
    manifest=json.loads((d/'manifest.json').read_text())
    if (selection.get('schema')!='pertsfm_3_3c_selection_v1' or
        manifest.get('schema')!='pertsfm_3_3c_v1' or
        selection.get('locked_before_lag1_scoring') is not True):
        raise RuntimeError("Missing/invalid locked selection")
    for filename,key in [('predictions.npz','predictions_sha256'),('validation.csv','validation_sha256'),('manifest.json','manifest_sha256')]:
        if sha256_file(d/filename)!=selection[key]:
            raise RuntimeError(f"Post-selection modifications: {filename}")
    if (d/'later.csv').exists():
        raise FileExistsError("Later scores already computed; refusing overwrite")
    if manifest.get('snapshot_path'):
        path=Path(manifest['snapshot_path'])
        if sha256_file(path)!=manifest['snapshot_sha256']:
            raise RuntimeError('Portable snapshot changed since selection')
        name,horizon,records=read_snapshot(path)
        if name!=manifest['task'] or horizon!=manifest['horizon']:
            raise RuntimeError('Snapshot task metadata mismatch')
    else:
        task=find_task(manifest['task'])
        records=load_records(task,0)
        horizon=int(task.horizon)
    wanted=set(manifest['ids'])
    records=[r for r in records if r['id'] in wanted]
    if ([r['id'] for r in records]!=manifest['ids'] or
        sha256_histories(records)!=manifest['history_sha256'] or
        horizon!=manifest['horizon']):
        raise RuntimeError("Historical cohort changed since locked selection")
    context_grid=manifest['contexts']
    with np.load(d/'predictions.npz',allow_pickle=False) as z:
        output=z['forecast']
        if z['contexts'].tolist()!=context_grid:
            raise RuntimeError("Predictions context grid changed")
    if output.shape!=(len(context_grid),2,len(records),int(horizon),4):
        raise RuntimeError("Forecast array shape mismatch")
    rows=[]
    for ci,c in enumerate(context_grid):
        truth=np.stack([case_windows(r,int(horizon),c,reveal_lag1=True)[1]['truth'] for r in records])
        scored=score(output[ci,1].astype(np.float64),truth)
        rows.append(dict(context=c,later_mae=scored['mae'],later_pinball3=scored['pinball3']))
    out=pd.DataFrame(rows).sort_values('context')
    out.to_csv(d/'later.csv',index=False)
    chosen=selection['chosen_context']
    selected=float(out.loc[out.context==chosen,'later_mae'].iloc[0])
    oracle=out.loc[out.later_mae.idxmin()]
    print("="*100)
    print("PERTSFM PHASE 3.3C | LAG1 HISTORICAL EVALUATION | LAG2 CHOICE FROZEN")
    print("="*100)
    print(out.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print(f"LOCKED SELECTION: C={chosen} | lag1 MAE={selected:.5f}")
    print(f"HINDSIGHT BEST (DIAGNOSTIC ONLY): C={int(oracle.context)} | MAE={oracle.later_mae:.5f}")
    print(f"PASS: {d}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',default='M_DENSE_1H')
    p.add_argument('--contexts',nargs='+',type=int,default=list(DEFAULT_CONTEXTS))
    p.add_argument('--max-entities',type=int,default=0, help='0: full matched cohort; >0: smoke subset')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--output-root',type=Path,default=OUTPUT)
    p.add_argument('--execute',action='store_true',help='Load Chronos-2 and run actual forecasts')
    p.add_argument('--score-later',type=Path,help='Score lag1 forecasts from a locked validation run')
    p.add_argument('--snapshot',type=Path,help='Preofficial full-history .npz; use in PERTSFM_ENV')
    p.add_argument('--export-snapshot',type=Path,help='Export snapshot from FEV in PERTSFM_DATA_ENV')
    args=p.parse_args()
    if args.export_snapshot:
        if args.snapshot or args.score_later or args.execute:
            raise ValueError('Export-only mode cannot combine with model execution or later scoring')
        task=find_task(args.task)
        records=load_records(task,0)
        save_snapshot(records,args.task,int(task.horizon),args.export_snapshot)
        print(f"PASS: preofficial full-history snapshot: {args.export_snapshot}")
        print(f"Entities={len(records)} | H={task.horizon} | SHA256={sha256_file(args.export_snapshot)}")
        print('Official FEV test labels were NOT accessed.')
    elif args.score_later:
        later(args)
    else:
        run(args)

if __name__=='__main__':
    main()
