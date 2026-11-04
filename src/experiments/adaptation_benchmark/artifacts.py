"""Replayable, checksummed trial-cache, result locks and no-overwrite writers."""
from __future__ import annotations
from pathlib import Path
import json
import os
import numpy as np
from .history import file_hash


def write_json(path: Path, item: dict):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    if temp.exists():
        raise FileExistsError(temp)
    try:
        temp.write_text(json.dumps(item,sort_keys=True,indent=2)+'\n')
        os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_npz(path: Path, **kwargs):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.stem + '.tmp.npz')
    if temp.exists():
        raise FileExistsError(temp)
    try:
        np.savez_compressed(temp,**kwargs)
        os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)


def save_trial(root: Path, trial_key: str, forecast: np.ndarray, metadata: dict):
    path = Path(root)/'trials'/f'{trial_key}.npz'
    meta = path.with_suffix('.json')
    if path.exists() or meta.exists():
        raise FileExistsError(f'Trial exists: {path}')
    atomic_npz(path,forecast=np.asarray(forecast,dtype=np.float32))
    write_json(meta,{**metadata,'forecast_sha256':file_hash(path)})


def read_trial(root: Path, trial_key: str, shape):
    path = Path(root)/'trials'/f'{trial_key}.npz'
    meta = path.with_suffix('.json')
    if not path.exists() and not meta.exists():
        return None
    if not (path.exists() and meta.exists()):
        raise RuntimeError(f'Incomplete cached trial {trial_key}: inspect before resuming')
    metadata = json.loads(meta.read_text())
    if metadata['forecast_sha256'] != file_hash(path) or metadata.get('trial_key')!=trial_key:
        raise RuntimeError(f'Corrupt cached trial {trial_key}')
    with np.load(path,allow_pickle=False) as z:
        forecast=np.asarray(z['forecast'],dtype=np.float32)
    if forecast.shape!=shape or not np.isfinite(forecast).all():
        raise RuntimeError(f'Invalid cached trial forecast: {trial_key}')
    return forecast, metadata
