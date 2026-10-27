"""Regression tests for the history-pool ablation; no actual model training."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('phase33d', ROOT/'scripts/phase3/lora_history_saturation.py')
mod=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def records(n=2,h=4,length=52):
    out=[]
    for i in range(n):
        out.append({'id':f'e{i}', 'y':np.arange(length,dtype=float)+i,
                    'time':pd.date_range('2020-01-01',periods=length,freq='h')})
    return out


def test_training_prefix_excludes_both_targets():
    cohort=records()
    pools=mod.training_series(cohort,4,32)
    assert all(len(x)==32 for x in pools)
    # full pre-lag2 prefix: [0..43]; final value must be 43, NOT 44..51.
    np.testing.assert_array_equal(pools[0],np.arange(12,44,dtype=np.float32))
    np.testing.assert_array_equal(pools[1],np.arange(13,45,dtype=np.float32))
    cohort[0]['y'][-8:]=1e8
    np.testing.assert_array_equal(mod.training_series(cohort,4,32)[0],pools[0])


def test_grid_validates_full_history_and_never_uses_lag1():
    cohort=records()
    counts=mod.validate_grid(cohort,4,16,[24,32,44],[2,20])
    assert counts=={'e0':44,'e1':44}
    with pytest.raises(RuntimeError,match='pre-lag2'):
        mod.validate_grid(cohort,4,16,[45],[2])
    with pytest.raises(ValueError,match='>= context'):
        mod.validate_grid(cohort,4,16,[23],[2])


def test_nonfinite_pre_lag2_rejected():
    cohort=records()
    cohort[0]['y'][-31]=np.nan
    with pytest.raises(RuntimeError,match='training pool'):
        mod.validate_grid(cohort,4,16,[24],[2])


def test_grid_no_dup_and_budgets_positive():
    cohort=records()
    with pytest.raises(ValueError,match='distinct'):
        mod.validate_grid(cohort,4,16,[24,24],[2])
    with pytest.raises(ValueError,match='positive'):
        mod.validate_grid(cohort,4,16,[24],[0,20])


def test_selection_zero_shot_tie_break_and_best_history():
    table=pd.DataFrame([
        {'model':'zero_shot','validation_mae':1.0,'steps':0,'train_history_steps':0},
        {'model':'lora_h24_s20','validation_mae':1.0,'steps':20,'train_history_steps':24},
        {'model':'lora_h32_s20','validation_mae':1.0,'steps':20,'train_history_steps':32},
    ])
    assert mod.choose_validation(table)=='zero_shot'
    table.loc[table.model=='lora_h24_s20','validation_mae']=.8
    assert mod.choose_validation(table)=='lora_h24_s20'
    table.loc[table.model=='lora_h32_s20','validation_mae']=.8
    assert mod.choose_validation(table)=='lora_h24_s20'


def test_history_validation_rejects_same_len_too_short():
    with pytest.raises(RuntimeError,match='validation or is unavailable'):
        mod.training_series(records(),4,45)


def make_locked_run(tmp_path):
    snap=tmp_path/'historical.npz'
    cohort=records(2,4,52)
    from scripts.phase3.context_saturation import save_snapshot,sha256_file,sha256_histories
    save_snapshot(cohort,'M_DENSE_1H',4,snap)
    out=tmp_path/'run'
    out.mkdir()
    names=['zero_shot','lora_h24_s2']
    forecasts=np.zeros((2,2,2,4,4),dtype=np.float32)
    # Later labels are original 48..51 / 49..52. Zero-shot is the best.
    for entity,r in enumerate(cohort):
        actual=r['y'][-4:]
        forecasts[0,1,entity,:,0]=actual
        forecasts[1,1,entity,:,0]=actual+3.0
    np.savez_compressed(out/'forecasts.npz',forecasts=forecasts,names=np.asarray(names,dtype='U64'))
    pd.DataFrame([
       {'model':'zero_shot','steps':0,'train_history_steps':0,'validation_mae':5.0,'training_wall_seconds':0.0},
       {'model':'lora_h24_s2','steps':2,'train_history_steps':24,'validation_mae':6.0,'training_wall_seconds':1.0},
    ]).to_csv(out/'validation.csv',index=False)
    meta={'schema':'pertsfm_3_3d_lora_history_v1','task':'M_DENSE_1H',
          'horizon':4,'ids':[r['id'] for r in cohort],'context':16,
          'snapshot_path':str(snap),'snapshot_sha256':sha256_file(snap),
          'history_sha256':sha256_histories(cohort)}
    (out/'manifest.json').write_text(json.dumps(meta))
    lock={'schema':'pertsfm_3_3d_lock_v1','selected_model':'zero_shot',
          'locked_before_lag1_scoring':True,'manifest_sha256':sha256_file(out/'manifest.json'),
          'validation_sha256':sha256_file(out/'validation.csv'),
          'forecasts_sha256':sha256_file(out/'forecasts.npz')}
    (out/'selection.json').write_text(json.dumps(lock))
    return out


def test_lag1_evaluated_only_after_valid_lock(tmp_path):
    out=make_locked_run(tmp_path)
    assert not (out/'later.csv').exists()
    mod.score_later(out)
    result=pd.read_csv(out/'later.csv').set_index('model')
    assert result.loc['zero_shot','later_mae']==0
    assert result.loc['lora_h24_s2','later_mae']==pytest.approx(3)
    with pytest.raises(FileExistsError):
        mod.score_later(out)


def test_modified_validation_rejected(tmp_path):
    out=make_locked_run(tmp_path)
    with (out/'validation.csv').open('a') as f:
        f.write('injected-row,1,24,0,0\n')
    with pytest.raises(RuntimeError,match='changed'):
        mod.score_later(out)


def test_modified_snapshot_rejected(tmp_path):
    out=make_locked_run(tmp_path)
    snapshot=tmp_path/'historical.npz'
    with snapshot.open('ab') as f:
        f.write(b'edit')
    with pytest.raises(RuntimeError,match='snapshot changed'):
        mod.score_later(out)


def test_selection_not_lag1_dependence():
    rows=[{'model':'zero_shot','steps':0,'train_history_steps':0,'validation_mae':4},
          {'model':'lora_h24_s2','steps':2,'train_history_steps':24,'validation_mae':3}]
    t=pd.DataFrame(rows)
    first=mod.choose_validation(t)
    t['later_mae']=[-100,100000]
    assert mod.choose_validation(t)==first
