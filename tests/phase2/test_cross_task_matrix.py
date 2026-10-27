"""Unit-level tests for Phase 2.13 scheduler, without any FEV/test labels."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

from scripts.phase2 import run_cross_task_matrix as sched


def mock_manifest(task='fev_0_10_0_044', config='M_DENSE_1H', n=3):
    return {'schema':'pertsfm_preofficial_calibration_v1',
            'task_id':task,'dataset_config':config,
            'role':'HISTORICAL_CALIBRATION_ONLY_NOT_OFFICIAL_EVALUATION',
            'official_ground_truth_accessed':False,'max_origins':3,
            'contexts_steps':[8,16,96,192,384,768,1536],
            'coverage':'FULL','n_cases':n,'input_sha256':'H' * 64,
            'family_id_exact':'family'}


def example_cases(task='fev_0_10_0_044', n_entities=2):
    records=[]
    base = 1_700_000_000_000_000_000
    for ent in range(n_entities):
        for lag in (1,2,3):
            hist_length = 180 - lag*8
            ts = base + np.arange(hist_length, dtype=np.int64)*3_600_000_000_000
            fut = ts[-1]+np.arange(1,9,dtype=np.int64)*3_600_000_000_000
            records.append(dict(case_id=f'{task}__{ent}__lag0{lag}',entity_id=str(ent),
                task_id=task,lag_before_first_official=lag,horizon=8,
                observed_steps_saved=hist_length,history=np.ones(hist_length).tolist(),
                history_timestamps_ns=ts.tolist(),future_timestamps_ns=fut.tolist(),
                history_missing_saved=0,feasible_contexts=[8,16,96]))
    return pd.DataFrame(records)


def test_preflight_full_cohort_context_and_case_matching():
    m=mock_manifest(n=6)
    df=example_cases()
    assert sched.preflight_task_context(m,df,96,3) == (6,2)
    with pytest.raises(ValueError, match='infeasible'):
        sched.preflight_task_context(m,df,384,3)
    with pytest.raises(ValueError, match='not in frozen'):
        sched.preflight_task_context(m,df,44,3)


def test_preflight_detects_irregular_cadence():
    m=mock_manifest(n=6)
    df=example_cases()
    ts=np.asarray(df.iloc[0]['history_timestamps_ns']).copy()
    ts[-2] += 1_000_000_000
    df.at[0,'history_timestamps_ns']=ts.tolist()
    with pytest.raises(ValueError,match='irregular'):
        sched.preflight_task_context(m,df,96,3)


def test_plan_skips_tirex_limit_and_invalid_context(monkeypatch,tmp_path):
    m=mock_manifest(n=6)
    m['contexts_steps']=[96,3072]
    df=example_cases()
    monkeypatch.setattr(sched,'load_frozen_inputs',lambda path:(m,df))
    result=sched.build_plan([(m,tmp_path/'fake')],['moirai2_small','tirex2'],[96,3072],3,'rearrange',output_root=tmp_path/'out')
    assert [x.status for x in result] == ['PENDING','SKIP_INFEASIBLE','PENDING','SKIP_INFEASIBLE']
    assert '2048' in result[-1].reason
    assert all(x.n_cases==6 for x in result if x.status=='PENDING')


def test_discovery_preserves_user_task_order(tmp_path):
    root=tmp_path/'exports'
    for ident,name in [('01','LOOP_SEATTLE_1H'),('02','M_DENSE_1H')]:
        folder=root/ident
        folder.mkdir(parents=True)
        (folder/'manifest.json').write_text(json.dumps(mock_manifest(ident,name,6)))
    selected=sched.discover_exports(root,['M_DENSE_1H','LOOP_SEATTLE_1H'])
    assert [m['dataset_config'] for m,_ in selected]==['M_DENSE_1H','LOOP_SEATTLE_1H']
    with pytest.raises(ValueError,match='one full-cohort'):
        sched.discover_exports(root,['unknown'])


def test_resume_rejects_incomplete_directory(tmp_path):
    j=sched.Job('task','config','family','chronos2','PERTSFM_ENV',96,3,6,2,'','','')
    folder=tmp_path/'out'
    folder.mkdir()
    status,reason=sched.classify_output(folder,j,mock_manifest(task='task'), 'rearrange')
    assert status=='BLOCKED' and 'Partial output' in reason


def test_resume_rejects_wrong_quantile_policy(monkeypatch,tmp_path):
    j=sched.Job('task','config','family','chronos2','PERTSFM_ENV',96,3,6,2,'','','')
    out=tmp_path/'out';out.mkdir()
    (out/'predictions.parquet').write_bytes(b'prediction bytes')
    input_dir=tmp_path/'input';(input_dir/'task').mkdir(parents=True)
    (input_dir/'task'/'manifest.json').write_text('{}')
    monkeypatch.setattr(sched,'INPUT_ROOT',input_dir)
    m=mock_manifest(task='task')
    run=dict(schema='pertsfm_preofficial_calibration_predictions_v1',
        role='PREOFFICIAL_HISTORICAL_CALIBRATION',task_id='task',model='chronos2',
        context_steps=96,k=3,n_cases=6,n_entities=2,cohort='FULL',
        input_sha256=m['input_sha256'], input_manifest_sha256=sched.sha256(input_dir/'task'/'manifest.json'),
        predictions_sha256=sched.sha256(out/'predictions.parquet'),quantile_policy='strict')
    (out/'run_manifest.json').write_text(json.dumps(run))
    status,reason=sched.classify_output(out,j,m,'rearrange')
    assert status=='BLOCKED' and 'quantile policy' in reason
    status,reason=sched.classify_output(out,j,m,'strict')
    assert status=='NEEDS_SCORE'


def test_no_labels_or_fev_in_scheduler_source():
    source=Path(sched.__file__).read_text()
    assert 'get_ground_truth(' not in source
    assert 'import fev' not in source
    assert 'historical_calibration_labels.parquet' not in source


def test_tirex_and_model_registry():
    assert sched.KNOWN_MODELS['moirai2_small']=='PERTSFM_MOIRAI_ENV'
    assert sched.KNOWN_MODELS['toto2_313m']=='PERTSFM_TOTO_ENV'
    assert sched.KNOWN_MODELS['chronos2']=='PERTSFM_ENV'


def test_case_catalog_enforces_full_case_count(tmp_path):
    j=sched.Job('t','c','family','chronos2','PERTSFM_ENV',96,3,2,1,'DONE','',str(tmp_path))
    p=tmp_path/'calibration_case_metrics.csv'
    p.write_text('case_id,entity_id,lag_before_first_official,mae,mean_pinball_3,pinball_q10,pinball_q50,pinball_q90\\n'
                 'c1,e,1,1.0,0.5,0.5,0.5,0.5\\n'
                 'c2,e,2,1.1,0.6,0.6,0.6,0.6\\n'.replace('\\n','\n'))
    output=tmp_path/'case_catalog.csv'
    assert sched.build_case_catalog([j], output) == 2
    tab=pd.read_csv(output)
    assert len(tab)==2 and set(tab['model'])=={'chronos2'}
    j.n_cases=3
    with pytest.raises(RuntimeError,match='Case count mismatch'):
        sched.build_case_catalog([j],output)
