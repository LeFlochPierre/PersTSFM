"""B3.7 tests, using a synthetic registry and real B3 frozen/shard contracts."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
from src.budgeted_forecasting import b37_suite as b37
from src.budgeted_forecasting import capability_atlas as b3


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    raw=tmp_path/'dummy_raw.csv'; raw.write_text('dummy')
    protocol=tmp_path/'protocol.csv'
    times=pd.to_datetime(['2026-01-04','2026-01-05','2026-01-11','2026-01-12'])
    frame=pd.DataFrame([dict(entity=e,eval_origin=t,target_start=t+pd.Timedelta(hours=1),
               target_end=t+pd.Timedelta(hours=2),split='validation',eval_index=i)
              for e in ['A','B'] for i,t in enumerate(times)])
    frame.to_csv(protocol,index=False)
    idx=pd.date_range('2026-01-01',periods=24*20,freq='h')
    series={e:pd.Series(np.arange(len(idx),dtype=float)+i*10,index=idx) for i,e in enumerate(['A','B'])}
    spec=SimpleNamespace(key='electricity', id_col='entity',raw_path=raw,protocol_path=protocol,
        default_split='validation',context_hours=(24,48),resolution_minutes=60,forecast_steps=2, loader_kind='dummy')
    def load_proto(_spec,split):
        return pd.read_csv(protocol,parse_dates=['eval_origin','target_start','target_end'])
    def extract(loaded,row,h):
        t=pd.Timestamp(row['eval_origin']); n=int(h); s=loaded.series[row.entity]
        return dict(case_id=f'x::{row.entity}::{t}',entity_id=row.entity,eval_origin=t,eval_index=int(row.eval_index),
             context=s.loc[t-pd.Timedelta(hours=n-1):t],truth=s.loc[t+pd.Timedelta(hours=1):t+pd.Timedelta(hours=2)])
    reg=SimpleNamespace(get_dataset_spec=lambda k:spec,load_protocol=load_proto,extract_case=extract,
        LoadedDataset=lambda **kw:SimpleNamespace(**kw),_load_dummy=lambda _s,ids:{id:series[id] for id in ids})
    monkeypatch.setattr(b3,'_registry',lambda:reg)
    monkeypatch.setattr(b37,'_registry',lambda:reg)
    cohort=tmp_path/'cohort'/'cohort.json'
    b3.freeze_cohort('electricity','validation',2,2,2,48,13,cohort.parent)
    shards=tmp_path/'shards'
    b3.run_single_action(cohort,'last_value',None,shards/'last_value.csv')
    return tmp_path, cohort, shards


def config_for(tiny):
    root,cohort,shards=tiny
    return dict(schema_version='pertsfm_b37_grid_v1', baseline_env='MAIN',
        model_envs={key:'SEPARATE' for key in b37.MODEL_KEYS},
        experiments=[dict(dataset='electricity',active=True,contexts_hours=[24,48],max_context_hours=48,
            protocol_split='validation',cohort=str(cohort),shards=str(shards),report_dir=str(root/'report'))])


def test_existing_action_validated(tiny):
    _,cohort,shards=tiny
    assert b37.validate_action(cohort,'last_value',None,shards/'last_value.csv')=='last_value'


def test_invalid_context_rejected(tiny):
    cfg=config_for(tiny)
    cfg['experiments'][0]['contexts_hours']=[24,72]
    cfg['experiments'][0]['max_context_hours']=72
    with pytest.raises(ValueError,match='unregistered context'):
        b37.inspect(cfg)


def test_invalid_max_context_rejected(tiny):
    cfg=config_for(tiny)
    cfg['experiments'][0]['max_context_hours']=24
    with pytest.raises(ValueError,match='max'):
        b37.inspect(cfg)


def test_requires_all_environments(tiny):
    cfg=config_for(tiny)
    del cfg['model_envs']['toto2_313m']
    with pytest.raises(ValueError,match='6 TSFMs'):
        b37.inspect(cfg)


def test_plan_resumes_valid_action_and_marks_missing(tiny):
    root,cohort,shards=tiny
    cfg=config_for(tiny)
    rows=b37.build_plan(cfg,root/'b37plan')
    assert any(x['model']=='last_value' and x['status']=='VERIFIED_EXISTING' for x in rows)
    assert any(x['model']=='toto2_22m' and x['status']=='PENDING' for x in rows)
    assert (root/'b37plan'/'run_missing.sh').is_file()
    txt=(root/'b37plan'/'run_missing.sh').read_text()
    assert 'conda run --no-capture-output -n SEPARATE' in txt
    assert '--context-hours 24.0' in txt
    assert 'verify-action' in txt
    assert len(rows)==14 # 2 baselines plus 6 models x 2 contexts


def test_plan_never_overwrites(tiny):
    root,*_=tiny
    cfg=config_for(tiny)
    b37.build_plan(cfg,root/'b37plan')
    with pytest.raises(FileExistsError,match='non-empty'):
        b37.build_plan(cfg,root/'b37plan')


def test_tampered_shard_refused(tiny):
    _,cohort,shards=tiny
    shard=shards/'last_value.csv'
    shard.write_text(shard.read_text().replace('electricity','tampered',1))
    with pytest.raises(ValueError,match='sha256'):
        b37.validate_action(cohort,'last_value',None,shard)


def test_wrong_cohort_refused(tiny):
    _,cohort,shards=tiny
    c=json.loads(cohort.read_text());c['seed']=999
    cohort.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='cohort_sha256'):
        b37.validate_action(cohort,'last_value',None,shards/'last_value.csv')


def test_tirex_cap_skipped(tiny):
    root,cohort,shards=tiny
    cfg=config_for(tiny)
    # Synthetic minute resolution, many steps: 48h -> 2880 >2048
    from src.budgeted_forecasting import b37_suite as bm
    orig=bm.inspect
    def mock_inspect(*args,**kwargs):
        out=orig(*args,**kwargs); out[0]['resolution_minutes']=1; return out
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bm,'inspect',mock_inspect)
        rows=bm.build_plan(cfg,root/'b37_cap')
    assert any(x['model']=='tirex2' and x['status']=='UNSUPPORTED_TIREX_MAX2048' for x in rows)
