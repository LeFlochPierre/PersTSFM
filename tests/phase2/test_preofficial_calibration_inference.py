"""Offline contracts for pre-official calibration forecasting and scoring."""
import numpy as np
import pandas as pd
import pytest

from scripts.phase2.run_preofficial_calibration import choose_cases, make_forecast_input, checked_output
from scripts.phase2.score_preofficial_calibration import score_join

HOUR=3600*10**9


def make_record(entity='e', lag=1, length=96, h=4, context=32):
    stamps=np.arange(length,dtype=np.int64)*HOUR + 1_600_000_000_000_000_000
    future=stamps[-1]+HOUR*np.arange(1,h+1,dtype=np.int64)
    return {'task_id':'t','case_id':f't__{entity}__lag{lag:02d}', 'entity_id':entity,
            'lag_before_first_official':lag,'feasible_contexts':[8,16,32],
            'history':np.arange(length,dtype=float).tolist(),
            'history_timestamps_ns':stamps.tolist(),'future_timestamps_ns':future.tolist(),
            'history_missing_saved':0,'observed_steps_saved':length,'horizon':h}


def test_case_choosing_uses_k_most_recent_origin_lags():
    rows=[make_record(entity=e,lag=lag) for e in ['a','b'] for lag in [1,2,3]]
    d=choose_cases(pd.DataFrame(rows),k=2,context=16)
    assert len(d)==4
    assert set(d.lag_before_first_official)=={1,2}


def test_case_choosing_does_not_drop_short_histories_silently():
    rows=[make_record(lag=lag) for lag in [1,2,3]]
    rows[0]['feasible_contexts']=[8]
    with pytest.raises(ValueError,match='infeasible'):
        choose_cases(pd.DataFrame(rows),k=3,context=32)


def test_missing_origin_rejected():
    with pytest.raises(ValueError,match='Not all entities'):
        choose_cases(pd.DataFrame([make_record(lag=1)]), k=2, context=8)


def test_regular_history_and_future_cadence():
    x,ts,future,minutes=make_forecast_input(make_record(),32)
    assert x.size==32 and ts.size==32 and future.size==4 and minutes==60


def test_irregular_calendar_or_gaps_rejected():
    r=make_record()
    r['future_timestamps_ns'][1]+=123456789
    with pytest.raises(ValueError,match='irregular'):
        make_forecast_input(r,32)


def test_missing_data_rejected():
    r=make_record()
    r['history'][-1]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):
        make_forecast_input(r,32)


def test_scoring_rejects_timestamp_change():
    r=make_record()
    y=pd.DataFrame([{'case_id':r['case_id'],'entity_id':'e','lag_before_first_official':1,
                     'target_timestamps_ns':r['future_timestamps_ns'], 'target_values':[1.,2.,3.,4.]}])
    p={'case_id':r['case_id'],'entity_id':'e','lag_before_first_official':1,'task_id':'t',
       'context_steps':32,'k':1,'model':'m','future_timestamps_ns':r['future_timestamps_ns'].copy(),
       'prediction':[1.,2.,3.,4.],'q10':[0.,1.,2.,3.],'q50':[1.,2.,3.,4.], 'q90':[2.,3.,4.,5.]}
    run={'context_steps':32,'k':1,'model':'m'}
    result=score_join(run,pd.DataFrame([r]),y,pd.DataFrame([p]))
    assert len(result)==1 and result.mae.iloc[0]==0
    p['future_timestamps_ns'][0]+=1
    with pytest.raises(ValueError,match='timestamps mismatch'):
        score_join(run,pd.DataFrame([r]),y,pd.DataFrame([p]))


def test_scoring_rejects_partial_predictions():
    r=make_record()
    with pytest.raises(ValueError,match='Case ID mismatch'):
        score_join({'context_steps':8,'k':1,'model':'m'},pd.DataFrame([r]),
                   pd.DataFrame([{'case_id':r['case_id']}]),pd.DataFrame(columns=['case_id']))


def test_full_pipeline_is_label_separated_and_rejects_tampered_inputs(tmp_path, monkeypatch):
    """Synthetic inference -> distinct scorer, with a tamper-integrity check."""
    pytest.importorskip("pyarrow")
    import json, sys, types
    from dataclasses import dataclass
    from scripts.phase2 import run_preofficial_calibration as runner
    from scripts.phase2 import score_preofficial_calibration as scorer

    root=tmp_path/'inputs'/'synthetic_task'
    root.mkdir(parents=True)
    rec=make_record(length=30, h=4)
    rec['case_id']='synthetic_task__e__lag01'
    rec['task_id']='synthetic_task'
    rec['feasible_contexts']=[8,16]
    inp=root/'calibration_inputs.parquet'
    pd.DataFrame([rec]).to_parquet(inp,index=False)
    truth=root/'historical_calibration_labels.parquet'
    pd.DataFrame([{
        'case_id':rec['case_id'],'entity_id':'e','lag_before_first_official':1,
        'target_timestamps_ns':rec['future_timestamps_ns'], 'target_values':[1.]*4
    }]).to_parquet(truth,index=False)
    m=dict(schema='pertsfm_preofficial_calibration_v1',
      role='HISTORICAL_CALIBRATION_ONLY_NOT_OFFICIAL_EVALUATION',
      official_ground_truth_accessed=False,official_test_target_values_included=False,
      timestamp_unit='ns',input_sha256=runner.sha256(inp), labels_sha256=runner.sha256(truth),
      task_id='synthetic_task',contexts_steps=[8,16],max_origins=1,n_cases=1,
      dataset_config='synthetic',horizon=4)
    (root/'manifest.json').write_text(json.dumps(m))

    @dataclass
    class ForecastInput:
        case_id: str
        timestamps: object
        values: object
    class FakeModel:
        def predict_batch(self, *, inputs, horizon, resolution_minutes, context_steps, batch_size):
            assert resolution_minutes == 60 and context_steps == 8 and horizon == 4
            return [types.SimpleNamespace(case_id=i.case_id,
                prediction=np.ones(horizon), q10=np.zeros(horizon),
                q50=np.ones(horizon),q90=np.full(horizon,2.)) for i in inputs]
    adapter=types.ModuleType('src.models.adapters')
    adapter.ForecastInput=ForecastInput
    adapter.build_model=lambda name,batch_size: FakeModel()
    monkeypatch.setitem(sys.modules,'src',types.ModuleType('src'))
    monkeypatch.setitem(sys.modules,'src.models',types.ModuleType('src.models'))
    monkeypatch.setitem(sys.modules,'src.models.adapters',adapter)
    output=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['prog','--task-id','synthetic_task','--model','fake',
        '--context','8','--k','1','--input-root',str(tmp_path/'inputs'), '--output-root',str(output)])
    runner.main()
    pred=output/'synthetic_task'/'fake'/'context_8'/'k1'/'full'/'predictions.parquet'
    assert pred.is_file()
    monkeypatch.setattr(sys,'argv',['prog','--predictions',str(pred),'--input-root',str(tmp_path/'inputs')])
    scorer.main()
    score=json.loads(pred.with_name('calibration_score.json').read_text())
    assert score['calibration_mean_case_mae']==0.0 and score['n_cases']==1
    # First-stage inference intentionally never needed to read historical_calibration_labels.
    with pytest.raises(FileExistsError):
        scorer.main()
    changed=pd.read_parquet(truth)
    changed.at[0,'target_values']=[500.,500.,500.,500.]
    changed.to_parquet(truth,index=False)
    with pytest.raises(RuntimeError,match='labels hash mismatch'):
        scorer.main()
