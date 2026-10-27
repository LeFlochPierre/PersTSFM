from pathlib import Path
import importlib.util
from types import SimpleNamespace
import numpy as np
import pytest

src = Path(__file__).resolve().parents[2] / 'scripts/phase3/PersTSFM_phase3_2_gpu_cost.py'
spec=importlib.util.spec_from_file_location('cost_benchmark',src)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_known_model_envs():
    assert m.MODEL_ENVS['chronos2'] == 'PERTSFM_ENV'
    assert m.MODEL_ENVS['moirai2_small'] == 'PERTSFM_MOIRAI_ENV'
    assert m.MODEL_ENVS['toto2_313m'] == 'PERTSFM_TOTO_ENV'


def test_context_limit_and_cohort_validation():
    cohort = {'case_id': ['a','b']}
    class FakeFrame(dict):
        def __getitem__(self,key):
            if key == 'case_id':
                return SimpleNamespace(astype=lambda x: ['a','b'])
            return super().__getitem__(key)
    c = FakeFrame(cohort)
    assert m.validate_job('chronos2',384,{},c)==('a','b')
    with pytest.raises(ValueError,match='max context'):
        m.validate_job('tirex2',3072,{},c)
    with pytest.raises(ValueError,match='Unequal'):
        m.validate_job('chronos2',384,{},c,expected_case_ids=('a','c'))


def test_predict_once_batches_and_checks_predictions():
    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available(): return False
    class FakeModel:
        def __init__(self):self.sizes=[]
        def predict_batch(self,inputs,**kwargs):
            self.sizes.append(len(inputs))
            return [SimpleNamespace(case_id=i.case_id,prediction=np.array([1.,2.])) for i in inputs]
    fake = FakeModel()
    inputs=[SimpleNamespace(case_id=str(i)) for i in range(5)]
    dt,n=m.predict_once(fake,inputs,2,60,384,2,FakeTorch)
    assert n==5 and dt>=0 and fake.sizes==[2,2,1]


def test_bad_model_outputs_rejected():
    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available(): return False
    class FakeModel:
        def predict_batch(self,inputs,**kwargs):
            return [SimpleNamespace(case_id=i.case_id,prediction=np.array([np.nan])) for i in inputs]
    with pytest.raises(RuntimeError,match='Invalid model point'):
        m.predict_once(FakeModel(),[SimpleNamespace(case_id='a')],1,60,384,1,FakeTorch)
