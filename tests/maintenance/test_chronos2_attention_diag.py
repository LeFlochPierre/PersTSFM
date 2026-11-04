import importlib.util
from pathlib import Path
import numpy as np
import pytest

FILE=Path(__file__).resolve().parents[2]/'scripts/maintenance/diagnose_chronos2_attention.py'
spec=importlib.util.spec_from_file_location('diag_attention',FILE)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class Config:
    _attn_implementation='eager'
class MHA:
    def __init__(self,v):
        self.config=Config()
        self.config._attn_implementation=v
class Stub:
    def __init__(self,mode):
        self.layers=[MHA(mode) for _ in range(3)]
    def named_modules(self):
        for i,m in enumerate(self.layers):
            yield 'x'+str(i),m

def test_modes_and_mismatch():
    d=mod.mismatch_summary(Stub('eager'),Stub('sdpa'))
    assert d['mismatched_attention_modules']==3
    assert d['trained_modes']=={'eager':3}

def test_force_mode():
    a=Stub('eager')
    assert mod.set_attention(a,'sdpa')==3
    assert dict(mod.modes(a))['x0']=='sdpa'

def test_unsupported_backend():
    with pytest.raises(ValueError):mod.set_attention(Stub('eager'),'flash')

def test_delta():
    a=np.array([1.,2.]);b=np.array([1.,2.1]);d=mod.delta(a,b)
    assert d['max_abs']==pytest.approx(.1)

def test_shapes():
    with pytest.raises(ValueError):mod.delta(np.array([1.]),np.array([1.,2.]))
