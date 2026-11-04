"""CPU tests for a standalone numerical diagnostic; no model downloads."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

FILE = Path(__file__).resolve().parents[2] / 'scripts' / 'maintenance' / 'diagnose_chronos2_precision.py'
spec = importlib.util.spec_from_file_location('chronos2_precision_diagnostic',FILE)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class FakeConfig:
    def __init__(self, **kwargs): self.kwargs=kwargs
    def to_dict(self): return dict(self.kwargs)


class FakePEFT:
    def __init__(self, config): self.peft_config={'default':FakeConfig(**config)}


def test_reports_exact_metadata_diff():
    train=FakePEFT({'r':8,'lora_alpha':16,'inference_mode':False})
    load=FakePEFT({'r':8,'lora_alpha':16,'inference_mode':True})
    changes=mod.compare_peft_metadata(train,load)
    assert changes == [{'adapter':'default','key':'inference_mode',
                        'trained':False,'reloaded':True}]


def test_no_metadata_diff():
    config={'r':8,'lora_alpha':16,'inference_mode':True}
    assert mod.compare_peft_metadata(FakePEFT(config),FakePEFT(config)) == []


def test_layout_detects_strides():
    torch=pytest.importorskip('torch')
    class Wrapper:
        def __init__(self, t): self.t=t
        def named_parameters(self): return iter([('w',self.t)])
        def named_buffers(self): return iter([])
    x=torch.arange(12.).reshape(3,4)
    a=Wrapper(x)
    b=Wrapper(x.t().contiguous().t())
    d=mod.compare_tensor_layouts(a,b)
    assert d['parameters']['n_differences']==1
    assert 'stride' in d['parameters']['first_differences'][0]
    assert d['buffers']['n_differences']==0


def test_layout_equal():
    torch=pytest.importorskip('torch')
    class Wrapper:
        def __init__(self, t): self.t=t
        def named_parameters(self): return iter([('w',self.t)])
        def named_buffers(self): return iter([])
    a=Wrapper(torch.ones(2,3))
    b=Wrapper(torch.zeros(2,3))
    assert mod.compare_tensor_layouts(a,b)['parameters']['n_differences']==0


def test_forecast_scaled_delta_is_not_mae():
    actual=mod.key_delta(np.array([100.,100.]),np.array([100.,99.]))
    assert actual['max_abs']==1.0
    assert actual['scaled_max']==0.01
    assert actual['mean_abs']==0.5
