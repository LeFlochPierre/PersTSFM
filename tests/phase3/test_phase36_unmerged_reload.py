"""CPU-only regression tests for exact PEFT checkpoint parity."""
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType
import pytest
import torch
from src.experiments.adaptation_benchmark.inference_safety import (
    compare_checkpoint_parameters, reload_unmerged_chronos2_adapter,
    verify_unmerged_parameter_parity,
)


def test_unmerged_loader_preserves_adapter_and_does_not_merge(monkeypatch, tmp_path):
    (tmp_path/'adapter_config.json').write_text('{}')
    calls=[]
    class PeftModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_B=torch.nn.Parameter(torch.ones(2))
        def merge_and_unload(self):
            raise AssertionError('Must never merge in strict parity path')
    class AutoPeftModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append((path,kwargs))
            return PeftModel()
    class Chronos2Pipeline:
        def __init__(self,model): self.model=model
    monkeypatch.setitem(sys.modules,'chronos',SimpleNamespace(Chronos2Pipeline=Chronos2Pipeline))
    monkeypatch.setitem(sys.modules,'peft',SimpleNamespace(AutoPeftModel=AutoPeftModel,
                              PeftModel=PeftModel,__version__='0.21.2'))
    p=reload_unmerged_chronos2_adapter(tmp_path)
    assert isinstance(p.model,PeftModel)
    assert not p.model.training
    assert calls==[(str(tmp_path), {'device_map':'cuda','import_allowlist':['chronos']})]


def test_missing_peft_checkpoint_is_rejected(tmp_path):
    with pytest.raises(RuntimeError,match='adapter_config.json'):
        reload_unmerged_chronos2_adapter(tmp_path)


def test_strict_unmerged_parameter_checks():
    a={'trained_parameter_tensors':3,'loaded_parameter_tensors':3,
       'missing_from_reload_count':0,'additional_in_reload_count':0,
       'shared_lora_tensors':2,'max_abs_shared_parameter_delta':0.0}
    assert verify_unmerged_parameter_parity(a)
    for field,bad in [('max_abs_shared_parameter_delta',2e-5),
                      ('missing_from_reload_count',1),('shared_lora_tensors',0),
                      ('loaded_parameter_tensors',2)]:
        b=dict(a);b[field]=bad
        with pytest.raises(RuntimeError):verify_unmerged_parameter_parity(b)


def test_unmerged_parameter_diff_not_merged_model():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_B=torch.nn.Linear(2,2,bias=False)
    a=Toy();b=Toy();b.load_state_dict(a.state_dict())
    report=compare_checkpoint_parameters(a,b)
    assert report['shared_lora_tensors']>0
    assert verify_unmerged_parameter_parity(report)
    with torch.no_grad():b.lora_B.weight[0,0]+=1.0
    report=compare_checkpoint_parameters(a,b)
    with pytest.raises(RuntimeError):verify_unmerged_parameter_parity(report)


def test_audited_backend_uses_unmerged_for_scoring():
    from src.experiments.adaptation_benchmark.capability_models import AuditedChronos2Backend
    s=inspect.getsource(AuditedChronos2Backend.fit_audited)
    assert 'reload_unmerged_chronos2_adapter' in s
    assert "self.adapter.pipeline=unmerged" in s
    assert 'official_merged_reload_scaled_delta' in s
    assert "audit['verified_checkpoint_reload']=False" in s
    assert 'capability_audit_failed.json' in s
