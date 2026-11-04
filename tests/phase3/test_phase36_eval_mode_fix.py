"""CPU-side regression tests for strict eval-mode parity fix."""
import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
from src.experiments.adaptation_benchmark.inference_safety import (
    ensure_eval_mode, compare_checkpoint_parameters,
)


def test_eval_mode_forces_dropout_off():
    model=torch.nn.Sequential(torch.nn.Dropout(p=1.0))
    model.train()
    sample=torch.ones(4)
    assert torch.all(model(sample) == 0)
    pipeline=SimpleNamespace(model=model)
    ensure_eval_mode(pipeline)
    assert model.training is False
    assert torch.all(model(sample) == sample)


def test_eval_missing_model_fails_closed():
    with pytest.raises(RuntimeError, match='no model.eval'):
        ensure_eval_mode(SimpleNamespace())


def test_parameter_diff_detects_mismatch():
    a=torch.nn.Linear(2,2)
    b=torch.nn.Linear(2,2)
    b.load_state_dict(a.state_dict())
    equal=compare_checkpoint_parameters(a,b)
    assert equal['unequal_shared_tensors']==0
    with torch.no_grad(): b.weight[0,0]+=1
    different=compare_checkpoint_parameters(a,b)
    assert different['unequal_shared_tensors']==1
    assert different['max_abs_shared_parameter_delta']>0.9


def test_pipeline_source_calls_eval_guard():
    import inspect
    from src.experiments.adaptation_benchmark.models import Chronos2Backend
    from src.experiments.adaptation_benchmark.capability_models import AuditedChronos2Backend
    assert 'ensure_eval_mode(self.adapter.pipeline)' in inspect.getsource(Chronos2Backend.forecast)
    s=inspect.getsource(AuditedChronos2Backend.fit_audited)
    assert 'ensure_eval_mode(trained)' in s
    assert 'ensure_eval_mode(loaded)' in s
    assert 'compare_checkpoint_parameters(' in s
