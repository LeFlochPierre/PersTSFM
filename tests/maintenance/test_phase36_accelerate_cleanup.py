"""CPU tests for guarded post-training mixed-precision normalization."""
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest

MODULE = Path(__file__).resolve().parents[2] / 'scripts/maintenance/install_phase36_accelerate_cleanup.py'
spec = importlib.util.spec_from_file_location('accelerate_cleanup_installer', MODULE)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def test_helper_added_once():
    src='def ensure_eval_mode(pipeline):\n    pass\n'
    x=installer.transform('inference_safety.py',src)
    assert x.count('def normalize_post_training_pipeline(')==1
    assert installer.transform('inference_safety.py',x)==x


def test_model_insertion_after_fit():
    x='from .inference_safety import ensure_eval_mode\n'+'        self.adapter.pipeline = fitted\n        return elapsed, 123\n'
    patched=installer.transform('models.py',x)
    assert 'normalize_post_training_pipeline(fitted)' in patched
    assert patched.count('normalize_post_training_pipeline(fitted)')==1
    assert installer.transform('models.py',patched)==patched


def test_capability_insertion_at_post_fit_boundary():
    x='from .inference_safety import (ensure_eval_mode, compare_checkpoint_parameters, x)\n'
    x+='        torch.cuda.synchronize()\n        duration=float(time.perf_counter()-begin)\n'
    x+="        audit={**weight,**train,**schedule,'mode':mode,'seed':seed,\n"
    patched=installer.transform('capability_models.py',x)
    assert 'forward_wrapper_removed=normalize_post_training_pipeline(trained)' in patched
    assert "'accelerate_forward_wrapper_removed':forward_wrapper_removed" in patched
    assert installer.transform('capability_models.py',patched)==patched


def test_reject_unknown_source():
    with pytest.raises(RuntimeError):
        installer.transform('models.py','some unrelated file')


def test_wrapper_removed_preserves_model_and_output(monkeypatch):
    # Model which simulates an Accelerate forward wrapper left after Trainer.fit.
    class Model:
        def __init__(self):
            self.training=True
            self.weight=3.0
            self._original_forward=self.true_forward
            self.forward=lambda x: round(self.true_forward(x), 1)  # precision loss
        def true_forward(self, x):
            return x * self.weight
        def eval(self):
            self.training=False
            return self
    class Pipeline:
        def __init__(self, model):
            self.model=model
    utils=types.ModuleType('accelerate.utils')
    def extract_model_from_parallel(model, *, keep_fp32_wrapper=True):
        assert keep_fp32_wrapper is False
        model.forward=model.__dict__.pop('_original_forward')
        return model
    utils.extract_model_from_parallel=extract_model_from_parallel
    fake=types.ModuleType('accelerate'); fake.utils=utils
    monkeypatch.setitem(sys.modules,'accelerate',fake)
    monkeypatch.setitem(sys.modules,'accelerate.utils',utils)
    # Execute actual patch helper with a minimal eval guard instead of importing heavy dependencies.
    ns={'ensure_eval_mode':lambda p: p.model.eval()}
    exec(installer.HELPER,ns)
    model=Model(); pipe=Pipeline(model)
    assert pipe.model.forward(0.38)==1.1
    assert ns['normalize_post_training_pipeline'](pipe) is True
    assert pipe.model is model
    assert model.weight==3.0
    assert pipe.model.forward(0.38)==pytest.approx(1.14)
    assert model.training is False
    assert ns['normalize_post_training_pipeline'](pipe) is False


def test_cpu_test_no_wrapper_still_eval():
    class Model:
        def __init__(self): self.training=True
        def eval(self): self.training=False
    pipe=types.SimpleNamespace(model=Model())
    ns={'ensure_eval_mode':lambda p:p.model.eval()}
    exec(installer.HELPER,ns)
    assert ns['normalize_post_training_pipeline'](pipe) is False
    assert pipe.model.training is False
