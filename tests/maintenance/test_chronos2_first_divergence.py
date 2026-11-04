import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn

SCRIPT = Path(__file__).resolve().parents[2]/'scripts/maintenance/diagnose_chronos2_first_divergence.py'
spec = importlib.util.spec_from_file_location('first_divergence',SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules['first_divergence'] = mod
spec.loader.exec_module(mod)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1=nn.Linear(3,4)
        self.l2=nn.Linear(4,2)
    def forward(self,context,num_output_patches):
        return type('Result',(),{'quantile_preds':self.l2(self.l1(context))})()


def test_hook_parity_and_cleanup():
    a=Toy().eval(); b=Toy().eval()
    b.load_state_dict(a.state_dict())
    x=torch.ones(1,3)
    left,y=mod.capture_model(a,x,1)
    right,z=mod.capture_model(b,x,1)
    summary=mod.summarize_traces(left,right)
    assert torch.equal(y,z)
    assert summary['first_divergence'] is None
    assert not a.l1._forward_hooks and not a.l1._forward_pre_hooks


def test_first_divergence_detected_at_linear():
    a=Toy().eval(); b=Toy().eval()
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        b.l2.weight.add_(0.1)
    x=torch.ones(1,3)
    left,_=mod.capture_model(a,x,1)
    right,_=mod.capture_model(b,x,1)
    summary=mod.summarize_traces(left,right)
    assert summary['first_divergence']['module'] == 'l2'
    assert summary['first_divergence']['first_divergence_inside_module']
    assert summary['first_divergence']['input']['equal']


def test_tensor_exact_and_numeric_diffs():
    a=torch.tensor([1.,2.]); b=torch.tensor([1.,2.])
    assert mod.tensor_diff(a,b)['equal']
    assert mod.tensor_diff(a,b+0.1)['max_abs'] > 0
    assert not mod.tensor_diff(a,a.double())['equal']


def test_nested_output_and_mismatched_calls():
    x=mod.tensor_leaves({'a':[torch.ones(2),torch.zeros(3)]})
    y=mod.tensor_leaves({'a':[torch.ones(2),torch.ones(3)]})
    assert not mod.compare_leaves(x,y)['equal']
    assert mod.compare_leaves(x,x)['equal']
    assert mod.summarize_traces([{'name':'a','occurrence':0,'input':{},'output':{}}],[])['missing_calls']==1


def test_lora_watch_and_runtime():
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A=nn.ModuleDict({'default': nn.Linear(2,2)})
            self.lora_B=nn.ModuleDict({'default': nn.Linear(2,2)})
            self.scaling={'default':2.}
            self.r={'default':8}
            self.lora_alpha={'default':16}
            self.merged_adapters=[]
            self.disable_adapters=False
            self.active_adapters=['default']
            self.lora_dropout=nn.ModuleDict({'default':nn.Dropout(0.1)})
    d=Dummy()
    assert mod.watch_module('layer',d)
    state=mod.lora_runtime(nn.Sequential(d))
    assert state['0']['scaling']['default']==2.0
