"""Locate first forward divergence of trained versus reloaded Chronos-2 LoRA.

Two-step historical development smoke ONLY; diagnostic does not modify existing
research sources or read lag1 targets. It compares exact identical input tensors,
with LoRA on/off, then records forward *inputs and outputs* from selected modules.
Run from the PersTSFM repository root in PERTSFM_ENV.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np


# Limit the trace to the model features and linear projection layers where a
# PEFT/non-PEFT computational discrepancy can first occur. Do not capture
# attention weight matrices or huge model-level output structures.
WATCH_CLASSES = frozenset({
    'Linear', 'Chronos2LayerNorm', 'ResidualBlock', 'MHA',
    'Chronos2EncoderBlock', 'Chronos2Encoder', 'InstanceNorm',
    'Patch', 'TimeSelfAttention', 'GroupSelfAttention',
    'FeedForward', 'Chronos2RotaryEmbedding', 'Embedding',
    'Dropout', 'LoraLayer',
})
MAX_TENSOR_ELEMENTS = 3_000_000


def watch_module(name, module):
    cls = type(module).__name__
    if cls in WATCH_CLASSES:
        return True
    # PEFT LoRA Linear inherits a tuner class but is not nn.Linear.
    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
        return True
    # Catch any other small leaves that do floating-point work.
    return False


def tensor_leaves(obj, prefix='', *, limit=12):
    """Copy a bounded number of tensors without changing values or dtypes."""
    import torch
    out = {}

    def rec(v, p, depth):
        if len(out) >= limit or depth > 5:
            return
        if isinstance(v, torch.Tensor):
            if v.numel() <= MAX_TENSOR_ELEMENTS:
                out[p or 'tensor'] = v.detach().to(device='cpu', copy=True)
            else:
                out[p or 'tensor'] = {'shape': list(v.shape), 'skipped': 'too_large'}
            return
        if isinstance(v, dict):
            for k in sorted(v, key=str):
                rec(v[k], f'{p}.{k}' if p else str(k), depth + 1)
        elif isinstance(v, (tuple, list)):
            for j, z in enumerate(v):
                rec(z, f'{p}[{j}]', depth + 1)
        # dataclass/model outputs are usually Mapping/tuple; inspect named
        # fields only as a fallback, never arbitrary object recursion.
        elif hasattr(v, '__dataclass_fields__'):
            for key in v.__dataclass_fields__:
                rec(getattr(v, key), f'{p}.{key}' if p else str(key), depth + 1)

    rec(obj, prefix, 0)
    return out


def tensor_diff(a, b):
    """Summarize tensor equality with native dtype and NaN-safe comparison."""
    import torch
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return {'equal': a == b if isinstance(a, dict) and isinstance(b, dict) else False,
                'note': 'non-tensor or skipped capture'}
    if a.shape != b.shape or a.dtype != b.dtype:
        return {'equal': False, 'shape_a': list(a.shape), 'shape_b': list(b.shape),
                'dtype_a': str(a.dtype), 'dtype_b': str(b.dtype)}
    exact = bool(torch.equal(a, b))
    if exact:
        return {'equal': True, 'max_abs': 0., 'mean_abs': 0.}
    if a.is_complex() or b.is_complex():
        diff = (a.to(torch.complex128) - b.to(torch.complex128)).abs().double()
        scale = a.abs().double().amax().item() if a.numel() else 0.
    else:
        try:
            diff = (a.double() - b.double()).abs()
            scale = a.double().abs().amax().item() if a.numel() else 0.
        except (TypeError, RuntimeError):
            return {'equal': False, 'note': 'non-numeric tensor'}
    valid = torch.isfinite(diff)
    if not bool(valid.all()):
        return {'equal': False, 'note': 'different NaN/nonfinite values'}
    return {'equal': False, 'max_abs': float(diff.amax().item()) if a.numel() else 0.,
            'mean_abs': float(diff.mean().item()) if a.numel() else 0.,
            'scaled_max': float(diff.amax().item() / max(1., scale)) if a.numel() else 0.}


def compare_leaves(a, b):
    keys = sorted(set(a) | set(b))
    differences = []
    for key in keys:
        if key not in a or key not in b:
            differences.append({'tensor': key, 'issue': 'missing',
                                'in_trained': key in a, 'in_loaded': key in b})
            continue
        d = tensor_diff(a[key], b[key])
        if not d['equal']:
            differences.append({'tensor': key, **d})
    return {'equal': not differences, 'n_tensors': len(keys), 'differences': differences[:5]}


def capture_model(model, x, patches):
    """Forward trace in execution order; hooks are always removed."""
    import torch
    records = []
    hooks = []
    pending = defaultdict(list)
    counters = Counter()
    completion_counter = [0]

    def pre_factory(name):
        def pre(module, args, kwargs):
            idx = counters[name]
            counters[name] += 1
            record = {'name': name, 'occurrence': idx, 'class': type(module).__name__,
                      'input': tensor_leaves({'args': args, 'kwargs': kwargs}),
                      'output': {}}
            records.append(record)
            pending[name].append(record)
        return pre

    def post_factory(name):
        def post(module, args, kwargs, output):
            if not pending[name]:
                raise RuntimeError('Forward hook without matching pre-hook')
            record = pending[name].pop()
            record['output'] = tensor_leaves(output)
            record['completion_rank'] = completion_counter[0]
            completion_counter[0] += 1
        return post

    try:
        for name, mod in model.named_modules():
            if watch_module(name, mod):
                hooks.append(mod.register_forward_pre_hook(pre_factory(name), with_kwargs=True))
                hooks.append(mod.register_forward_hook(post_factory(name), with_kwargs=True))
        if not hooks:
            raise RuntimeError('No compatible model modules were found')
        model.eval()
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            output = model(context=x, num_output_patches=patches).quantile_preds
        if any(pending.values()):
            raise RuntimeError('Unclosed hooks in trace')
        return records, output.detach().cpu().clone()
    finally:
        for handle in reversed(hooks):
            handle.remove()


def summarize_traces(trained, loaded, *, cap=12):
    left = {(r['name'],r['occurrence']):r for r in trained}
    right = {(r['name'],r['occurrence']):r for r in loaded}
    # Compare *completed* modules in forward order, not their pre-hook
    # registration order. Parent blocks execute before children but complete
    # after them; postorder highlights the earliest local computation mismatch.
    trained_order = sorted(trained, key=lambda r: r.get('completion_rank', 10**12))
    loaded_order = sorted(loaded, key=lambda r: r.get('completion_rank', 10**12))
    ordered = list(dict.fromkeys([(r['name'],r['occurrence']) for r in trained_order]
                                 + [(r['name'],r['occurrence']) for r in loaded_order]))
    first = None
    offenders = []
    mismatched_calls = 0
    for rank, key in enumerate(ordered):
        if key not in left or key not in right:
            entry = {'rank': rank, 'module': key[0], 'occurrence':key[1],
                     'reason':'module invocation missing from one model'}
            mismatched_calls += 1
        else:
            a, b = left[key],right[key]
            inp = compare_leaves(a['input'], b['input'])
            out = compare_leaves(a['output'], b['output'])
            if inp['equal'] and out['equal']:
                continue
            entry = {'rank':rank, 'module':key[0], 'occurrence':key[1],
                     'module_class':a['class'], 'input':inp, 'output':out,
                     'first_divergence_inside_module': bool(inp['equal'] and not out['equal'])}
        if first is None:
            first = entry
        if len(offenders) < cap:
            offenders.append(entry)
    return {'trained_calls': len(trained), 'reloaded_calls': len(loaded),
            'missing_calls':mismatched_calls, 'first_divergence':first,
            'first_different_calls':offenders}


def lora_runtime(module):
    result={}
    for name, m in module.named_modules():
        if hasattr(m,'lora_A') and hasattr(m,'lora_B'):
            def safe(v):
                if isinstance(v,dict): return {str(k):safe(z) for k,z in v.items()}
                if isinstance(v,(list,tuple)):return [safe(z) for z in v]
                if isinstance(v,(int,float,str,bool)) or v is None:return v
                return str(v)
            result[name] = {
                'scaling':safe(getattr(m,'scaling',None)),
                'merged_adapters':safe(getattr(m,'merged_adapters',None)),
                'disable_adapters':safe(getattr(m,'disable_adapters',None)),
                'active_adapters':safe(getattr(m,'active_adapters',None)),
                'r':safe(getattr(m,'r',None)),
                'lora_alpha':safe(getattr(m,'lora_alpha',None)),
                'dropout_p':{k:float(getattr(v,'p',-1)) for k,v in getattr(m,'lora_dropout',{}).items()},
            }
    return result


def run(config, out):
    from src.experiments.adaptation_benchmark.capability import make_plan
    from src.experiments.adaptation_benchmark.capability_models import AuditedChronos2Backend
    from src.experiments.adaptation_benchmark.history import training_pools, case_window
    from src.experiments.adaptation_benchmark.inference_safety import (
        ensure_eval_mode, reload_unmerged_chronos2_adapter)
    import torch

    plan,cfg,history=make_plan(json.loads(Path(config).read_text()))
    if len(plan.records)!=4 or len(cfg['variants'])!=1 or cfg['variants'][0]['steps'] != 2:
        raise RuntimeError('Only the four-entity two-step capability smoke configuration is supported')
    v=cfg['variants'][0]
    if v['mode']!='lora':
        raise RuntimeError('Smoke must be a LoRA variant')
    out=Path(out)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    b=AuditedChronos2Backend()
    random.seed(v['seed']); np.random.seed(v['seed']); b.transformers.set_seed(v['seed'])
    pools=training_pools(plan.records,plan.snapshot.horizon,history)
    print('Training exactly 2 LoRA steps; historical development inputs only ...',flush=True)
    trained=b.base.fit(inputs=pools,prediction_length=plan.snapshot.horizon,
                       validation_inputs=None,finetune_mode='lora',context_length=cfg['context'],
                       learning_rate=v['learning_rate'],num_steps=v['steps'],batch_size=v['batch_size'],
                       output_dir=out/'train',remove_printer_callback=True,disable_data_parallel=True)
    loaded=reload_unmerged_chronos2_adapter(out/'train'/'finetuned-ckpt',device_map='cuda')
    ensure_eval_mode(trained);ensure_eval_mode(loaded)
    series,_,_=case_window(plan.records[0],plan.snapshot.horizon,cfg['context'],lag=2)
    x=torch.tensor(series,device='cuda',dtype=torch.float32)[None]
    patches=(plan.snapshot.horizon+b.base.model_output_patch_size-1)//b.base.model_output_patch_size

    before_a,before_b=trained.model,loaded.model
    runtime_a,runtime_b=lora_runtime(before_a),lora_runtime(before_b)
    differences=[{'module':name, 'trained':runtime_a.get(name), 'reloaded':runtime_b.get(name)}
                 for name in sorted(set(runtime_a)|set(runtime_b))
                 if runtime_a.get(name)!=runtime_b.get(name)]
    print(f'LORA RUNTIME STATE: layers trained={len(runtime_a)} reloaded={len(runtime_b)} differences={len(differences)}')
    for item in differences[:5]:print('  ',item['module'],str(item)[:2000])

    print('\n=== ADAPTERS ENABLED ===',flush=True)
    left,aout=capture_model(before_a,x,patches)
    right,bout=capture_model(before_b,x,patches)
    enabled=summarize_traces(left,right)
    enabled['final_output']=tensor_diff(aout,bout)
    print('OUTPUT:',enabled['final_output'])
    print('FIRST DIVERGENCE:',json.dumps(enabled['first_divergence'],indent=2))
    print('NEXT DIFFERING MODULES:',json.dumps(enabled['first_different_calls'][1:4],indent=2))

    print('\n=== BOTH ADAPTERS DISABLED ===',flush=True)
    with ExitStack() as stack:
        stack.enter_context(before_a.disable_adapter())
        stack.enter_context(before_b.disable_adapter())
        left_off,aoff=capture_model(before_a,x,patches)
        right_off,boff=capture_model(before_b,x,patches)
    disabled=summarize_traces(left_off,right_off)
    disabled['final_output']=tensor_diff(aoff,boff)
    print('OUTPUT:',disabled['final_output'])
    print('FIRST DIVERGENCE:',json.dumps(disabled['first_divergence'],indent=2))

    result={'data':{'dataset':plan.task,'n_entities':len(plan.records),'context':cfg['context'],
                    'horizon':plan.snapshot.horizon,'steps':v['steps']},
            'lora_runtime':{'n_differences':len(differences),'differences':differences[:20]},
            'enabled':enabled,'disabled':disabled,
            'precision':torch.get_float32_matmul_precision(),
            'note':'Historical development data. No lag1 scoring. No source files modified.'}
    (out/'first_divergence_report.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    print('\nSAVED:',out/'first_divergence_report.json')
    if disabled['final_output'].get('equal') and not enabled['final_output'].get('equal'):
        print('INTERPRETATION: discrepancy requires enabled adapters; investigate first divergent LoRA layer')
    elif not disabled['final_output'].get('equal'):
        print('INTERPRETATION: discrepancy persists with adapters disabled; investigate first base-model divergence')
    else:
        print('INTERPRETATION: both configurations show parity; inspect state change caused by instrumentation')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/phase3/capability_loop_smoke.json'))
    p.add_argument('--out',type=Path,default=None)
    a=p.parse_args()
    out=a.out or Path('results/phase3/checkpoint_diagnostics') / ('first_divergence_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run(a.config,out)


if __name__=='__main__':main()
