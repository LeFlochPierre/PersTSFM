"""Read-only-in-code, two-step Chronos-2 trained-versus-reloaded PEFT diagnostic.

Standalone investigation for PersTSFM Phase 3.6. It creates a NEW temporary
output directory, never changes existing runners, and never scores lag1.
Run from the repository root in PERTSFM_ENV.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import traceback

import numpy as np


def tensor_comparison(lhs, rhs):
    import torch
    a = dict(lhs)
    b = dict(rhs)
    common = sorted(a.keys() & b.keys())
    changes = []
    different_dtypes = []
    for key in common:
        x, y = a[key].detach(), b[key].detach()
        if x.dtype != y.dtype:
            different_dtypes.append({'name': key, 'trained': str(x.dtype), 'reloaded': str(y.dtype)})
        if x.shape != y.shape:
            changes.append({'name': key, 'reason': 'shape mismatch'})
            continue
        xx = x.cpu().float()
        yy = y.cpu().float()
        if not torch.equal(xx, yy):
            changes.append({'name': key, 'max_abs': float((xx - yy).abs().max().item())})
    return {
        'trained_count': len(a), 'reloaded_count': len(b),
        'missing_names': sorted(a.keys() - b.keys())[:15],
        'extra_names': sorted(b.keys() - a.keys())[:15],
        'n_value_differences': len(changes), 'value_differences': changes[:10],
        'n_dtype_differences': len(different_dtypes),
        'dtype_differences': different_dtypes[:12],
        'trained_dtypes': dict(Counter(str(p.dtype) for p in a.values())),
        'reloaded_dtypes': dict(Counter(str(p.dtype) for p in b.values())),
    }


def simple_json(x):
    if isinstance(x, dict):
        return {str(k): simple_json(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)):
        return [simple_json(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, 'to_dict'):
        return simple_json(x.to_dict())
    return str(x)


def key_delta(x, y):
    """Finite scalar diff, with stable scale; different from MAE degradation."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        return {'shape_mismatch': [list(x.shape), list(y.shape)]}
    scale=max(1.,float(np.max(np.abs(x))))
    return {'max_abs':float(np.max(np.abs(x-y))), 'scaled_max':float(np.max(np.abs(x-y)))/scale,
            'mean_abs':float(np.mean(np.abs(x-y)))}


def module_comparison(m1, m2):
    def describe(m):
        ans = {}
        for name, obj in m.named_modules():
            if hasattr(obj, 'scaling') and isinstance(obj.scaling, dict):
                ans[name] = {
                    'scaling': simple_json(obj.scaling),
                    'merged': simple_json(getattr(obj,'merged',None)),
                    'active_adapters': simple_json(getattr(obj,'active_adapters',None)),
                    'disable_adapters': simple_json(getattr(obj,'disable_adapters',None)),
                }
        return ans
    a,b=describe(m1),describe(m2)
    differences=[k for k in sorted(a.keys() & b.keys()) if a[k]!=b[k]]
    return {'train_lora_modules':len(a), 'reload_lora_modules':len(b),
            'missing_lora_modules':sorted(a.keys()-b.keys())[:12],
            'extra_lora_modules':sorted(b.keys()-a.keys())[:12],
            'n_lora_configuration_differences':len(differences),
            'lora_configuration_differences':[{ 'module':k,'train':a[k],'reload':b[k]} for k in differences[:5]]}


def runtime(torch):
    data={'torch':torch.__version__,'cuda':torch.version.cuda,
          'matmul_precision':torch.get_float32_matmul_precision(),
          'autocast_cuda_enabled':bool(torch.is_autocast_enabled('cuda')),
          'autocast_cuda_dtype':str(torch.get_autocast_dtype('cuda')),
          'cudnn_allow_tf32':bool(torch.backends.cudnn.allow_tf32)}
    try: data['cuda_matmul_allow_tf32']=bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception: data['cuda_matmul_allow_tf32']='unavailable (torch 2.9+)'
    return data


def predict(backend, pipe, records, horizon, context, cadence):
    from src.experiments.adaptation_benchmark.inference_safety import ensure_eval_mode
    backend.adapter.pipeline = pipe
    ensure_eval_mode(pipe)
    out, _, _ = backend.forecast(records[:1],horizon=horizon,context=context,
                                lag=2,batch_size=1,cadence=cadence)
    return out


def run(config, output):
    from src.experiments.adaptation_benchmark.capability import make_plan
    from src.experiments.adaptation_benchmark.capability_models import AuditedChronos2Backend
    from src.experiments.adaptation_benchmark.history import training_pools, case_window
    from src.experiments.adaptation_benchmark.inference_safety import reload_unmerged_chronos2_adapter,ensure_eval_mode
    import torch
    raw=json.loads(config.read_text())
    plan,cfg,history=make_plan(raw)
    if len(plan.records)!=4 or len(cfg['variants'])!=1:
        raise RuntimeError('Use the four-entity, one-variant smoke configuration only')
    v=cfg['variants'][0]
    if v['mode']!='lora' or v['steps']!=2:
        raise RuntimeError('Diagnostic is restricted to the two-step LoRA smoke trial')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True)
    b=AuditedChronos2Backend()
    random.seed(v['seed']);np.random.seed(v['seed']); b.transformers.set_seed(v['seed'])
    pools=training_pools(plan.records, plan.snapshot.horizon,history)
    print('Two-step training for in-memory trained-versus-reloaded model comparison ...',flush=True)
    trained=b.base.fit(inputs=pools,prediction_length=plan.snapshot.horizon,
                       validation_inputs=None,finetune_mode='lora',context_length=cfg['context'],
                       learning_rate=v['learning_rate'],num_steps=v['steps'],batch_size=v['batch_size'],
                       output_dir=output/'train',remove_printer_callback=True,disable_data_parallel=True)
    ensure_eval_mode(trained)
    checkpoint=output/'train'/'finetuned-ckpt'
    loaded=reload_unmerged_chronos2_adapter(checkpoint, device_map='cuda')
    ensure_eval_mode(loaded)
    p1=predict(b,trained,plan.records,plan.snapshot.horizon,cfg['context'],plan.cadence)
    p2=predict(b,loaded,plan.records,plan.snapshot.horizon,cfg['context'],plan.cadence)
    p1r=predict(b,trained,plan.records,plan.snapshot.horizon,cfg['context'],plan.cadence)
    p2r=predict(b,loaded,plan.records,plan.snapshot.horizon,cfg['context'],plan.cadence)
    # Compare direct model forward at exactly the same floating tensor and parameters.
    x,_,_=case_window(plan.records[0],plan.snapshot.horizon,cfg['context'],lag=2)
    x=torch.tensor(x,device='cuda',dtype=torch.float32)[None]
    patches=(plan.snapshot.horizon+b.base.model_output_patch_size-1)//b.base.model_output_patch_size
    direct={}
    try:
        with torch.inference_mode():
            d1=trained.model(context=x,num_output_patches=patches).quantile_preds
            d2=loaded.model(context=x,num_output_patches=patches).quantile_preds
        direct=key_delta(d1.detach().cpu().float().numpy(), d2.detach().cpu().float().numpy())
    except Exception as exc:
        direct={'error':repr(exc)}
    # Config and metadata with named buffers and non-persistent registered buffers.
    train_base=trained.model.get_base_model()
    load_base=loaded.model.get_base_model()
    report={
        'data':{'task':plan.task,'n_entities':len(plan.records),'horizon':plan.snapshot.horizon,
                'context':cfg['context'],'train_history':history,'steps':v['steps']},
        'runtime':runtime(torch),
        'trained_vs_loaded':key_delta(p1,p2),
        'trained_repeat':key_delta(p1,p1r),
        'reloaded_repeat':key_delta(p2,p2r),
        'raw_forward':direct,
        'parameters':tensor_comparison(trained.model.named_parameters(), loaded.model.named_parameters()),
        'buffers':tensor_comparison(trained.model.named_buffers(), loaded.model.named_buffers()),
        'lora_modules':module_comparison(trained.model,loaded.model),
        'model_training_flags':{'trained':bool(trained.model.training), 'loaded':bool(loaded.model.training)},
        'base_config_equal':simple_json(train_base.config.to_dict())==simple_json(load_base.config.to_dict()),
        'base_config_differences':[],
        'peft_config_equal':simple_json(trained.model.peft_config)==simple_json(loaded.model.peft_config),
        'chronos_config_train':simple_json(getattr(train_base,'chronos_config',None)),
        'chronos_config_reloaded':simple_json(getattr(load_base,'chronos_config',None)),
        'trained_active_adapters':simple_json(getattr(trained.model,'active_adapters',None)),
        'reloaded_active_adapters':simple_json(getattr(loaded.model,'active_adapters',None)),
    }
    c1=simple_json(train_base.config.to_dict());c2=simple_json(load_base.config.to_dict())
    report['base_config_differences']=[{'key':k,'trained':c1.get(k),'reloaded':c2.get(k)}
        for k in sorted(c1.keys()|c2.keys()) if c1.get(k)!=c2.get(k)][:20]
    dest=output/'parity_report.json'
    dest.write_text(json.dumps(report,indent=2,default=str)+'\n')
    print('\n=== PARITY DIAGNOSTICS ===')
    for k in ['trained_vs_loaded','trained_repeat','reloaded_repeat','raw_forward',
              'base_config_equal','base_config_differences','peft_config_equal',
              'trained_active_adapters','reloaded_active_adapters']:
        print(f'{k}: {report[k]}')
    for k in ['parameters','buffers']:
        x=report[k]
        print(f'{k}: values_different={x["n_value_differences"]} '
              f'dtypes_different={x["n_dtype_differences"]} '
              f'missing={len(x["missing_names"])} extra={len(x["extra_names"])}')
        if x['dtype_differences']:print('   first dtype differences:',x['dtype_differences'][:4])
    print('lora_modules:',report['lora_modules'])
    print('runtime:',report['runtime'])
    print('PASS: diagnostic saved to',dest)
    print('NOTE: no lag1 targets were read; research pipelines were not modified')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/phase3/capability_loop_smoke.json'))
    p.add_argument('--out',type=Path,default=None)
    a=p.parse_args()
    out=a.out or Path('results/phase3/checkpoint_diagnostics')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run(a.config,out)


if __name__=='__main__':main()
