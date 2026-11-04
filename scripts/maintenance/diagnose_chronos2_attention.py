"""Diagnose Chronos-2 attention-backend mismatch without altering PersTSFM.

Runs exactly two LoRA training steps using the existing smoke configuration.
Compares identically-weighted PEFT models as saved and under forced matched
attention backends. Does not inspect or score lag1 targets.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np


def modes(model):
    """Summarise actual attention kernel selection in Chronos2 MHA modules."""
    entries = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'MHA' and hasattr(module, 'config'):
            config = module.config
            entries.append((name, str(getattr(config, '_attn_implementation', None))))
    if not entries:
        raise RuntimeError('No Chronos2 MHA modules detected; incompatible model')
    return entries


def mismatch_summary(lhs, rhs):
    a, b = dict(modes(lhs)), dict(modes(rhs))
    if set(a) != set(b):
        raise RuntimeError('Trained and loaded attention module paths differ')
    return {
        'n_attention_modules':len(a),
        'trained_modes':dict(Counter(a.values())),
        'loaded_modes':dict(Counter(b.values())),
        'mismatched_attention_modules':sum(a[k] != b[k] for k in a),
        'examples':[{'module':k,'trained':a[k],'loaded':b[k]} for k in a if a[k] != b[k]][:8],
    }


def set_attention(model, value):
    """Set all attention *layer* configs, not only base-model config.

    Chronos2Model.__init__ deepcopies config to create encoder_config,
    so setting only the root config may not affect MHA layers.
    """
    if value not in ('eager', 'sdpa'):
        raise ValueError('Unsupported Chronos2 attention implementation')
    seen = set()
    n = 0
    for _, module in model.named_modules():
        if module.__class__.__name__ == 'MHA' and hasattr(module, 'config'):
            cfg = module.config
            if id(cfg) not in seen:
                seen.add(id(cfg))
                # PretrainedConfig implements an attention property in recent
                # Transformers; use its public setter if supported.
                try:
                    cfg._attn_implementation = value
                except (AttributeError, TypeError):
                    cfg._attn_implementation_internal = value
                if getattr(cfg, '_attn_implementation', None) != value:
                    raise RuntimeError('Unable to set MHA attention implementation')
            n += 1
    if not n:
        raise RuntimeError('No MHA layers changed')
    return n


def delta(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError('Forecast shapes differ')
    diff = np.abs(x-y)
    scale = max(1., float(np.max(np.abs(x))))
    return {'max_abs':float(diff.max()), 'mean_abs':float(diff.mean()),
            'scaled_max':float(diff.max()/scale)}


def compare(trained, loaded, *, b, plan, context, x, patches):
    """Same GPU, same input, identical eval mode for direct and end-to-end."""
    import torch
    from src.experiments.adaptation_benchmark.inference_safety import ensure_eval_mode
    for model in (trained, loaded):
        ensure_eval_mode(model)
    with torch.inference_mode(), torch.autocast('cuda', enabled=False):
        t = trained.model(context=x, num_output_patches=patches).quantile_preds
        l = loaded.model(context=x, num_output_patches=patches).quantile_preds
    def predict(pipe):
        b.adapter.pipeline=pipe
        output,_,_=b.forecast(plan.records[:1], horizon=plan.snapshot.horizon,
                              context=context,lag=2,batch_size=1,cadence=plan.cadence)
        return output
    return {'direct':delta(t.detach().float().cpu(),l.detach().float().cpu()),
            'pipeline':delta(predict(trained),predict(loaded))}


def run(config_file, out):
    from src.experiments.adaptation_benchmark.capability import make_plan
    from src.experiments.adaptation_benchmark.capability_models import AuditedChronos2Backend
    from src.experiments.adaptation_benchmark.history import training_pools, case_window
    from src.experiments.adaptation_benchmark.inference_safety import reload_unmerged_chronos2_adapter
    import torch
    raw=json.loads(Path(config_file).read_text())
    plan, cfg, history=make_plan(raw)
    if len(plan.records)!=4 or len(cfg['variants'])!=1:
        raise RuntimeError('Requires four-entity one-LoRA-variant smoke config')
    v=cfg['variants'][0]
    if v['mode']!='lora' or v['steps']!=2:
        raise RuntimeError('Only two-step LoRA smoke configuration supported')
    out=Path(out)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    b=AuditedChronos2Backend()
    random.seed(v['seed']);np.random.seed(v['seed']);b.transformers.set_seed(v['seed'])
    pools=training_pools(plan.records,plan.snapshot.horizon,history)
    print('Training 2 LoRA steps for attention-backend parity analysis ...',flush=True)
    trained=b.base.fit(inputs=pools,prediction_length=plan.snapshot.horizon,validation_inputs=None,
                       finetune_mode='lora',context_length=cfg['context'],learning_rate=v['learning_rate'],
                       num_steps=v['steps'],batch_size=v['batch_size'],output_dir=out/'train',
                       remove_printer_callback=True,disable_data_parallel=True)
    loaded=reload_unmerged_chronos2_adapter(out/'train'/'finetuned-ckpt',device_map='cuda')
    x,_,_=case_window(plan.records[0],plan.snapshot.horizon,cfg['context'],lag=2)
    x=torch.tensor(x,device='cuda',dtype=torch.float32)[None]
    patches=(plan.snapshot.horizon+b.base.model_output_patch_size-1)//b.base.model_output_patch_size
    report={'attention_before':mismatch_summary(trained.model,loaded.model)}
    print('\n=== ATTENTION BACKENDS BEFORE ALIGNMENT ===')
    print(json.dumps(report['attention_before'],indent=2))
    report['original']=compare(trained,loaded,b=b,plan=plan,context=cfg['context'],x=x,patches=patches)
    print('ORIGINAL PARITY:',report['original'])
    # Force both models to the same backend; no changes to weights.
    # This is diagnostic only. This modifies in-memory configurations, never
    # serialized checkpoints or the production forecasting pipeline.
    report['matched']={}
    for attention in ('eager','sdpa'):
        count_train=set_attention(trained.model,attention)
        count_reload=set_attention(loaded.model,attention)
        result=compare(trained,loaded,b=b,plan=plan,context=cfg['context'],x=x,patches=patches)
        report['matched'][attention]={'n_trained':count_train,'n_reloaded':count_reload,**result}
        print(f'MATCHED {attention.upper()} PARITY:',result)
    report['inference_protocol']='torch.inference_mode; eval; autocast disabled; identical lag2 inputs'
    report['note']='Experimental training from PREOFFICIAL history only; no lag1 targets read'
    (out/'attention_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: diagnostic saved:',out/'attention_report.json')
    if report['attention_before']['mismatched_attention_modules']:
        print('ATTENTION CONFIGURATION MISMATCH DETECTED')
    if any(item['direct']['scaled_max'] < 1e-5 for item in report['matched'].values()):
        print('EXPLANATION SUPPORTED: parity restored with a common attention implementation')
    else:
        print('NOT RESOLVED: matched attention backend did not restore direct parity; next step is layer-by-layer hooks')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/phase3/capability_loop_smoke.json'))
    p.add_argument('--out',type=Path,default=None)
    a=p.parse_args()
    dest=a.out or Path('results/phase3/checkpoint_diagnostics')/('attention_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run(a.config,dest)


if __name__=='__main__': main()
