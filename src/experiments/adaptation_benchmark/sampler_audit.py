"""LoRA data-provenance audit; truthful about limits of tracing Chronos internals.

This audits *eligible* training-window support, not realized gradient-sampled
windows. Actual sample tracing requires a version-matched Chronos dataset sampler
hook; we refuse to infer coverage from just supplying long input arrays.
"""
from __future__ import annotations
from pathlib import Path
import inspect
import json
import numpy as np
import pandas as pd
from .history import training_pools, timestamp_ns, file_hash
from .artifacts import write_json


def pool_window_audit(records,horizon, *,context,history,validation_end=None):
    if history < max(context,horizon)+horizon or context<2 or horizon<1:
        raise ValueError('Need history >= max(context,horizon) + horizon and valid H/C')
    pools=training_pools(records,horizon,history)
    rows=[]
    for record,pool in zip(records,pools):
        # All full-history consecutive windows consisting of C input and H target,
        # strictly within the available pre-lag2 training data.
        n_windows=history-context-horizon+1
        if n_windows<1:
            raise RuntimeError('No eligible train windows')
        tail_time=record.time[-2*horizon-1]
        start_time=record.time[-2*horizon-history]
        if validation_end is not None and tail_time >= validation_end:
            raise RuntimeError('Pre-validation training time guard failed')
        if timestamp_ns(record.time[-2*horizon-1:-2*horizon])[0] >= \
           timestamp_ns(record.time[-2*horizon:-2*horizon+1])[0]:
            raise RuntimeError('Training ends after validation starts')
        rows.append({'entity_id':record.id, 'pool_points':len(pool),
                     'eligible_context_target_windows':n_windows,
                     'eligible_train_cutoffs_default_min_past_H':history-2*horizon+1,
                     'earliest_training_time':str(start_time),
                     'latest_training_time':str(tail_time),
                     'finite_training_points':int(np.isfinite(pool).sum()),
                     'first_quartile_candidate_starts':int(np.ceil(n_windows/4)),
                     'actual_draws_observed':False})
    return pd.DataFrame(rows)


def inspect_sampler_source(pipeline=None):
    """Record inspected source locations, NOT a claim about real sample frequencies."""
    if pipeline is None:
        return {'fit_source_available':False,'fit_file':None,'fit_source_sha256':None,
                'realized_sample_trace':'NOT_OBSERVED',
                'warning':'No training sampler has been instrumented; eligible windows != sampled windows'}
    try:
        path=inspect.getsourcefile(pipeline.fit)
        source=inspect.getsource(pipeline.fit)
        import hashlib
        sha=hashlib.sha256(source.encode()).hexdigest()
    except (OSError,TypeError):
        path,sha=None,None
    return {'fit_source_available':sha is not None,'fit_file':str(path) if path else None,
            'fit_source_sha256':sha,'realized_sample_trace':'NOT_OBSERVED',
            'warning':'This does not trace sampled windows; instrument installed Chronos version before claiming sampling coverage'}


def print_sampler_audit(plan, *,context,history):
    df=pool_window_audit(plan.records,plan.snapshot.horizon,context=context,history=history)
    print('CHRONOS-2 LoRA | TRAINING POOL PROVENANCE AUDIT')
    print(f'entities={len(df)} history={history} context={context} horizon={plan.snapshot.horizon}')
    print(df[['pool_points','eligible_context_target_windows']].describe().to_string(float_format=lambda x:f'{x:.1f}'))
    print('PASS: input pools end strictly before lag2; all candidate C+H windows are finite.')
    print('Chronos-2 default min_past=H; the sampler may draw shorter-than-C context windows.')
    print('NOT YET VERIFIED: distribution of windows actually sampled for gradient updates.')
    print('This is an input-eligibility audit, NOT internal Chronos-2 sampler instrumentation.')
    return df

# Opt-in realized-draw tracing: version-guarded shim on np.random.randint.
# This records the actual draws made by Chronos2Dataset._construct_slice, without
# changing their arguments, return value, or RNG state. The main training process
# must remain single-threaded during tracing (Chronos2.fit uses dataloader_num_workers=0
# in the inspected upstream implementation). Not enabled by default.
from contextlib import contextmanager
import hashlib
import sys


@contextmanager
def trace_chronos_draws(*, dataset_class=None):
    """Record observed Chronos2Dataset TRAIN crop cutoffs.

    Strict source-code signature guard prevents silently passing with a changed
    sampler implementation. Wrap ONLY one sequential training call. On completion,
    `trace.rows` contains one row for every intercepted _construct_slice crop.
    """
    if dataset_class is None:
        from chronos.chronos2.dataset import Chronos2Dataset
        dataset_class=Chronos2Dataset
    method=dataset_class._construct_slice
    try:
        source=inspect.getsource(method)
    except (OSError,TypeError) as exc:
        raise RuntimeError('Cannot inspect installed sampler source; refusing unverifiable trace') from exc
    expected='np.random.randint(self.min_past, full_length - self.prediction_length + 1)'
    # Python test fixture uses same exact crop expression.
    if expected not in source:
        raise RuntimeError('Installed Chronos sampler differs from audited _construct_slice; refusing trace')
    original=np.random.randint

    class Trace:
        rows=[]
        source_sha256=hashlib.sha256(source.encode()).hexdigest()
    trace=Trace()
    trace.rows=[]

    def observed_randint(*args,**kwargs):
        v=original(*args,**kwargs)  # exactly ONE original random draw, unchanged RNG
        frame=sys._getframe(1)
        if frame.f_code.co_name=='_construct_slice':
            obj=frame.f_locals.get('self')
            if isinstance(obj,dataset_class) and str(getattr(obj,'mode','')).lower().endswith('train'):
                idx=int(frame.f_locals['input_idx'])
                total=int(obj.inputs[idx]['context'].shape[-1])
                h=int(obj.prediction_length)
                val=int(v)
                if not (int(obj.min_past)<=val<=total-h):
                    raise RuntimeError('Observed sampler crop outside expected permissible range')
                trace.rows.append({'input_index':idx,'pool_length':total,
                                   'target_start':val,'context_start':max(0,val-int(obj.context_length)),
                                   'context_used':min(val,int(obj.context_length)),
                                   'target_end_exclusive':val+h,'horizon':h,
                                   'relative_target_start':val/total,
                                   'within_early_half':bool(val<total/2)})
        return v
    if np.random.randint is not original:
        raise RuntimeError('np.random.randint already instrumented by another tracer')
    np.random.randint=observed_randint
    try:
        yield trace
    finally:
        np.random.randint=original


def save_observed_draws(trace, output: Path, *, expected_steps=None):
    """Persist actual drawn cutoffs, preserve no-overwrite semantics."""
    output=Path(output)
    if not trace.rows:
        raise RuntimeError('No realized Chronos training crop draws captured: cannot certify sampling')
    path=output/'sampler_draws.csv'
    summary_path=output/'sampler_trace_summary.json'
    if path.exists() or summary_path.exists():
        raise FileExistsError('Sampler trace already exists')
    output.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(trace.rows)
    df.to_csv(path,index=False)
    info={'schema':'pertsfm_chronos2_actual_sampler_trace_v1',
          'draws_observed':int(len(df)),
          'distinct_entity_indices':int(df.input_index.nunique()),
          'distinct_target_start_pairs':int(df[['input_index','target_start']].drop_duplicates().shape[0]),
          'frac_earlier_half':float(df.within_early_half.mean()),
          'min_relative_target_start':float(df.relative_target_start.min()),
          'median_relative_target_start':float(df.relative_target_start.median()),
          'max_relative_target_start':float(df.relative_target_start.max()),
          'source_sha256':trace.source_sha256,
          'expected_steps':int(expected_steps) if expected_steps is not None else None,
          'draws_sha256':file_hash(path),
          'limitations':'Single run only; model-independent adaptation evidence requires multiple seeds/origins.'}
    write_json(summary_path,info)
    print(f"ACTUAL CHRONOS TRAIN CROP TRACE: n={len(df)} entities={info['distinct_entity_indices']} "
          f"early_half={info['frac_earlier_half']:.3f}",flush=True)
    print(f'PASS: actual sampling trace saved to {path}',flush=True)
    return info


class TracedChronos2Backend:
    """Delegating backend with strict, opt-in real training-draw instrumentation."""
    def __init__(self):
        from .models import Chronos2Backend
        self._inner=Chronos2Backend()
        for attr in ('max_context','gpu','model_name','torch','peft','transformers'):
            setattr(self,attr,getattr(self._inner,attr))

    def forecast(self,*args,**kwargs):
        return self._inner.forecast(*args,**kwargs)

    def reset(self):
        return self._inner.reset()

    def fit_lora(self,records,*,horizon,context,history,steps,seed,batch_size,learning_rate,output_dir):
        with trace_chronos_draws() as trace:
            result=self._inner.fit_lora(records,horizon=horizon,context=context,
                  history=history,steps=steps,seed=seed,batch_size=batch_size,
                  learning_rate=learning_rate,output_dir=output_dir)
        save_observed_draws(trace,Path(output_dir),expected_steps=steps)
        return result
