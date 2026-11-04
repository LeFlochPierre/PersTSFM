"""Audited Chronos-2 fine-tuning backend: weights, optimiser, checkpoint, crops.

GPU imports lazy. The original Phase 3.4 model adapter stays unchanged.
The backend is intentionally strict: a failed checkpoint reload is not hidden.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import random
import time
import numpy as np
import pandas as pd
from .history import training_pools
from .models import Chronos2Backend
from .artifacts import write_json
from .inference_safety import (ensure_eval_mode, normalize_post_training_pipeline, compare_checkpoint_parameters,
                               reload_unmerged_chronos2_adapter, verify_unmerged_parameter_parity)


def trainable_summary(model):
    params=list(model.named_parameters())
    trainable=int(sum(p.numel() for _,p in params if p.requires_grad))
    total=int(sum(p.numel() for _,p in params))
    if total<1 or trainable<1:
        raise RuntimeError('Fine-tuned model has no trainable parameters')
    return {'trainable_parameters':trainable,'total_parameters':total,
            'trained_parameters_fraction':trainable/total}


def lora_weight_change(model):
    """PEFT LoRA B starts zero in default Chronos-2 LoRA initialisation.

    A non-zero LoRA-B tensor certifies a learned effective adapter update.
    Used with default PEFT initialization only (custom LoRA init unsupported).
    """
    pairs=[(name,p.detach()) for name,p in model.named_parameters() if 'lora_B' in name]
    if not pairs:
        raise RuntimeError('No lora_B parameters: wrong model/adaptation mode')
    total_nonzero=sum(int((p != 0).sum().item()) for _,p in pairs)
    max_abs=max(float(p.abs().max().item()) for _,p in pairs)
    return {'lora_B_tensors':len(pairs),'lora_B_nonzero_weights':total_nonzero,
            'lora_B_max_abs':max_abs,'verified_weight_update':bool(total_nonzero>0 and max_abs>1e-10)}


def full_weight_change(fitted_model,base_model):
    """Compare weights by matching state-dict keys, not prediction change alone."""
    base=dict(base_model.named_parameters())
    changed=0; scanned=0; maximum=0.
    for name,param in fitted_model.named_parameters():
        if name not in base:
            continue
        orig=base[name]
        if orig.shape!=param.shape:
            continue
        delta=(param.detach().float() - orig.detach().float()).abs().max().item()
        maximum=max(maximum,float(delta));scanned+=1
        if delta>0:
            changed+=1
    return {'compared_parameter_tensors':scanned,'changed_parameter_tensors':changed,
            'max_parameter_abs_change':maximum,
            'verified_weight_update':bool(scanned>0 and changed>0 and maximum>1e-10)}


class LearningRateTrace:
    """Compatibility wrapper that dynamically builds an HF TrainerCallback."""
    @staticmethod
    def create():
        from transformers import TrainerCallback
        class RecordingCallback(TrainerCallback):
            def __init__(self):
                self.steps=[]
            def on_step_end(self,args,state,control,**kwargs):
                step=int(state.global_step)
                scheduler=kwargs.get('lr_scheduler')
                optimizer=kwargs.get('optimizer')
                lr=None
                if scheduler is not None:
                    vals=scheduler.get_last_lr()
                    if vals:lr=float(vals[0])
                if lr is None and optimizer is not None:
                    lr=float(optimizer.param_groups[0]['lr'])
                if lr is None:
                    raise RuntimeError('Chronos Trainer callback cannot observe learning rate')
                self.steps.append({'step':step,'learning_rate':lr})
        return RecordingCallback()


def validate_lr_trace(trace,expected_steps):
    if len(trace)!=expected_steps:
        raise RuntimeError(f'Expected {expected_steps} optimizer steps, observed {len(trace)}')
    steps=[r['step'] for r in trace]
    lrs=np.asarray([r['learning_rate'] for r in trace],dtype=float)
    if steps != list(range(1,expected_steps+1)) or not np.isfinite(lrs).all() or np.any(lrs<0):
        raise RuntimeError('Invalid learning-rate schedule/step trace')
    return {'observed_steps':expected_steps,'first_step_lr':float(lrs[0]),
            'last_step_lr':float(lrs[-1]),'max_observed_lr':float(lrs.max())}


class AuditedChronos2Backend(Chronos2Backend):
    def __init__(self):
        super().__init__()
        self._last_training=None

    def fit_audited(self,records,*,horizon,context,history,mode,steps,seed,batch_size,
                    learning_rate,output_dir,trace_sampler,reload_check,cadence,forecast_batch_size):
        if mode not in ('lora','full'):
            raise ValueError('Only full or LoRA fine-tuning supported')
        torch=self.torch
        output_dir=Path(output_dir)
        if output_dir.exists():
            raise FileExistsError(f'Previous attempted checkpoint exists, inspect first: {output_dir}')
        output_dir.parent.mkdir(parents=True,exist_ok=True)
        # No implicit reuse of a previously trained pipeline.
        self.adapter.pipeline=self.base
        random.seed(seed);np.random.seed(seed);self.transformers.set_seed(seed)
        pools=training_pools(records,horizon,history)
        callback=LearningRateTrace.create()
        from contextlib import nullcontext
        from .sampler_audit import trace_chronos_draws,save_observed_draws
        context_manager=trace_chronos_draws() if trace_sampler else nullcontext(None)
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        begin=time.perf_counter()
        with context_manager as trace:
            trained=self.base.fit(inputs=pools,prediction_length=horizon,
                validation_inputs=None,finetune_mode=mode,context_length=context,
                learning_rate=learning_rate,num_steps=steps,batch_size=batch_size,
                output_dir=output_dir,remove_printer_callback=True,
                disable_data_parallel=True,callbacks=[callback])
        forward_wrapper_removed=normalize_post_training_pipeline(trained)
        torch.cuda.synchronize()
        duration=float(time.perf_counter()-begin)
        peak=float(torch.cuda.max_memory_allocated(0)/1048576.)
        if mode=='lora':
            if not isinstance(trained.model,self.peft_cls):
                raise RuntimeError('Requested LoRA but got no PEFT model')
            weight=lora_weight_change(trained.model)
        else:
            if isinstance(trained.model,self.peft_cls):
                raise RuntimeError('Requested full tuning but received LoRA')
            weight=full_weight_change(trained.model,self.base.model)
        train=trainable_summary(trained.model)
        schedule=validate_lr_trace(callback.steps,steps)
        audit={**weight,**train,**schedule,'mode':mode,'seed':seed,
               'accelerate_forward_wrapper_removed':forward_wrapper_removed,
               'learning_rate_requested':learning_rate,'batch_size':batch_size,
               'training_wall_seconds':duration,'peak_allocated_mib':peak,
               'checkpoint':str(output_dir/'finetuned-ckpt'),
               'verified_checkpoint_reload':False,'fit_returned_peft':isinstance(trained.model,self.peft_cls)}
        if not audit['verified_weight_update']:
            raise RuntimeError('Weights did not show a meaningful update; do not accept forecast')
        # Keep a strictly immutable record: later functions read exactly this CSV.
        output_dir.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(callback.steps).to_csv(output_dir/'learning_rate_trace.csv',index=False)
        if trace_sampler:
            stats=save_observed_draws(trace,output_dir,expected_steps=steps)
            if stats['draws_observed']!=steps*batch_size:
                raise RuntimeError('Unexpected Chronos sample count; cannot certify gradient coverage')
            audit['sampling_trace']=stats
        else:
            audit['sampling_trace']='DISABLED'
        checkpoint=output_dir/'finetuned-ckpt'
        if not checkpoint.is_dir() or not any(checkpoint.iterdir()):
            raise RuntimeError('Chronos did not save a usable checkpoint folder')
        if reload_check:
            # The trained model is an active PeftModel. The official Chronos-2
            # from_pretrained path auto-merges it, changing the numerical operation.
            # Audit trained vs an *unmerged* reload first; separately measure the
            # official merged-model drift without using it for experiment scores.
            self.adapter.pipeline = trained
            ensure_eval_mode(trained)
            ref,_,_=self.forecast(records[:1],horizon=horizon,context=context,lag=2,
                                  batch_size=1,cadence=cadence)
            ref_repeat,_,_=self.forecast(records[:1],horizon=horizon,context=context,lag=2,
                                         batch_size=1,cadence=cadence)
            loaded = reload_unmerged_chronos2_adapter(checkpoint,device_map='cuda')
            ensure_eval_mode(loaded)
            unmerged = loaded
            self.adapter.pipeline = unmerged
            actual,_,_=self.forecast(records[:1],horizon=horizon,context=context,lag=2,
                                     batch_size=1,cadence=cadence)
            scale=max(1.,float(np.max(np.abs(ref))))
            delta=float(np.max(np.abs(ref-actual)))
            repeat_delta=float(np.max(np.abs(ref-ref_repeat)))
            audit['checkpoint_reload_representation']='active_unmerged_peft'
            audit['trained_model_class']=str(type(trained.model))
            audit['loaded_model_class']=str(type(unmerged.model))
            audit['trained_model_eval']=not trained.model.training
            audit['loaded_model_eval']=not unmerged.model.training
            audit['repeated_original_max_abs_forecast_delta']=repeat_delta
            audit['repeated_original_scaled_delta']=repeat_delta/scale
            audit['checkpoint_reload_max_abs_forecast_delta']=delta
            audit['checkpoint_reload_relative_max_delta']=delta/scale
            comparison=compare_checkpoint_parameters(trained.model,unmerged.model)
            audit['checkpoint_parameter_comparison']=comparison
            audit['verified_checkpoint_reload']=bool(delta/scale<=1e-3 and
                                                    repeat_delta/scale<=1e-3)
            try:
                verify_unmerged_parameter_parity(comparison,max_abs_tolerance=1e-6)
            except RuntimeError as exc:
                audit['unmerged_parameter_parity_error']=str(exc)
                audit['verified_checkpoint_reload']=False
            # Quantify the official merged checkpoint's numerical drift; this
            # never substitutes for the unmerged parity pass or alters scoring.
            try:
                from chronos import Chronos2Pipeline
                merged=Chronos2Pipeline.from_pretrained(str(checkpoint),device_map='cuda')
                ensure_eval_mode(merged)
                self.adapter.pipeline=merged
                merged_pred,_,_=self.forecast(records[:1],horizon=horizon,context=context,lag=2,
                                              batch_size=1,cadence=cadence)
                audit['official_merged_reload_scaled_delta']=float(
                    np.max(np.abs(ref-merged_pred)))/scale
                audit['official_merged_model_class']=str(type(merged.model))
            except Exception as exc:
                # The merged representation is only a diagnostic; keep
                # the causal unmerged verification strict and independent.
                audit['official_merged_reload_error']=repr(exc)
            # Use the verified reloaded, *unmerged* LoRA pipeline for both
            # lag2 and lag1 predictions. It is not merged into the base model.
            self.adapter.pipeline=unmerged
            if not audit['verified_checkpoint_reload']:
                write_json(output_dir/'capability_audit_failed.json',audit)
                raise RuntimeError(
                    f'Unmerged PEFT checkpoint reload parity FAILED '
                    f'(scaled max diff {delta/scale:.5g}; parameter diagnostics in '
                    f'{output_dir}/capability_audit_failed.json)'
                )
        else:
            # Defer strict scientific claim until a later reload-audited pilot.
            self.adapter.pipeline=trained
            audit['verified_checkpoint_reload']=False
            raise RuntimeError('Phase 3.6 refuses unverifiable checkpoint reuse: set reload_check=true')
        write_json(output_dir/'capability_audit.json',audit)
        print(f"ADAPTATION AUDIT: {mode} steps={steps} trainable={train['trainable_parameters']:,} "
              f"weight_change={weight['verified_weight_update']} reload={audit['verified_checkpoint_reload']} "
              f"LR {schedule['first_step_lr']:.2e}->{schedule['last_step_lr']:.2e}",flush=True)
        return duration,peak,audit
