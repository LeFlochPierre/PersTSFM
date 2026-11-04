"""TSFM backend interface with working Chronos-2/PEFT implementation.

For future models implement forecast() and fit_lora() with identical semantics.
Importing this module never imports torch or fev; CUDA imports are lazy.
"""
from __future__ import annotations
import gc
from pathlib import Path
import random
import time
import numpy as np
from .history import case_window, training_pools
from .metrics import organise_forecast
from .inference_safety import ensure_eval_mode, normalize_post_training_pipeline

class Chronos2Backend:
    """Single base-model load; each LoRA trial calls fit() from pretrained reference."""
    def __init__(self):
        import torch, peft, transformers
        from peft import PeftModel
        from src.models.adapters import Chronos2Model, ForecastInput
        if not torch.cuda.is_available() or not transformers.utils.is_peft_available():
            raise RuntimeError('CUDA + PEFT required; will not silently run CPU/full fine-tune')
        self.torch = torch
        self.transformers = transformers
        self.peft = peft
        self.peft_cls = PeftModel
        self.ForecastInput = ForecastInput
        self.adapter = Chronos2Model()
        self.base = self.adapter.pipeline
        self.max_context = int(self.base.model_context_length)
        self.model_name = 'amazon/chronos-2'
        self.gpu = torch.cuda.get_device_name(0)

    def forecast(self, records, *, horizon, context, lag, batch_size, cadence):
        torch = self.torch
        # Forecasts at lag1 are saved, NOT evaluated until separate scoring command.
        if context > self.max_context:
            raise ValueError(f'Requested context {context} exceeds actual max {self.max_context}')
        ensure_eval_mode(self.adapter.pipeline)
        result = np.empty((len(records),horizon,4),dtype=np.float32)
        crosses = 0
        torch.cuda.synchronize()
        tic = time.perf_counter()
        for pos in range(0,len(records),batch_size):
            subset = records[pos:pos+batch_size]
            cases=[]
            for r in subset:
                x, ts, _target = case_window(r,horizon,context,lag)
                cases.append(self.ForecastInput(case_id=r.id,timestamps=ts,values=x))
            raw = self.adapter.predict_batch(cases,horizon=horizon,
                 resolution_minutes=cadence,context_steps=context,batch_size=batch_size)
            mapping = {str(p.case_id):p for p in raw}
            if len(mapping) != len(cases) or set(mapping) != {r.id for r in subset}:
                raise RuntimeError('Forecast identity mismatch or duplicates')
            for j, r in enumerate(subset):
                p = mapping[r.id]
                result[pos+j], crossed = organise_forecast(p.prediction,p.q10,p.q50,p.q90,horizon)
                crosses += crossed
        torch.cuda.synchronize()
        return result, crosses, float(time.perf_counter()-tic)

    def fit_lora(self, records, *, horizon, context, history, steps, seed,
                 batch_size, learning_rate, output_dir):
        """Fresh PEFT adapter per (history, steps, seed); pretrained kept as reference."""
        torch = self.torch
        self.adapter.pipeline = self.base
        random.seed(seed)
        np.random.seed(seed)
        self.transformers.set_seed(seed)
        pools = training_pools(records,horizon,history)
        output_dir = Path(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            raise FileExistsError(f'Refusing to reuse a LoRA attempt directory: {output_dir}')
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        tic = time.perf_counter()
        fitted = self.base.fit(inputs=pools,prediction_length=horizon,validation_inputs=None,
                 finetune_mode='lora',context_length=context,learning_rate=learning_rate,
                 num_steps=steps,batch_size=batch_size,output_dir=output_dir,
                 remove_printer_callback=True,disable_data_parallel=True)
        torch.cuda.synchronize()
        elapsed = float(time.perf_counter()-tic)
        if not isinstance(fitted.model,self.peft_cls):
            raise RuntimeError('LoRA fit did not return a PEFT model; refuse unsafe fallback')
        normalize_post_training_pipeline(fitted)
        self.adapter.pipeline = fitted
        return elapsed,float(torch.cuda.max_memory_allocated(0)/1048576.)

    def reset(self):
        self.adapter.pipeline = self.base
        gc.collect()
        self.torch.cuda.empty_cache()
