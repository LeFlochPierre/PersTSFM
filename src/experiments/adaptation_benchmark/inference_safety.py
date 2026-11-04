"""Strict model-mode guard for reproducible TSFM inference.

``torch.no_grad()`` and ``torch.inference_mode()`` do not switch modules
out of training mode. Fine-tuning via HF Trainer can leave a model in train
mode. Apply this guard to *every* forward pass used for evaluation.
"""
from __future__ import annotations


def ensure_eval_mode(pipeline) -> None:
    """Switch a Chronos/PEFT pipeline to eval mode, or fail explicitly."""
    model = getattr(pipeline, 'model', None)
    if model is None or not callable(getattr(model, 'eval', None)):
        raise RuntimeError('Pipeline has no model.eval(): cannot ensure deterministic evaluation')
    model.eval()
    if model.training:
        raise RuntimeError('Model remains in training mode after calling eval()')


def compare_checkpoint_parameters(trained_model, loaded_model) -> dict:
    """Diagnose whether loaded PEFT weights match in-memory weights.

    Called only on failed forecast parity. Compares named parameter tensors,
    with LoRA weights counted separately. No inference data or targets used.
    """
    import torch
    trained = dict(trained_model.named_parameters())
    loaded = dict(loaded_model.named_parameters())
    shared = sorted(set(trained).intersection(loaded))
    common_lora = [k for k in shared if 'lora_' in k.lower()]
    unequal = 0
    max_abs = 0.0
    first_unequal = None
    for name in shared:
        a, b = trained[name].detach(), loaded[name].detach()
        if a.shape != b.shape:
            unequal += 1
            if first_unequal is None:
                first_unequal = name + ' (shape mismatch)'
            continue
        d = (a.to(device='cpu', dtype=torch.float32) -
             b.to(device='cpu', dtype=torch.float32)).abs().max().item()
        if d > 0.0:
            unequal += 1
            if first_unequal is None:
                first_unequal = name
            max_abs = max(max_abs, float(d))
    return {
        'trained_parameter_tensors': len(trained),
        'loaded_parameter_tensors': len(loaded),
        'shared_parameter_tensors': len(shared),
        'shared_lora_tensors': len(common_lora),
        'missing_from_reload_count': len(set(trained) - set(loaded)),
        'additional_in_reload_count': len(set(loaded) - set(trained)),
        'unequal_shared_tensors': unequal,
        'max_abs_shared_parameter_delta': max_abs,
        'first_unequal_tensor': first_unequal,
    }


def reload_unmerged_chronos2_adapter(checkpoint, *, device_map="cuda"):
    """Reload the exact PEFT adapter without Chronos2Pipeline's automatic merge.

    The official Chronos-2 LoRA from_pretrained route calls merge_and_unload().
    That is a *different numerical implementation* from the trained PeftModel.
    Keep the adapter active for strict trained-vs-reloaded comparisons, and
    keep the returned unmerged pipeline for audited scientific evaluation.
    """
    from pathlib import Path
    ckpt = Path(checkpoint)
    if not (ckpt / "adapter_config.json").is_file():
        raise RuntimeError(f"Expected PEFT adapter_config.json in {ckpt}")
    from chronos import Chronos2Pipeline
    from peft import AutoPeftModel, PeftModel, __version__ as peft_version
    from packaging.version import Version
    kwargs = {"device_map": device_map}
    # Match the exact import allowance used by official Chronos-2.
    if Version(peft_version) >= Version("0.20.0"):
        kwargs["import_allowlist"] = ["chronos"]
    model = AutoPeftModel.from_pretrained(str(ckpt), **kwargs)
    if not isinstance(model, PeftModel):
        raise RuntimeError("Unmerged reload did not produce a PeftModel")
    pipeline = Chronos2Pipeline(model=model)
    ensure_eval_mode(pipeline)
    return pipeline


def verify_unmerged_parameter_parity(comparison, *, max_abs_tolerance=1e-6):
    """Fail closed on missing, mismatched or numerically different PEFT parameters."""
    if comparison["trained_parameter_tensors"] != comparison["loaded_parameter_tensors"]:
        raise RuntimeError("Unmerged adapter parameter counts differ")
    if comparison["missing_from_reload_count"] or comparison["additional_in_reload_count"]:
        raise RuntimeError("Unmerged reload has missing/extra parameter names")
    if comparison["shared_lora_tensors"] < 1:
        raise RuntimeError("No shared LoRA tensors in reload parity comparison")
    if comparison["max_abs_shared_parameter_delta"] > max_abs_tolerance:
        raise RuntimeError("Unmerged reload parameter values do not match")
    return True


def normalize_post_training_pipeline(pipeline) -> bool:
    """Remove Accelerate's training autocast forward wrapper *after* fit().

    The in-memory PEFT model may still use BF16 autocast from Hugging Face
    Trainer, unlike the fresh checkpoint reload. Never remove this wrapper
    during training; do so before comparing or using model predictions.

    Returns whether a retained wrapper was observed. Strict parity checks
    remain mandatory and are NOT bypassed by this helper.
    """
    model = getattr(pipeline, 'model', None)
    if model is None:
        raise RuntimeError('Chronos pipeline has no model to normalize')
    had_wrapper = '_original_forward' in vars(model)
    if had_wrapper:
        from accelerate.utils import extract_model_from_parallel
        result = extract_model_from_parallel(model, keep_fp32_wrapper=False)
        pipeline.model = result
        if '_original_forward' in vars(result):
            raise RuntimeError('Accelerate forward wrapper was not removed')
    ensure_eval_mode(pipeline)
    return had_wrapper
