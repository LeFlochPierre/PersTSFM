# PersTSFM Phase 3.6: post-training mixed-precision wrapper cleanup

## Evidence and root-cause hypothesis

The first differing Chronos-2 module after LoRA training was `input_patch_embedding.hidden_layer`. Its inputs and weights were identical but the in-memory trained model produced `torch.bfloat16` outputs while the freshly loaded model produced `torch.float32` outputs. This divergence persisted with LoRA disabled and with aligned attention backends. Named parameters and buffers were identical.

Chronos-2's `fit()` enables `bf16=True` on supported GPUs and trains through Hugging Face Trainer/Accelerate. Accelerate can install a forward wrapper with `_original_forward` to run autocast. This wrapper can remain on the model object returned from training. It is not serialized as model weights, so loading a checkpoint can restore FP32 inference while the in-memory model still uses BF16 internally.

This release removes the retained forward wrapper **after `fit()` returns** using `accelerate.utils.extract_model_from_parallel(model, keep_fp32_wrapper=False)`. It then calls `model.eval()` and retains the strict checkpoint equality check. It **does not** change optimizer steps, training precision, sampling, data splits, loss calculation, checkpoint weights, or historical results.

## Installation

```bash
cd ~/RDS/home/projects/PersTSFM
conda activate PERTSFM_ENV
unzip -n PersTSFM_phase3_6_accelerate_forward_fix.zip
python scripts/maintenance/install_phase36_accelerate_cleanup.py
python scripts/maintenance/install_phase36_accelerate_cleanup.py --apply
python -m pytest -q tests/phase2 tests/phase3 tests/maintenance/test_phase36_accelerate_cleanup.py
```

Installation performs a dry-run by default, verifies exact code patterns, and backs up all modified files under `archive/patch_backups/phase36_accelerate_forward_<timestamp>/`.

## GPU verification

Use a **new output directory**. Do not reuse failed or previously scored trial directories.

```bash
set -o pipefail
python -m scripts.phase3.adaptation_capability run \
  --config configs/phase3/capability_loop_smoke.json \
  --output-root results/phase3/capability_accelerate_smoke_v4 \
  2>&1 | tee results/phase3/capability_accelerate_smoke_v4.log
```

Expected audit summary if the hypothesis is correct:

- `accelerate_forward_wrapper_removed: true` (in saved `capability_audit.json`)
- `weight_change=True`
- `reload=True`
- `checkpoint_reload_relative_max_delta <= 1e-3` (ideally close to numerical equality)

The run must still fail if the strict parity criteria are unmet. In that case, report `capability_audit_failed.json` and inspect the actual `accelerate_forward_wrapper_removed` field before proceeding.

Only after the smoke test passes, run the full capability pilot with a new output directory. Preserve previous results as historical development evidence, but do not treat LoRA comparisons made without a consistent post-training inference policy as definitive.

## Research implications

Earlier PersTSFM 3.3–3.5 LoRA forecasts may have been computed with training-time BF16 autocast, while zero-shot forecasts used standard inference precision. These are still useful exploratory observations, but performance comparisons require re-evaluation with matched inference semantics before publication.

Chronos-2 training with BF16 is valid; the issue is comparing a model with a retained training autocast wrapper against a reloaded model under a different execution precision. Do not disable BF16 training merely to force parity.

## Sources

- Chronos-2 `pipeline.py` `fit`: https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py
- Accelerate `extract_model_from_parallel` implementation: https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/other.py
