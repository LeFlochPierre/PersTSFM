# PersTSFM Phase 3.6 — First forward divergence diagnostic

## Reason

Previous tests showed identical named parameters, buffers, dtypes, LoRA scaling,
repeatable in-memory predictions, and matching attention backends. Yet direct
forecasts after reloading an unmerged PEFT checkpoint differ by a scaled maximum
of approximately 0.001756. The mismatch is **not** the forecasting MAE.

## Install

From the repository root, `unzip -n PersTSFM_phase3_6_first_divergence.zip`.

```bash
conda activate PERTSFM_ENV
python -m pytest -q tests/maintenance/test_chronos2_first_divergence.py
python -m scripts.maintenance.diagnose_chronos2_first_divergence \
  --config configs/phase3/capability_loop_smoke.json \
  2>&1 | tee results/phase3/first_divergence_diagnostic.log
```

The diagnostic runs exactly two LoRA steps and captures named module outputs for
identical direct-forward inputs. It also tests both models with adapters disabled.
It never edits shared training code, reads lag1 targets, or overwrites an old run.

## Interpret

- **Disabled-adapter outputs equal; enabled outputs differ:** LoRA-specific
  runtime behaviour is responsible. Inspect `LORA RUNTIME STATE` and first
  divergent layer in `ADAPTERS ENABLED`.
- **Disabled-adapter outputs differ:** investigate the earliest divergent base
  model layer and runtime configuration, not LoRA serialization.
- **An equal input but differing output at a specific layer:** localizes the
  non-equivalent computation to that layer or its untraced inner operations.
- **The first output difference has differing inputs:** look earlier in the
  trace; some operations are functional and not instrumented as modules.

Outputs include `first_divergence_report.json` and concise terminal output.
This is *not* proof of scientific forecast performance. Do not launch full
fine-tuning until checkpoint parity is understood or a validated inferential
protocol is selected and its implications documented.
