# PersTSFM Phase 3.6 — Chronos-2 adaptation-capability audit

## Scientific purpose

Determine whether the *previously observed negative LoRA results* on LOOP Seattle
reflect a lack of useful adaptation or an underpowered fine-tuning schedule.
This is a **development / implementation audit**, not evidence from an independent
held-out dataset family. The LOOP Seattle historical lag-1 outcomes have already
been inspected in preceding phases.

## Reuse and chronology

This patch is ADDITIVE. It does not modify Phase 3.4/3.5 modules. It calls
`history.py`, `design.py`, `models.py`, `metrics.py`, `artifacts.py`, and
`sampler_audit.py`. It reads a fixed, checksummed Phase 3.5 fold snapshot
(`results/phase3/rolling/loop_full_nonoverlap_v1/folds/fold_00/history.npz`).
It never opens FEV official test targets and does not require the FEV data
environment. LoRA/full training consumes **only the strict pre-lag2 training
prefix**. Validation truth is lag2; lag1 forecasts are stored but not scored
until `score-later`. Selection is locked before later scoring. This does not
make the already-studied LOOP Seattle data a pristine holdout.

## What is audited

* Full from-scratch Chronos-2 `.fit()` for each explicit configuration;
  no incremental continuation from an earlier budget.
* Actual `TrainerCallback.on_step_end` observed LR by optimizer step, exported
  to `learning_rate_trace.csv`. Chronos currently uses **linear LR decay**.
  Thus changing `steps` also changes the LR schedule, an experimental caveat.
* Trainable-parameter count and fractional parameters of the returned model.
* For the stock LoRA initialization, **nonzero LoRA-B weights**, validating
  that the effective PEFT adapter was updated; for full tuning, compare model
  tensors with the frozen pretrained reference.
* Mandatory reload of the **saved fine-tuned checkpoint** and a point/quantile
  forecast parity check on a validation input. **All subsequent forecasts use
  the reloaded checkpoint** rather than the original in-memory model.
* Optional version-guarded actual crop tracing from Phase 3.5; verifies draw
  count is `num_steps * train_batch_size` and records sampled history positions.
* Training time, peak CUDA allocated memory, validation metrics and model
  performance on the same fixed entity cohort.
* Reusable checksummed per-trial cache, strict resume, frozen selection,
  separate later-period scoring.

## Installation / test

Unzip the package into repository root (use `unzip -n`).

```bash
cd ~/RDS/home/projects/PersTSFM
conda activate PERTSFM_ENV
python -m pytest -q tests/phase3/test_adaptation_capability.py
python -m pytest -q tests/phase2 tests/phase3
```

## Step 1: GPU smoke test (4 entities, LoRA 2 steps)

```bash
python -m scripts.phase3.adaptation_capability plan \
  --config configs/phase3/capability_loop_smoke.json
set -o pipefail
python -m scripts.phase3.adaptation_capability run \
  --config configs/phase3/capability_loop_smoke.json \
  2>&1 | tee results/phase3/capability_smoke.log
```

Expected terminal: zero-shot parity table, `ADAPTATION AUDIT: lora...`
with `weight_change=True reload=True`, 2 observed optimizer LR steps,
nonzero `lora_B` weights and a sampler file with 16 traced crop draws.
**The checkpoint reload may expose an installed-package compatibility issue**
not detectable by CPU tests. If it fails, stop and inspect the traceback;
do not disable the check or automatically launch the larger pilot.

## Step 2: full-entity capability pilot

Only after successful smoke, use the explicit predeclared grid: LOOP Seattle
323 entities, fine-tuning context 768, zero-shot contexts 768 and 4096;
LoRA `steps={100,500,1000}`, `LR={1e-5,3e-5}`, batch=32, seed=42.
These are **six independent training jobs**, NOT 3 seeds. We first look
for capability to improve; only repeat multiple seeds on promising schedules.
The two zero-shot validations are checked against the already-known Phase 3.5
fold-0 MAEs (C768 = 3.44538, C4096 = 3.43108), tolerance 0.02.

```bash
python -m scripts.phase3.adaptation_capability plan \
  --config configs/phase3/capability_loop_pilot.json
python -m scripts.phase3.adaptation_capability run \
  --config configs/phase3/capability_loop_pilot.json \
  2>&1 | tee results/phase3/capability_loop_pilot.log
```

If interrupted, resume an existing directory (do not delete completed trial
outputs). The printed path looks like `results/phase3/capability/<stamp>_LOOP_SEATTLE_1H_n323`:

```bash
python -m scripts.phase3.adaptation_capability run \
  --config configs/phase3/capability_loop_pilot.json \
  --resume results/phase3/capability/<ACTUAL_RUN_DIR>
```

**Do not score lag1 while making adaptation schedule choices.** After the full
validation experiment is locked, `score-later` can evaluate the selected
strategy on historical data, but it is not an unseen official FEV test.

```bash
python -m scripts.phase3.adaptation_capability score-later \
  --run results/phase3/capability/<ACTUAL_RUN_DIR>
```

### Optional full tuning

`configs/phase3/capability_loop_full_ft_optional.json` defines one full
fine-tuning sanity check (100 steps, batch=16, LR=1e-6). This is NOT a
compute-matched comparison with LoRA; do not include as a claim of superiority.
Assess GPU memory and reproducibility first. The config is separate so it
cannot silently add work to the main LoRA pilot.

## Main analysis and limitations

We need to compare against the *best zero-shot context* as well as the *same
context* zero-shot baseline. Current study has only one seed for the large
pilot, one data family, and development periods already inspected. It cannot
support significance claims or a claim that LoRA always fails/helps.

Chronos-2 `.fit()` defaults (as inspected upstream in Oct 2026) include
1000 steps, batch 256, `learning_rate=1e-6`, with `1e-5` suggested for LoRA.
Our batch32 pilot is a controlled expansion over previous batch8 experiments,
not an attempt to exactly replicate the default batch256 schedule.
Reference: https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py

If an adaptation configuration shows a repeatable improvement, follow up with
multiple independently drawn seeds, matched compute accounting, additional
unseen dataset families and another architecture. If it does not, next explore
simpler calibration before committing to complex adaptation algorithms.
