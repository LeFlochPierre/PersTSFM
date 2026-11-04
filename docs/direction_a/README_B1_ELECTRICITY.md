# Direction A / B1.1: Real Electricity Forecast Atlas

**Status:** small, real-data, GPU-enabled feasibility pilot. This is not a
learned router, online policy, independent benchmark, or publication result.

This overlay adds `src/budgeted_forecasting/real_atlas.py` and a new test file.
It does **not** replace the existing B0 code, dataset registry, model adapters,
checkpoints, experiment results, or historical tests.

## Why this is the first real experiment

Validate an end-to-end **same-case** comparison of exactly three actions:

- `seasonal_naive`: repeat last 96 values (one day at 15-minute resolution)
- `chronos2_4d`: existing `amazon/chronos-2` adapter with 384 context steps
- `chronos2_14d`: same model with 1344 context steps

All actions predict the next **96 steps (one day)** from the **same** origin.
The 4-day and 14-day predictions are separate inference calls. There is
**no** ensemble, model selection learned online, or training in this stage.

## Environment assumptions

Run from the PersTSFM repository root with `PERTSFM_ENV` active. The code
imports `get_dataset_spec`, `load_protocol`, `extract_case`, `LoadedDataset`,
`_load_electricity` from `src.data.registry`, and `ForecastInput` / `build_model`
from `src.models.adapters`. These interfaces were checked against the committed
`research/phase2-tsfm` code. Your RTX6-local, **untracked** recent Phase-3 code
has not been inspected directly.

Input files from the existing registry:

- `data/processed/electricity/electricity_evaluation_origins.csv`
- `data/raw/electricity/LD2011_2014.txt`

The pilot chooses **the existing `validation` protocol** by default. It does
not regenerate forecasting origins or access an official test split. By design
it samples a small number of entities and origin timestamps, using stable SHA-256
entity ranking; `--seed` controls this.

## Installation

From the repository root:

```bash
unzip -l PersTSFM_DirectionA_B1_1_real_electricity.zip
unzip -n PersTSFM_DirectionA_B1_1_real_electricity.zip
python -m pytest -q tests/direction_a
python -m pytest -q
```

The new 14 B1 tests passed alongside the previous 17 B0 tests in a standalone
Python 3.13 environment. Run them again in your actual Python 3.11 environment.

## 1. CPU-only preflight: do this first

```bash
python -m src.budgeted_forecasting.real_atlas preflight \
  --entities 2 --calibration-origins 3 --evaluation-origins 3
```

This loads only the selected electricity columns and checks exact 14-day
history availability, horizon shape, 15-minute timestamps, finite values,
unique origins, and **pooled temporal embargo**. It never loads a TSFM.
If it fails, retain the traceback rather than silently dropping cases.

## 2. Run the GPU pilot

```bash
python -m src.budgeted_forecasting.real_atlas run \
  --entities 2 --calibration-origins 3 --evaluation-origins 3 \
  --output-dir results/direction_a/b1_electricity_smoke
```

`run` requires CUDA; it refuses a silent CPU fallback. Chronos-2 loads only
once. Both context lengths receive a separate warmup call; per-origin predictions
are performed one case at a time, batch size 1. The reported inference latency
is **CUDA-synchronised wall time**, not a CUDA-event GPU-kernel time. It
excludes model loading and warmup and is not representative of high-throughput
batched deployment. Peak GPU allocated memory (including resident model) is
recorded in a separate cost file.

## Expected files

- `forecast_cache.csv`: B0 long-form schema; one row per case/action/step
- `measured_costs.csv`: per-origin/action latency and peak allocated CUDA memory
- `manifest.json`: entity IDs, times, model/checkpoint metadata when available,
  environment/package versions, Git state, loading/warmup costs, budget thresholds
- `feasibility.json`: B0 validation-selected / cheapest / hindsight-oracle report
- `decisions.csv`: matched per-case/action decision evaluations

Outputs are under `results/` and ignored by the repository's current Git rules.
Existing files are not overwritten. Use a distinct `--output-dir` on retry.

Once the smoke pilot is valid, scale the cohort:

```bash
python -m src.budgeted_forecasting.real_atlas preflight \
  --entities 4 --calibration-origins 4 --evaluation-origins 4

python -m src.budgeted_forecasting.real_atlas run \
  --entities 4 --calibration-origins 4 --evaluation-origins 4 \
  --output-dir results/direction_a/b1_electricity_pilot4
```

## Scientific interpretation

- The chronology is **strictly globally split**: all calibration targets must
  mature before the first evaluation origin. The 15-minute series index is
  source-timezone-naive. We tag it UTC solely to meet B0's timezone-aware
  order-preserving timestamp contract; this is *not* a claim that original
  observations were recorded in UTC. `label_available_at` is conservatively
  assumed to be one sample interval after the last target. For true online
  deployment, the data release delay must be verified.
- Reported error uses native electricity units (MAE), not a normalised
  cross-domain benchmark score.
- The **hindsight oracle** sees evaluation targets and is not deployable.
  Its gap is an optimistic upper bound, not evidence of an achievable policy.
- Action eligibility is based on **median calibration** runtime. Budget
  overruns measured on evaluation are reported, not hidden.
- Exact checkpoint revision may be unavailable from the existing wrapper; the
  manifest records `checkpoint_revision_if_available` and must not claim
  checkpoint-level reproducibility if that field is null.
- This small pilot does not control for pretraining contamination, dataset
  shifts, distribution-level fairness, or batching versus model-load costs.

## Failure triage

- `FileNotFoundError`: dataset paths from `src.data.registry` unavailable.
- `Not enough calibration origins`: the selected validation protocol cannot
  supply the requested globally embargoed split. Reduce requested origin
  counts, **without** selecting evaluation outcomes based on their errors.
- `Selected entities missing calibration/evaluation`: adjust the entity count
  or deterministic seed and log that adjustment; do not silently drop series.
- `Irregular or misaligned context`: retained raw data do not match the protocol.
  Inspect the underlying timestamps before changing any extraction logic.
- `Chronos` inference error: verify current PERTSFM_ENV dependencies and the
  adapter's supported `context_steps`; don't alter the frozen adapter blindly.

## What comes after B1.1

1. Inspect actual per-action runtime and performance, with threshold budgets
   based on calibrated measured latencies.
2. Check whether the oracle gap is nontrivial on real data, then test
   predictability with a temporal-only feature selector.
3. Expand to more models/datasets only when the integrity and budget semantics
   have been validated. This B1.1 runner is purposefully **not** the general
   B1 multi-dataset engine.
