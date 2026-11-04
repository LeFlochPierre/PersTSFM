# PersTSFM Direction A — B1.2 matched multi-model extension

## Why B1.2?

The tiny B1.1 pilot (2 entities × 3 calibration + 3 evaluation origins = 12
cases) produced a 0.9% *hindsight* oracle gap at the highest measured threshold.
This is **not** enough to conclude that adaptive computation is unhelpful:
the sample is tiny, and the competing neural actions use the same Chronos-2
checkpoint with different context lengths.

B1.2 adds independently executed TSFM actions to the **exact same selected
forecast origins** and their targets. It never overwrites B1.1 outputs and
never assumes that different models share a compatible Python environment.

It uses the already-existing PersTSFM `build_model()` and
`ForecastInput`/`ForecastOutput` adapter contract. No new model is trained.

## Install and verify

From the repository root:

```bash
unzip -l PersTSFM_DirectionA_B1_2_multimodel.zip
unzip -n PersTSFM_DirectionA_B1_2_multimodel.zip
python -m pytest -q tests/direction_a
python -m pytest -q
```

Only new paths are included in the ZIP. The package does not change the B0
cache code, B1.1 runner, existing model wrappers, or Git settings.

## Step A — Increase the number of real cases

The original 12-case pilot is a smoke test, not robust evidence. First inspect
whether the existing *validation* protocol supports more origin times:

```bash
python -m src.budgeted_forecasting.real_atlas preflight \
  --entities 8 --calibration-origins 3 --evaluation-origins 3
```

If this succeeds, use a new output directory (never overwrite smoke results):

```bash
python -m src.budgeted_forecasting.real_atlas run \
  --entities 8 --calibration-origins 3 --evaluation-origins 3 \
  --output-dir results/direction_a/b1_electricity_8x8
```

This yields 48 cases, comprising 24 calibration and 24 evaluation cases.
If preflight fails, **inspect the message** instead of inventing origin windows
or relaxing the temporal leakage guard. Increase the entity count first, or
reduce origin counts to the number available in the protocol.

## Step B — Match the baseline before GPU inference

Using either the expanded dataset or original smoke output as the baseline:

```bash
BASE=results/direction_a/b1_electricity_8x6

python -m src.budgeted_forecasting.extend_atlas run-action \
  --base-cache "$BASE/forecast_cache.csv" \
  --base-manifest "$BASE/manifest.json" \
  --model tirex2 --context-hours 96 \
  --output results/direction_a/b12_shards/tirex2_4d.csv \
  --dry-run
```

`--dry-run` validates selected real origins, splits, all target values and
history, but **does not load a GPU model**.

## Step C — Generate another model's predictions

Run each model in a separate shell process to release resident GPU memory.
Use the Conda environment in which that model's *existing* PersTSFM adapter
works. If `tirex2` is absent in `PERTSFM_ENV`, do not install packages blindly:
use the existing TiRex-2 environment, or substitute another supported model.

```bash
python -m src.budgeted_forecasting.extend_atlas run-action \
  --base-cache "$BASE/forecast_cache.csv" \
  --base-manifest "$BASE/manifest.json" \
  --model tirex2 --context-hours 96 \
  --output results/direction_a/b12_shards/tirex2_4d.csv
```

Supported model keys are defined in the existing `build_model()` factory:
`chronos2`, `timesfm3`, `tirex2`, `toto2_22m`, `toto2_313m`, and
`moirai2_small`. Allowed contexts are 96h and 336h; unsupported checkpoint
or context combinations will fail explicitly rather than be silently altered.

The action produces:

- `<name>.csv`: B0-compatible long-form per-step prediction cache
- `<name>_costs.csv`: observed warm per-origin latency and peak CUDA memory
- `<name>_manifest.json`: model, environment, warmup and source checksums

## Step D — Merge and compare offline

```bash
python -m src.budgeted_forecasting.extend_atlas merge \
  --base-cache "$BASE/forecast_cache.csv" \
  --shard results/direction_a/b12_shards/tirex2_4d.csv \
  --output-dir results/direction_a/b12_electricity_comparison
```

To add more experts, run a new action for each model/context pair and repeat
`--shard <path>` in `merge`. The merge checks *exact* case IDs, targets,
origins, label availability, source checksums and unique action names.

The resulting `feasibility.json` and `decisions.csv` use the existing B0
fixed-action and hindsight-oracle analysis. **This is not yet a sequential
compute policy:** the oracle has access to evaluation truths, and each action's
cost is its independent warm call latency. Sequentially invoking two models
would cost at least both calls, plus overhead.

## Interpreting the next results

1. Report the number of unique entities and *evaluation* origins separately.
2. Compare per-action calibration and evaluation errors, not just the oracle.
3. Look for meaningful per-case complementary errors between independent TSFMs.
4. Compare predictive gains at matched compute (calibration-median budgets),
   and report actual budget overruns and cold-start overhead separately.
5. Do not tune model/context selection on evaluation errors, and do not claim
   performance generalisation from this pilot.
6. If the oracle gap remains tiny after adding genuinely different models,
   consider reducing the action set or revisiting the research hypothesis
   *before* building a complex learned router.

## Scope and limitations

- Does not change existing `src/models/` or raw dataset protocols.
- Performs one-origin-per-call warm profiling; not a throughput benchmark.
- Runs a single model per invocation to mitigate GPU residency and dependency
  conflicts; identical hardware is still required for meaningful cost comparisons.
- Contexts are explicitly specified; action support must be checked for each
  backbone and checkpoint revision.
- `--dry-run` and the entire `merge` step require no CUDA.
- No model-inference results are supplied with this code: they must be run on
  the real dataset using RTX6.
