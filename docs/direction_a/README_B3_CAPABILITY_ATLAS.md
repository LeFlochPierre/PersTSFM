# PersTSFM Direction A — B3.0: Zero-Shot Model × Context Capability Atlas

## Research objective

Identify the best **zero-shot** forecasting model and historical context length across distinct forecasting datasets. B3 does **not** train LoRA or other adapters; that belongs to B4. The primary comparison is forecasting accuracy on identical held-out origins. Warm batch-1 latency and peak allocated GPU memory are recorded for context, **not** used to construct millisecond budget thresholds.

B3 deliberately reuses the existing `src.data.registry` dataset protocols/loaders and `src.models.adapters` model factory. It does not modify them. It adds a small independent CLI at `src.budgeted_forecasting.capability_atlas`.

### Three commands

1. `freeze`: Select a small, leakage-safe development cohort with a maximum history length. Save immutable `(entity, origin)` identities, truth/history hashes and protocol provenance in `cohort.json`. The selection uses globally shared origin timestamps, so entities do not necessarily have identical numbers of cases. All calibration labels must be mature before the first evaluation forecast.
2. `run-action`: Independently recompute and validate *exactly the frozen cases*, then run a single model × context. Each action gets its own forecast CSV, costs CSV, and SHA-256 manifest. New outputs never overwrite earlier ones. `--dry-run` checks data without loading a TSFM.
3. `report`: Require complete, target-identical, per-origin-matched shards and generate evaluation/calibration MAE (origin-weighted and series-weighted), RMSE, warm latency and per-origin hindsight wins. Evaluation labels are **not** used to train/select a model. A calibration-selected action is also displayed as a historical baseline.
4. `aggregate`: After separate dataset reports, collect within-dataset relative MAEs into a cross-dataset CSV. Raw MAE values are *not* pooled across domains.

## Installation

Place the ZIP in your local PersTSFM root. It contains only new paths; extraction with `unzip -n` will not replace existing files.

```bash
cd ~/RDS/home/projects/PersTSFM
conda activate PERTSFM_ENV
unzip -l PersTSFM_DirectionA_B3_0_capability_atlas.zip
unzip -n PersTSFM_DirectionA_B3_0_capability_atlas.zip
python -m pytest -q tests/direction_a
python -m pytest -q
git status --short
```

## Electricity smoke pilot (8 entities; up to 14 days of history)

```bash
OUT=results/direction_a/b30_electricity_8
COHORT="$OUT/cohort.json"
mkdir -p "$OUT/shards"

# Data-only cohort freeze; no GPU.
python -m src.budgeted_forecasting.capability_atlas freeze \
  --dataset electricity --protocol-split validation \
  --entities 8 --calibration-origins 3 --evaluation-origins 3 \
  --max-context-hours 336 --seed 20261102 \
  --output-dir "$OUT"

# Check model and target reconstruction; still no GPU.
python -m src.budgeted_forecasting.capability_atlas run-action \
  --cohort "$COHORT" --model chronos2 --context-hours 96 \
  --output "$OUT/shards/chronos2_4d.csv" --dry-run

# Cheap baseline.
python -m src.budgeted_forecasting.capability_atlas run-action \
  --cohort "$COHORT" --model seasonal_naive \
  --output "$OUT/shards/seasonal_naive.csv"

# Two Chronos-2 context lengths.
python -m src.budgeted_forecasting.capability_atlas run-action \
  --cohort "$COHORT" --model chronos2 --context-hours 96 \
  --output "$OUT/shards/chronos2_4d.csv"
python -m src.budgeted_forecasting.capability_atlas run-action \
  --cohort "$COHORT" --model chronos2 --context-hours 336 \
  --output "$OUT/shards/chronos2_14d.csv"

# Independent model architecture.
python -m src.budgeted_forecasting.capability_atlas run-action \
  --cohort "$COHORT" --model tirex2 --context-hours 96 \
  --output "$OUT/shards/tirex2_4d.csv"

# Join and evaluate the four-action capability atlas.
python -m src.budgeted_forecasting.capability_atlas report \
  --cohort "$COHORT" \
  --shard "$OUT/shards/seasonal_naive.csv" \
  --shard "$OUT/shards/chronos2_4d.csv" \
  --shard "$OUT/shards/chronos2_14d.csv" \
  --shard "$OUT/shards/tirex2_4d.csv" \
  --output-dir "$OUT/report_four_actions"
```

**Follow-on models:** run each independently with `--model moirai2_small`, `toto2_22m`, `toto2_313m`, or `timesfm3` and a context registered for the dataset. Model dependencies and compatible GPUs may require the appropriate PersTSFM environment; don't blindly reinstall packages into `PERTSFM_ENV`. For each extra model, use a new shard filename and a new report output directory. Supported model keys are listed in `python -m src.budgeted_forecasting.capability_atlas run-action --help`.

## Then broaden across datasets

Only after the electricity pilot passes, freeze separate cohorts using the registry's own protocols and context limits:

| Dataset | Forecast horizon | Candidate contexts | Example maximum |
|---|---:|---|---:|
| `electricity` | 96 × 15 minutes | 4d, 14d, optionally 30d | 336 h |
| `pems_bay` | 12 × 5 minutes | 24h, 72h, 168h | 168 h |
| `solar` | 6 × 10 minutes | 6h, 24h, 96h | 96 h |
| `hupa_ucm` | 12 × 5 minutes | 6h, 24h, 72h | 72 h |
| `vitaldb` | 15 × 1 minute | 0.5h, 2h, 4h | 4 h |
| `physionet2019` | 6 × 60 minutes | 6h, 24h, 48h | 48 h |

These are supported *registry context lengths*, not evidence that every model accepts the maximum. Check each adapter/model's context and prediction-length constraints. A cohort with non-overlapping entity-specific time ranges, insufficient history or a non-regular observation grid may fail preflight; that is intentional. Do not silently drop failures to make model scores look better.

For example, after validating solar's protocol:

```bash
python -m src.budgeted_forecasting.capability_atlas freeze \
  --dataset solar --entities 6 --calibration-origins 3 \
  --evaluation-origins 3 --max-context-hours 96 \
  --seed 20261102 --output-dir results/direction_a/b30_solar_6
```

For shorter-context datasets (`vitaldb`), use `last_value` rather than `seasonal_naive`, which needs a complete daily period. The aggregate tool prefers `seasonal_naive` if available and falls back to `last_value`.

### Cross-dataset aggregation

```bash
python -m src.budgeted_forecasting.capability_atlas aggregate \
  --report results/direction_a/b30_electricity_8/report_four_actions/report.json \
  --report results/direction_a/b30_solar_6/report/report.json \
  --output-dir results/direction_a/b30_cross_dataset
```

Only use paths that actually exist; the solar example is a future run. Different datasets may have different supported context actions. The cross-dataset CSV describes within-dataset relative MAE and makes no global model ranking claim.

## Important interpretation and limitations

- The previously inspected B0–B2 electricity data is **development data**. B3 electricity comparisons must remain exploratory until independently replicated on untouched tasks/datasets.
- The first B3 implementation intentionally uses small development cohorts. It freezes globally shared forecast times to ensure a single pooled calibration/evaluation cutoff. It is *not* yet a general per-entity cross-temporal benchmark for every clinical dataset.
- Warm, single-origin batch-1 inference timing excludes model loading and amortised training cost. Very small millisecond differences should not be the centre of the scientific claim.
- Aggregating MAE across different scales is invalid; report per-dataset results and within-dataset ratios. Model comparison always uses the exact same series, origins and horizon.
- `report` includes per-origin evaluation winners strictly for **hindsight diagnostics**. These cannot be used to train a selector without a separate future holdout.
- No missing-case tolerance: model actions that fail on some origins are *not* silently compared on a reduced easy subset.
- The model checkpoint revision, package versions, accelerator and precise environment should be pinned before final experiments. The output manifest records the available model ID, source commit, warmup/load time, and GPU.
- This B3.0 package was unit-tested using a mocked registry and real B0–B2 logic, but has **not** been run on your RTX6 datasets or downloaded model checkpoints. Your local test and smoke runs are necessary.

## Research decisions after B3

B3.1: increase entities/origins, add supported models and at least three domains; report performance by context budget and model size, per-series win fractions, and uncertainty over independent entities/datasets.

B3.2: create a defensible zero-shot capability map and quantify the best possible per-series model/context improvements without fitting selection policies to evaluation outcomes.

B4: reuse the matched, frozen settings for no-adaptation vs LoRA / lightweight / full fine-tuning experiments. Strictly respect prior-history budgets and charge training cost amortised over future forecasts.
