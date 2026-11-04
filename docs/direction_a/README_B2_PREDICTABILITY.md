# Direction A — B2.0 history-only predictability pilot

**Research status:** exploratory. The original 8-entity electricity pilot is too small to substantiate a paper claim. B2.0 deliberately adds no new neural-model runs and makes no model adaptation or sequential-computation decisions.

## Purpose

Test whether **information available at the forecast origin** can predict which existing action has the smallest *per-origin loss*. Compare three viable policies at matched eligibility budgets:

- Global calibration-selected fixed expert (existing B0 comparator).
- Per-entity historical best expert, using the existing calibration origins only (may overfit with just 3 origins).
- History-feature ridge router, with prespecified `alpha=10`, fitted only to calibration origins. Targets are historical per-action MAEs divided by a pre-origin scale to compare series of different magnitudes.

The hindsight oracle is reported **only as an unattainable reference**.

## Data contract and leakage prevention

Uses `B1.2 forecast_cache.csv` and `feasibility.json` plus raw Electricity data through the existing `src.data.registry` loader. Each feature vector is reconstructed **strictly from the 14-day window ending at `origin_time`**, never the actual horizon. The naive timestamps from the original electricity dataset are UTC-tagged only for indexing compatibility with B1; do not interpret them as true local time. The B0 cache loader verifies that *all pooled calibration outcomes are mature before the first evaluation origin*. Both the global and per-entity policies are frozen before evaluation begins. Evaluation truth is used to score decisions, never to select actions. Budget eligibility uses *calibration median inference times*, and actual overruns are reported.

### Usage

```bash
cd ~/RDS/home/projects/PersTSFM
conda activate PERTSFM_ENV
unzip -n PersTSFM_DirectionA_B2_0_predictability.zip
python -m pytest -q tests/direction_a

BASE=results/direction_a/b12_electricity_comparison
python -m src.budgeted_forecasting.predictability \
  --input-cache "$BASE/forecast_cache.csv" \
  --feasibility-json "$BASE/feasibility.json" \
  --bootstrap 2000 \
  --output-dir results/direction_a/b20_electricity_history_router
```

Outputs: `predictability.json` (action counts, matched-budget losses, calibration costs, cluster bootstrap CI) and `decisions.csv` (one row per evaluation case per budget). Existing outputs are never overwritten.

## Interpretation

The **primary comparison** is `ridge_selected_mae` versus `fixed_mae` and `entity_selected_mae`, *not* ridge versus the hindsight oracle. An improvement on eight entities is exploratory; a negative or zero improvement is also useful evidence. Report budget overruns. Do **not** adjust features or alpha after viewing the evaluation outcomes, because that would contaminate the holdout.

If results indicate predictability, the next milestone is a substantially larger, prospectively frozen electricity benchmark (e.g. 50+ entities with more valid calibration origins), followed by a dataset-family holdout and delayed online replay. If not, test whether complementarity is only an oracle artifact before investing in a sequential method.

## Limitations

- History-feature extraction and model-loading overhead are not included in measured costs.
- A one-shot action selection is not an adaptive *sequence* of TSFM calls.
- Median calibration latency is not a hard per-origin computational budget.
- One limited dataset cannot establish cross-domain generalisation.
- The observed oracle gap is defined using future truths; it cannot be achieved by inference alone.
- The entity bootstrap does not capture model-selection or dataset-distribution uncertainty.
