# Direction A — B1.3 Complementarity and oracle-gap diagnostics

This is an evaluation-only analysis of **existing B1.2 forecasts**. It does not load a GPU model, fit a router, or change any forecast results. It provides interpretable evidence about expert complementarity before building a learned selection policy.

## Install

From the PersTSFM repository root:

```bash
unzip -l PersTSFM_DirectionA_B1_3_diagnostics.zip
unzip -n PersTSFM_DirectionA_B1_3_diagnostics.zip
python -m pytest -q tests/direction_a
python -m pytest -q
```

The archive includes only new paths. It does not overwrite B0–B1.2, models, dataset protocols, or other scripts.

## Run on the real electricity benchmark

```bash
BASE=results/direction_a/b12_electricity_comparison
python -m src.budgeted_forecasting.diagnostics \
  --input-cache "$BASE/forecast_cache.csv" \
  --feasibility-json "$BASE/feasibility.json" \
  --bootstrap 2000 \
  --output-dir results/direction_a/b13_electricity_diagnostics
```

`diagnostics.json` summarises per-action calibration/evaluation MAE, warm inference latency, expert wins, pairwise disagreements, contributions to the oracle when each action is removed, and budget-conditional entity-cluster bootstrap intervals.

`per_series_oracle_gap.csv` reports the fixed versus oracle gap for each entity. `per_origin_oracle_gap.csv` provides transparent case-level hindsight diagnostics (not deployable predictions).

The command checks that the feasibility report agrees with the cache and refuses an existing output directory. The study should report **24 evaluation cases and eight entities** for the 8×6 electricity pilot, provided the protocol selects all eight entities at every origin.

## Interpretation and decision gate

1. Compare 14% to its entity-cluster interval: eight independent entities is a small sample.
2. Inspect whether TiRex-2 wins on cases where Chronos-2 makes larger errors.
3. Use `marginal_oracle_mae_contribution_when_action_removed` to see whether the additional TSFM improves the oracle at a matched cost threshold.
4. Inspect `evaluation_budget_overruns`, because median calibration latency is not a hard guarantee.
5. Do not use evaluation labels or these oracle winner identities as input to a learned policy; they are for diagnostic reporting. The next phase needs a fresh temporally separated development/validation protocol and only pre-forecast features.

**Important limitation:** A 14% hindsight gap is not an achievable policy gain. The budget here refers to choosing *one* action; a sequential policy that runs multiple models must pay cumulative action costs. B1.3 does not establish predictability or generalisation.

If complementarity is clear and consistent across entities, proceed to B2: learn a simple selection policy using *calibration-only labels* and pre-origin characteristics, then evaluate on a locked future block. If little complementarity remains under bootstrap or leave-one-action-out checks, broaden datasets and experts rather than prematurely implementing a complex policy.
