# B2.2 — Evaluation-only cost and all-fixed-action audit

## Motivation

At an approximately 20.568 ms budget, the B2 ridge router improves over the **calibration-selected** fixed action (seasonal naive), which is orders of magnitude faster. That is not an equal-cost improvement. This audit compares the previously frozen decisions against **every eligible constant action** at its actual evaluation latency, including always using TiRex-2.

No hyperparameters or decisions are changed, and no GPU models are run. The analysis does not construct a new policy. The evaluation-best fixed action is labeled a **retrospective diagnostic**, since selecting it using evaluation outcomes would leak future information.

## Install

From the PersTSFM root, after downloading the ZIP:

```bash
unzip -n PersTSFM_DirectionA_B2_2_cost_audit.zip
python -m pytest -q tests/direction_a
python -m pytest -q
```

## Run

```bash
BASE=results/direction_a/b21_electricity_53_multimodel
PRED=results/direction_a/b21_predictability_53
python -m src.budgeted_forecasting.cost_audit \
  --input-cache "$BASE/forecast_cache.csv" \
  --feasibility-json "$BASE/feasibility.json" \
  --predictability-json "$PRED/predictability.json" \
  --decisions "$PRED/decisions.csv" \
  --output-dir results/direction_a/b22_electricity_cost_audit
```

## Outputs

- `per_action.csv`: calibration and evaluation MAE and latency for every eligible static action.
- `policy_frontier.csv`: fixed/per-series/ridge/oracle action choices and actual incurred latency; retrospective evaluation-best constant expert is explicitly labeled non-deployable.
- `ridge_excess.csv`: per-case loss and latency differences for evaluating error concentration.
- `audit.json`: counts, provenance SHA256 digests and limitations.

## Interpretation

Do **not** claim improved accuracy per unit compute from fixed-vs-ridge MAE alone: compare entire policy frontier (MAE and *actual runtime*) and optionally quantify cost for history extraction and router inference. No results here are held-out, independently replicated or cross-dataset. With 52 evaluation entities (103 origins), the development findings remain exploratory. Next experiments should freeze a conservative switching rule using grouped/temporal inner validation within calibration and then evaluate on untouched datasets.
