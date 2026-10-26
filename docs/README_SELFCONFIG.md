# PersTSFM Self-Configuration

PersTSFM includes an nnU-Net-inspired self-configuring forecasting framework.

Given a time-series dataset, the framework derives a leakage-safe dataset
fingerprint, constructs a feasible forecasting search space, and proposes a
small set of model/context configurations for empirical validation.

## Phase 1

Phase 1 configures:

- forecasting model:
  - Chronos-2
  - TimesFM3
  - TiRex-2
- historical context length

The model itself is not selected by a heuristic. The planner proposes
model/context candidates, which are subsequently evaluated empirically.

## Pipeline

Dataset
→ leakage-safe pre-evaluation histories
→ dataset fingerprint
→ feasible search space
→ deterministic rule-based pruning
→ candidate forecasting configurations
→ validation
→ selected configuration

## Fingerprint components

The Phase-1 fingerprint includes:

- forecast horizon and sampling frequency;
- dataset size and history availability;
- missingness and distributional properties;
- lagged autocorrelation at multiple temporal scales;
- dominant period and seasonality strength;
- spectral entropy;
- temporal persistence;
- regime/profile statistics;
- between-series heterogeneity;
- leakage-safe history capacity.

## History feasibility

Candidate contexts are constrained by available pre-evaluation history.

The robust history limit is based on the 25th percentile of history lengths.

A fixed 4096-step computational ceiling is used in Phase 1.

This ceiling is a resource constraint, not an estimate of the optimal
context length.

## Development datasets

The Phase-1 rules were developed using:

- Electricity
- PEMS-BAY
- HUPA-UCM
- VitalDB
- Solar
- PhysioNet 2019

The rules are frozen before evaluating forecasting performance.

See:

- `docs/selfconfig_protocol.md`
- `docs/selfconfig_phase1_freeze.md`

## Current entry point

Generate a fingerprint and candidate plan with:

```bash
python -m src.selfconfig.benchmark \
    --dataset electricity \
    --fingerprint-only


