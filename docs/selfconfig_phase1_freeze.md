# PersTSFM Self-Configuration — Phase 1 Freeze

## Status

Phase-1 self-configuration rules frozen after development on:

- electricity
- pems_bay
- hupa_ucm
- vitaldb
- solar
- physionet2019

Rules must not be modified in response to subsequent forecasting performance
on these datasets.

## Objective

Automatically construct a small, interpretable forecasting configuration
candidate set from leakage-safe pre-evaluation dataset characteristics.

Phase 1 configures:

- TSFM family:
  - Chronos-2
  - TimesFM3
  - TiRex-2
- context length

Model choice itself remains empirical: each retained model/context candidate
is evaluated on validation data.

## Self-configuration pipeline

Dataset
→ leakage-safe historical data
→ dataset fingerprint
→ feasible search space
→ deterministic rule-based pruning
→ validation
→ selected forecasting configuration

## Fingerprint components

### Forecast scale

Forecast horizon H.

### Temporal structure

- lag-1 ACF
- ACF(H)
- ACF(2H)
- ACF(4H)
- ACF(8H)
- ACF(16H)
- ACF(32H)

### Periodicity

- dominant spectral period P
- seasonality strength
- spectral entropy
- ACF(P)
- ACF(2P)

### Temporal persistence

Persistence scales derived from multi-scale ACF.

### Regime / profile structure

- recent-vs-prior mean shift
- recent-vs-prior scale shift
- recent-vs-prior IQR shift
- recent-vs-prior quantile shift
- rolling mean variability
- rolling scale variability
- adjacent mean shift
- window-mean persistence

### History capacity

Leakage-safe available context is summarized from pre-evaluation histories.

Robust history capacity:

    Q25(history length)

Effective context limit:

    min(Q25(history length), 4096 steps)

The 4096-step ceiling is a fixed Phase-1 computational budget, not an
estimate of the optimal context length.

## Frozen Phase-1 proposed plans

### Electricity

- 96 steps
- 192 steps
- 768 steps

### PEMS-BAY

- 12 steps
- 293 steps
- 1172 steps

### HUPA-UCM

- 12 steps
- 96 steps
- 1536 steps

### VitalDB

- 15 steps
- 60 steps
- 120 steps

### Solar

- 6 steps
- 144 steps
- 575 steps

### PhysioNet2019

- 6 steps
- 24 steps
- 47 steps

The same three context candidates are currently evaluated for each TSFM.

## Scientific rule

These six datasets constitute the Phase-1 development suite.

After this freeze, forecasting results may be used to evaluate the
self-configuration system but must not be used to retroactively alter its
rules.

Primary evaluation quantities will include:

1. forecasting performance;
2. regret relative to exhaustive validation search;
3. candidate/search-space reduction;
4. configuration-selection agreement;
5. computational cost.

Later experiments should evaluate the frozen configurator on held-out
datasets/domains not used to construct these rules.
