# Three-model zero-shot context-sensitivity freeze

Frozen before any domain adaptation / fine-tuning experiments.

## Models

- Chronos-2
- TimesFM3
- TiRex-2

## Common-context evaluation regimes

| Domain | Recent context | Long context |
|---|---:|---:|
| PEMS-BAY | 24 h | 168 h |
| Electricity | 96 h | 192 h |
| HUPA-UCM | 24 h | 96 h |
| VitalDB | 0.5 h | 4 h |
| Solar | 24 h | 96 h |
| PhysioNet 2019 | 6 h | 48 h |

PEMS uses 168 h as the common architecture-compatible benchmark because
TiRex-2 supports a maximum context length of 2048 steps.

No TiRex-specific context length was selected by validation performance.

## Primary state-level result

Difficulty-adjusted within-entity / within-episode correlations:

| Domain | Chronos-2 vs TimesFM3 | Chronos-2 vs TiRex-2 | TimesFM3 vs TiRex-2 |
|---|---:|---:|---:|
| PEMS-BAY | 0.550 | 0.515 | 0.669 |
| Electricity | 0.544 | 0.523 | 0.641 |
| HUPA-UCM | 0.478 | 0.273 | 0.427 |
| VitalDB | 0.558 | 0.594 | 0.744 |
| Solar | 0.253 | 0.244 | 0.497 |
| PhysioNet 2019 | 0.657 | 0.612 | 0.750 |

Summary:

- 18 / 18 correlations positive.
- 18 / 18 within-entity permutation tests p < 0.05.
- Mean adjusted rho = 0.529.
- Median adjusted rho = 0.547.
- Range = 0.244 to 0.750.

The analysis:
1. normalises entity-specific gain using a fixed entity/model short-context
   denominator;
2. removes each entity's mean context sensitivity;
3. adjusts for origin-specific short-context forecasting difficulty;
4. re-centres residuals within entity;
5. tests cross-model state agreement using within-entity permutations.

## Frozen interpretation

The value of entity- or episode-specific historical context is state-dependent,
and this state dependence is reproducible across independently pretrained
time-series foundation model architectures, after controlling for stable
entity effects and short-context forecasting difficulty.

This does NOT establish that a particular latent physiological or behavioural
state has been identified. The measured object is state-dependent historical-
context value.

## Next phase

Domain adaptation / fine-tuning will test whether information obtained from
historical context can instead be absorbed into model weights.
