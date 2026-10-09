# PersTSFM Development Freeze

Date: 2026-10-09

This document freezes the development-stage hypotheses, primary analyses,
context choices, and interpretation before any held-out test evaluation.

## Central question

What information do frozen time-series foundation models acquire from
historical context, and when does that information become entity-specific?

## Main hypotheses

### H1 — Long-context benefit
Additional historical context can improve frozen TSFM forecasting beyond
short recent context.

### H2 — Entity-specific context information
Part of the long-context benefit depends on the older history belonging to
the correct entity.

Primary counterfactual:

- short
- same_entity
- matched_other_affine

Primary entity-specific effect:

G_entity = MAE(matched_other_affine) - MAE(same_entity)

Primary generic-history effect:

G_generic = MAE(short) - MAE(matched_other_affine)

### H3 — Context regimes differ across domains
Domains can differ in the balance between transferable process-level
information and entity-specific information.

Development evidence:

- PEMS-BAY: mixed generic + entity-specific regime.
- Electricity: strongly entity-dominated regime.
- HUPA-UCM: entity-dominated pattern, with smaller N and therefore
  greater uncertainty.

### H4 — Information acquisition is staged
On PEMS-BAY, generic information is acquired at shorter characteristic
context scales than entity-specific information.

Population-level acquisition scale is considered meaningful.

Per-entity acquisition tau is NOT treated as a primary result because
individual fits were unstable.

### H5 — Entity identifiability predicts personalization strength
Pre-evaluation distinctiveness relative to within-entity variability is
associated with larger entity-specific context benefit.

This association replicated across Chronos-2 and TimesFM 3.

The current evidence supports personalization strength, NOT reliably
per-entity acquisition speed.

### H6 — Controlled heterogeneity and noise determine entity dependence
In synthetic hierarchical time series:

- increasing between-entity heterogeneity increases entity-specific gain;
- increasing within-entity noise generally attenuates entity-specific gain;
- h^2 / sigma^2 strongly organizes the phase surface;
- the phase surface reproduces across Chronos-2 and TimesFM 3.

Primary synthetic outcome:

G_entity = MAE(other_entity) - MAE(same_entity)

Entity-contribution ratio is secondary because it is not bounded to [0, 1].

## Frozen real-data context settings

### PEMS-BAY

Recent context: 24 h

Chronos-2:
- long context: 240 h / 10 d

TimesFM 3:
- long context: 504 h / 21 d

### Electricity

Recent context: 96 h / 4 d

Chronos-2:
- long context: 192 h / 8 d

TimesFM 3:
- long context: 240 h / 10 d

### HUPA-UCM

Recent context: 24 h

Chronos-2:
- long context: 96 h / 4 d

TimesFM 3:
- long context: 168 h / 7 d

No context-length search will be performed on held-out test data.

## Primary intervention

matched_other_affine is the primary wrong-entity control.

Donor matching uses recent target history only.

Older donor history is affine-normalized using donor recent-history
mean/SD and target recent-history mean/SD.

Target recent history remains unchanged.

For HUPA-UCM, relative-time donor windows use the same frozen eval_index.

## Secondary intervention

same_entity_day_shuffle is secondary.

It provides evidence about temporal organization but is not used as the
primary definition of personalization.

It is structurally uninformative when only one old daily block exists
and can be noisy when only a few blocks are available.

## Primary statistical unit

Entity.

Forecast cases/origins are first aggregated within entity.

Primary uncertainty/inference:

- entity-level bootstrap confidence intervals
- paired entity-level sign-flip tests

## Development findings to be tested, not retuned, on held-out data

1. Same-entity long context improves forecasting over short context.
2. Correct-entity history outperforms matched wrong-entity history.
3. PEMS retains substantial generic-history benefit.
4. Electricity shows little robust generic-history benefit.
5. HUPA shows little robust generic-history benefit.
6. Entity-specific benefit is qualitatively reproducible across both TSFMs.

## Explicit non-claims

We do NOT claim:

- a universal optimal context length;
- that L_eff equals entity-identification time;
- reliable per-entity acquisition tau;
- that temporal ordering universally matters more than entity identity;
- that generic and entity-specific effects form an additive causal
  variance decomposition;
- that entity-contribution ratios are probabilities or bounded shares.

## Held-out discipline

Held-out outcomes have not been used during development.

Held-out evaluation will:

- use only frozen context settings;
- use frozen intervention definitions;
- use frozen statistical analyses;
- not search context length;
- not modify donor matching based on held-out results;
- not retune hypotheses after observing held-out outcomes.

