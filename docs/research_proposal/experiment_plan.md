# Cheapest falsification experiment and expansion gates

## Data and access

Pilot: [UCI Electricity Load Diagrams](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014), 370 clients at 15-minute cadence. Audit pre-enrolment zero fill and daylight-saving artefacts before origins are frozen. Independent energy validation: [London Smart Meters](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d), 5,567 households, half-hourly readings from November 2011 to February 2014; audit missingness, dToU tariff subgroup, licence and source overlap. This is **within energy**, not cross-domain generalisation. Glucose cohorts are optional external tests only after access/cadence/subject-count audits. Do not download data for this plan.

## Minimal protocol

Freeze one simple global forecaster trained on training entities only. Split entities into train, validation and untouched test sets by identifier; fit all priors, gates and hyperparameters on train/validation entities. For a test entity reveal a chronological prefix of 1, 3, 7 and 14 days where feasible; adapt from prefix only. Forecast prespecified nonoverlapping future origins at 1-hour, 1-day and 1-week horizons where the dataset supports them. A forecast may use an outcome only after that outcome's time has passed. Ensure the global input history is identical across corrections. Repeat across seeds/splits on development, then lock one external evaluation. Synthetic controls: random intercepts, AR states with varying reversion, shocks and source-to-target mismatch.

## Arms and ablations

Global frozen; seasonal naive; individual-only regularised lag model; fixed residual correction; exponentially decayed correction; fitted empirical Bayes; Kalman/hierarchical state-space; generic global/local gate; equal-capacity horizon-aware gate; POSL-style ensemble; design B; and A as a reference. Match train entities, residual features, validation budget and parameter count where possible. Compare B with its persistence summary removed; compare stable-only, transient-only and full correction. A proper comparison with LEAF/TMAML is conditional on an aligned task and feasible faithful implementation.

## Outcomes and uncertainty

Primary: entity-macro MAE and paired benefit versus global by history and horizon. Secondary: RMSE; if a method produces predictive distributions, CRPS or weighted quantile loss, coverage and width; fraction harmed, worst-decile entity deterioration, runtime and memory. Estimate uncertainty by resampling independent entities, with sensitivity to common calendar shocks; do not count overlapping windows as independent. Report prediction of benefit separately from forecasting accuracy. Predeclare a meaningful gain/harm threshold using validation entities.

## Gates

1. **Phenomenon:** if persistence varies little or horizon changes do not alter useful correction on synthetic controls and real test entities, stop H1.
2. **Simple-baseline challenge:** if EB/Kalman plus tuned residual decay matches B within uncertainty at all relevant early-history/horizon strata, pivot.
3. **Mechanism:** if B only beats a gate by extra capacity/features or loses its advantage when persistence summaries are ablated, reject the claimed mechanism.
4. **Transfer:** if gain occurs only on one source or disappears in London, retain as exploratory and do not claim generality.

No large sweep before gates 1–3 pass on a development slice.
