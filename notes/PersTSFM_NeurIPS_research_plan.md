# PersTSFM Research Plan: Implicit Personalization in Time-Series Foundation Models

## Working title

**Implicit Personalization in Time-Series Foundation Models: When Does History Identify the Entity?**

Alternative titles:

- **From History to Identity: Understanding In-Context Personalization in Time-Series Foundation Models**
- **The Personalization Crossover in Time-Series Foundation Models**
- **When Does Context Become Personalization? A Study of Entity Inference in Time-Series Foundation Models**

---

# 1. Central research question

The current empirical result is clear:

> Increasing same-entity historical context can substantially improve forecasting performance in frozen time-series foundation models (TSFMs), but the useful context scale varies strongly across datasets and model architectures.

The deeper question is:

> **What latent information about an unseen entity does a frozen TSFM infer from its history, how much history is required, and when is in-context personalization sufficient versus explicit parameter adaptation?**

The project should move beyond asking whether longer context helps.

The target contribution is a theory and empirical characterization of **implicit personalization** in frozen TSFMs.

---

# 2. Core hypothesis

Historical context serves at least two distinct roles:

1. **Current-state estimation**
   - What is happening to the entity now?
   - Recent level, trend, short-term dynamics, current regime.

2. **Entity identification**
   - What kind of time series is this?
   - Stable level, amplitude, periodicity, characteristic dynamics, habitual regimes, individual response patterns.

The central hypothesis is:

> **Long same-entity context improves forecasting because the frozen TSFM progressively infers entity-specific latent structure, not merely because it receives more observations.**

This motivates the interpretation:

\[
\boxed{
\text{In-context personalization}
=
\text{amortized inference of entity-specific latent structure}
}
\]

---

# 3. Latent-entity theoretical framework

Let entity \(i\) have a relatively stable latent characteristic \(\theta_i\) and a time-varying latent state \(z_{i,t}\):

\[
\theta_i \sim p(\theta),
\]

\[
z_{i,t} \sim p(z_{i,t}\mid z_{i,t-1},\theta_i),
\]

\[
y_{i,t} \sim p(y_{i,t}\mid z_{i,t},\theta_i).
\]

Given historical context

\[
C_{i,L}=y_{i,t-L+1:t},
\]

a frozen TSFM produces

\[
p(y_{i,t+1:t+H}\mid C_{i,L}).
\]

Conceptually, this can be interpreted as approximating

\[
p(y_{\mathrm{future}}\mid C)
=
\int
p(y_{\mathrm{future}}\mid \theta_i,z_t)
p(\theta_i,z_t\mid C)
\,d\theta\,dz.
\]

As context increases, the model may reduce uncertainty over:

- current state \(z_t\);
- stable entity characteristics \(\theta_i\);
- periodicity;
- amplitude;
- baseline level;
- dynamic response;
- recurrent regimes.

The empirical context curve may therefore reflect the rate at which the model identifies the entity.

---

# 4. Effective personalization context

Use the empirical threshold

\[
L_{1\%}
=
\min
\left\{
L:
\mathrm{MAE}(L)
\leq
1.01\,\mathrm{MAE}_{\min}
\right\}.
\]

Interpret cautiously:

- it is a development-set quantity;
- it measures the earliest context reaching near-best observed performance;
- it is not necessarily the true optimum;
- if the best observed result occurs at the maximum tested context, the result is right-censored.

Current development results:

| Dataset | Model | Reference | Best observed | Improvement | \(L_{1\%}\) | Right-censored |
|---|---|---:|---:|---:|---:|---|
| Electricity | Chronos-2 | 4 d | 12 d | 11.47% | 8 d | No |
| Electricity | TimesFM 3 | 4 d | 10 d | 15.99% | 10 d | No |
| PEMS-BAY | Chronos-2 | 1 d | 14 d | 39.51% | 10 d | No |
| PEMS-BAY | TimesFM 3 | 1 d | 28 d | 41.04% | 21 d | Yes |
| HUPA-UCM | Chronos-2 | 1 d | 4 d | 7.08% | 4 d | No |
| HUPA-UCM | TimesFM 3 | 1 d | 7 d | 5.98% | 7 d | Yes |

The next goal is to explain **why these scales differ**.

---

# 5. Simple theory of entity identification

Consider

\[
y_{i,t}=\mu_i+\epsilon_{i,t},
\]

with

\[
\mu_i\sim \mathcal N(0,\tau^2),
\qquad
\epsilon_{i,t}\sim\mathcal N(0,\sigma^2).
\]

After observing \(n\) samples,

\[
\mathrm{Var}(\mu_i\mid y_{i,1:n})
=
\left(
\frac{1}{\tau^2}
+
\frac{n}{\sigma^2}
\right)^{-1}.
\]

This gives a characteristic entity-identification scale

\[
n^*\sim\frac{\sigma^2}{\tau^2}.
\]

Interpretation:

- high within-entity noise \(\sigma^2\) means more history is needed;
- strong between-entity heterogeneity \(\tau^2\) makes entities easier to distinguish;
- weak between-entity heterogeneity means personalization provides less information.

For correlated time series, replace raw sample size with an effective sample size:

\[
n_{\mathrm{eff}}(L).
\]

A first theoretical prediction is therefore

\[
L_{\mathrm{eff}}
\propto
\frac{
\text{within-entity unpredictability}
}{
\text{between-entity distinguishability}
}.
\]

---

# 6. Information versus staleness

Longer history is not always beneficial.

If the entity changes over time,

\[
\theta_{i,t+1}
=
\theta_{i,t}
+
\eta_t,
\qquad
\eta_t\sim\mathcal N(0,q),
\]

then increasing context has two competing effects:

1. **Information accumulation** — more evidence about entity-specific structure.
2. **Staleness** — older observations may describe an outdated entity state.

A conceptual risk decomposition is

\[
R(L)
\approx
R_{\infty}
+
\frac{A}{I_\theta(L)}
+
B\,D(L),
\]

where:

- \(I_\theta(L)\) is information about entity identity available in context;
- \(D(L)\) is a drift or staleness penalty.

The theoretical claim becomes

\[
\boxed{
L_{\mathrm{eff}}
=
\text{the balance between entity information accumulation and context staleness}
}
\]

---

# 7. Research Question 1: Is the gain actually entity-specific?

The strongest alternative explanation for the current results is that longer context simply exposes generic temporal structure.

For example, the PEMS-BAY gain might mainly occur because the model finally sees a weekly cycle.

The next experiment should therefore causally separate:

- generic temporal information;
- entity-specific information.

---

# 8. Experiment A: Context counterfactuals

Keep the most recent context fixed and manipulate only the additional older history.

## A1. Short context

Baseline:

\[
C_{\mathrm{short}}.
\]

## A2. True same-entity long context

\[
[H_i^{\mathrm{old}},H_i^{\mathrm{recent}}].
\]

This is the current long-context condition.

## A3. Matched other-entity history

\[
[H_j^{\mathrm{old}},H_i^{\mathrm{recent}}].
\]

Replace the additional historical block with history from another entity.

Match donor entities where possible on:

- mean;
- variance;
- recent level;
- dominant period;
- time-of-day / weekday alignment.

This separates generic temporal information from same-entity information.

## A4. Same entity, shuffled old history

Preserve:

- values;
- marginal distribution;
- approximate scale.

Destroy:

- temporal order;
- recurrence;
- phase.

This tests whether gains arise from entity-specific distributions or entity-specific dynamics.

## A5. Same entity, phase/frequency-disturbed history

Possible interventions:

- phase scrambling;
- removal of dominant periodic components;
- seasonal-component removal;
- low-pass or high-pass filtering.

This tests whether the gain is carried by periodic structure.

## A6. Same entity, stale history

Use the same amount of historical evidence but draw it farther into the past.

This directly measures the staleness penalty.

## A7. Period-aligned sparse history

Instead of contiguous context, selectively provide historically relevant observations:

- same hour-of-day;
- same weekday;
- same phase of the dominant cycle.

This tests whether **informative context** is more fundamental than raw context length.

---

# 9. Decompose the long-context gain

Define total long-context benefit:

\[
G_{\mathrm{long}}(L)
=
R(C_{\mathrm{short}})
-
R(C_{\mathrm{same}}).
\]

Define the entity-specific component:

\[
G_{\mathrm{entity}}(L)
=
R(C_{\mathrm{other}})
-
R(C_{\mathrm{same}}).
\]

A useful normalized measure is

\[
P_{\mathrm{entity}}
=
\frac{G_{\mathrm{entity}}}{G_{\mathrm{long}}}.
\]

Interpretation:

- \(P_{\mathrm{entity}}\approx0\): benefit is mostly generic temporal information;
- \(P_{\mathrm{entity}}\approx1\): benefit is predominantly entity-specific;
- intermediate values indicate a mixture.

This could become a central result of the paper.

---

# 10. Research Question 2: Which entities need more history?

The six dataset/model aggregate context scales are not enough for mechanistic analysis.

Estimate entity-level quantities.

For every entity \(i\),

\[
\Delta_i(L)
=
\mathrm{MAE}_i(L_{\mathrm{ref}})
-
\mathrm{MAE}_i(L).
\]

Where stable enough, define

\[
L_{\mathrm{eff},i}.
\]

Current development entities:

- Electricity: 53;
- PEMS-BAY: 49;
- HUPA-UCM: 20.

Total: **122 development entities**.

This provides enough units for entity-level explanatory analysis.

---

# 11. Entity characteristics to measure

All features must be estimated strictly from pre-evaluation history.

## Noise / predictability

- variance;
- coefficient of variation;
- residual variance after seasonal decomposition;
- entropy-rate proxy;
- forecastability;
- spectral entropy.

## Recurrence

- lag-1 autocorrelation;
- daily recurrence;
- weekly recurrence;
- dominant-period autocorrelation;
- recurrence strength at the strongest spectral peak.

## Periodicity

- dominant period;
- peak spectral power;
- dominant-frequency power / total power;
- number of strong spectral modes.

## Drift / nonstationarity

- rolling-mean drift;
- rolling-variance drift;
- change-point rate;
- distribution shift between early and late history;
- trend magnitude.

## Entity distinctiveness

Possible measures:

- distance from the population mean;
- distance in spectral-feature space;
- nearest-neighbour distance;
- within-entity versus between-entity variance ratio;
- classification confidence from hand-engineered features.

---

# 12. Entity-level explanatory model

Model either

\[
L_{\mathrm{eff},i}
=
f(
\text{noise}_i,
\text{recurrence}_i,
\text{drift}_i,
\text{distinctiveness}_i
)
\]

or directly model

\[
\Delta_i(L)
=
f(
L,
\text{noise}_i,
\text{recurrence}_i,
\text{drift}_i,
\text{distinctiveness}_i
).
\]

Candidate approaches:

- linear models;
- mixed-effects models;
- Bayesian hierarchical regression;
- monotonic GAMs;
- survival-style analysis for right-censored \(L_{\mathrm{eff}}\).

A particularly interesting framing is:

> **How much context is required before an entity reaches near-asymptotic forecasting performance?**

Right-censoring naturally fits this question.

---

# 13. Research Question 3: Does entity information emerge internally?

If technically feasible, extract frozen representations from Chronos-2 and TimesFM 3.

For context length \(L\), obtain a representation

\[
h_{i,L}.
\]

Then train lightweight probes using development data only.

## Probe A: Entity identification

Predict entity ID from \(h_{i,L}\).

Measure entity classification or retrieval performance as a function of context.

Expected pattern:

\[
A_{\mathrm{ID}}(L)\uparrow.
\]

## Probe B: Stable entity attributes

Predict:

- long-term mean;
- variance;
- dominant period;
- amplitude;
- spectral characteristics.

## Probe C: Context swapping

Take recent context from entity \(i\) and prepend historical context from entity \(j\).

Measure whether the representation shifts toward the donor entity.

---

# 14. Representation–performance coupling

Compare saturation of entity information with saturation of forecasting performance.

If

\[
L_{\mathrm{ID}}
\approx
L_{\mathrm{eff}},
\]

this supports the hypothesis that long context improves forecasts because the model increasingly identifies the entity.

An ideal figure would show:

- entity decodability increasing with context;
- forecasting error decreasing with context;
- both saturating at similar scales.

---

# 15. Black-box alternative to representation probing

If internal representations are difficult to extract reliably, use temporal context interventions.

Partition history into blocks

\[
B_1,\ldots,B_K.
\]

Measure the effect of removing each block:

\[
I_k
=
\left|
\hat y(C)
-
\hat y(C\setminus B_k)
\right|.
\]

This yields an empirical temporal influence profile.

Questions:

- How far back does each model actually use context?
- Are recurrent lags especially influential?
- Do Chronos-2 and TimesFM 3 use historical context differently?
- Does the empirical receptive field align with \(L_{\mathrm{eff}}\)?

---

# 16. Research Question 4: Can theory predict effective context?

Real datasets are observational and confounded.

Build a controlled hierarchical synthetic benchmark.

A simple generator is

\[
y_{i,t}
=
\mu_i
+
A_i
\sin\left(
\frac{2\pi t}{P_i}+\phi_i
\right)
+
x_{i,t}
+
\epsilon_{i,t}.
\]

Control:

- \(\mu_i\): entity baseline;
- \(A_i\): entity-specific amplitude;
- \(P_i\): period;
- \(\phi_i\): phase;
- \(\sigma^2\): observation noise;
- \(q\): entity drift;
- optional AR dynamics.

---

# 17. Synthetic factors to vary

Systematically vary:

## Between-entity heterogeneity

\[
\tau_\mu^2,\quad
\tau_A^2,\quad
\tau_P^2.
\]

## Within-entity noise

\[
\sigma^2.
\]

## Drift

\[
q.
\]

## Periodicity strength

Relative seasonal amplitude.

## Characteristic period

Short versus long recurrence scale.

## Entity dimensionality

How many latent parameters differ between entities.

---

# 18. Synthetic personalization phase diagram

Map effective context over controlled axes.

Example:

\[
x
=
\frac{
\text{within-entity uncertainty}
}{
\text{between-entity heterogeneity}
},
\]

\[
y
=
\text{drift},
\]

with color representing

\[
L_{\mathrm{eff}}.
\]

Predictions:

- high heterogeneity + low noise → rapid personalization;
- low heterogeneity + high noise → slow or negligible personalization;
- stable strong recurrence → longer history remains useful;
- high drift → old history eventually becomes harmful;
- higher-dimensional entity structure → more evidence required.

The synthetic benchmark should test theory rather than merely illustrate it.

---

# 19. Research Question 5: When does in-context personalization beat explicit adaptation?

After understanding the frozen-model mechanism, return to the original personalization-crossover idea.

Compare methods under the **same amount of available entity-specific data**:

1. short-context zero-shot TSFM;
2. long-context frozen TSFM;
3. parameter-efficient adaptation;
4. lightweight local model;
5. optionally Bayesian/state-space local adaptation.

---

# 20. Personalization crossover

Let

\[
n_i
\]

be available same-entity history, and let

\[
\rho_i
\]

represent compatibility between the target entity and the pretrained representation.

Expected qualitative regimes:

\[
\boxed{
\text{global zero-shot}
\rightarrow
\text{in-context personalization}
\rightarrow
\text{explicit adaptation}
}
\]

with a possible additional local-model regime when pretrained features are strongly mismatched.

Potential phase-diagram axes:

- amount of same-entity history;
- representation mismatch;
- noise;
- drift;
- entity distinctiveness.

---

# 21. Statistical analysis

Preserve the hierarchical structure of all datasets.

The unit of inference should generally be the **entity**, not individual forecast timestamps.

Recommended analyses:

- paired entity-level bootstrap;
- paired sign-flip or permutation tests;
- confidence intervals on paired MAE differences;
- entity-first cohort aggregation;
- mixed-effects models for repeated context lengths;
- correction for multiple intervention comparisons where appropriate.

For HUPA-UCM, participant-first aggregation remains essential because evaluation origins within a participant are correlated.

---

# 22. Held-out evaluation discipline

Held-out test sets remain untouched until hypotheses and choices are frozen.

Current development-selected \(L_{1\%}\):

| Dataset | Model | Frozen candidate context |
|---|---|---:|
| Electricity | Chronos-2 | 8 d |
| Electricity | TimesFM 3 | 10 d |
| PEMS-BAY | Chronos-2 | 10 d |
| PEMS-BAY | TimesFM 3 | 21 d |
| HUPA-UCM | Chronos-2 | 4 d |
| HUPA-UCM | TimesFM 3 | 7 d |

Held-out evaluation should **not** repeat the entire context-grid search.

Instead compare:

\[
L_{\mathrm{reference}}
\]

versus

\[
L_{\mathrm{selected}}.
\]

Mechanistic interventions and statistical tests should also be frozen from development data before held-out evaluation.

---

# 23. Proposed paper contributions

## Contribution 1 — Phenomenon

Show that frozen TSFMs gain substantially from accumulating same-entity context, with model- and domain-dependent effective context scales.

## Contribution 2 — Mechanism

Show that long-context gains contain an entity-specific component that cannot be explained solely by generic recurrence.

Use:

- same-entity versus other-entity history;
- shuffled history;
- stale history;
- frequency / phase interventions.

## Contribution 3 — Theory

Introduce a latent-entity framework in which useful context is governed by the tradeoff among:

- entity information;
- within-entity noise;
- temporal redundancy;
- staleness / drift.

Validate the predictions on controlled synthetic data.

## Contribution 4 — Personalization regimes

Characterize when:

- zero-shot forecasting is sufficient;
- in-context personalization is sufficient;
- explicit adaptation becomes preferable.

---

# 24. Candidate headline claim

> **Historical context in frozen time-series foundation models performs two functions: state estimation and latent entity identification. The amount of context required for effective personalization is governed by entity distinguishability, predictability, recurrence, and drift; beyond this regime, additional context saturates or becomes stale, and explicit adaptation may become preferable.**

---

# 25. Proposed figure set

## Figure 1 — Conceptual framework

\[
\text{history}
\rightarrow
\{
\text{state inference},
\text{entity inference}
\}
\rightarrow
\text{forecast}.
\]

## Figure 2 — Existing context curves

Six development curves:

- 3 datasets;
- 2 models;
- reference context;
- \(L_{1\%}\);
- right-censoring.

## Figure 3 — Context-counterfactual decomposition

Compare:

- short context;
- same-entity long context;
- matched other-entity context;
- shuffled context;
- stale context.

Show how much gain is entity-specific.

## Figure 4 — Entity-level effective context

Show the distribution of \(L_{\mathrm{eff},i}\) and associations with:

- noise;
- recurrence;
- drift;
- distinctiveness.

## Figure 5 — Synthetic phase diagram

Show effective context as controlled noise, heterogeneity, recurrence, and drift vary.

## Figure 6 — Representation / influence analysis

Either:

- entity-decoding accuracy versus context;
- representation geometry;
- temporal context influence maps.

## Figure 7 — Personalization crossover

Compare:

- zero-shot;
- frozen long-context;
- PEFT;
- local model.

Show the winning regime over personal-history size and mismatch.

---

# 26. Priority order

## Phase 1 — Establish that the effect is personalization

**Highest priority.**

Implement context counterfactuals:

1. same entity;
2. matched other entity;
3. shuffled same entity;
4. stale same entity.

Use development splits only.

### Go/no-go criterion

If same-entity history substantially outperforms matched other-entity history, the implicit-personalization hypothesis receives direct support.

If not, reframe more conservatively around temporal information acquisition rather than entity personalization.

---

## Phase 2 — Explain heterogeneity across entities

Calculate entity-level context gains and candidate predictors.

Main question:

> Why does one entity require more historical evidence than another?

Develop the information/noise/drift theory here.

---

## Phase 3 — Synthetic theory validation

Build the hierarchical benchmark and test whether theoretical predictions hold when:

- noise;
- heterogeneity;
- recurrence;
- drift

are experimentally controlled.

---

## Phase 4 — Mechanistic model analysis

Attempt representation probes.

If internal-state access is unreliable, use black-box context occlusion/intervention.

---

## Phase 5 — Explicit adaptation crossover

Only after the frozen-model phenomenon is understood, compare in-context personalization with explicit adaptation.

---

## Phase 6 — Freeze and run held-out validation

Freeze:

- hypotheses;
- selected context lengths;
- intervention definitions;
- statistical tests.

Then open held-out splits once.

---

# 27. Immediate next implementation target

Create a generic context-intervention runner.

Suggested interface:

```bash
python -m src.experiments.run_context_intervention \
  --dataset pems_bay \
  --model chronos2 \
  --context-hours 336 \
  --recent-hours 24 \
  --intervention same_entity
```

Initial interventions:

```text
same_entity
matched_other_entity
shuffled_old
stale_old
```

Later extensions:

```text
phase_scrambled
period_aligned_sparse
```

Standardized output:

```text
results/<dataset>/interventions/<model>/
```

with:

```text
case_metrics.csv
entity_metrics.csv
summary.csv
predictions.parquet
```

---

# 28. What not to do yet

Avoid:

- adding many more datasets;
- training arbitrary neural architectures;
- tuning on held-out test sets;
- fine-tuning TSFMs before understanding the frozen-model phenomenon;
- large numbers of loosely motivated ablations;
- treating the best observed context as a universal optimum;
- making causal claims from correlation-only recurrence analyses.

The next phase should prioritize **mechanistic identification over breadth**.

---

# 29. Desired end-state

The strongest version of PersTSFM is not:

> Longer context improves forecasting.

It is:

> **Frozen time-series foundation models implicitly personalize to unseen entities by using historical context to infer latent entity-specific structure. The required evidence depends systematically on entity heterogeneity, noise, recurrence, and drift, producing distinct regimes in which zero-shot forecasting, in-context personalization, or explicit adaptation is optimal.**

That is the central research programme for the next phase of PersTSFM.
