# Research Concept: When Should a Time-Series Foundation Model Personalize?

**Working subtitle:** Theory and methods for the population-to-individual transition in deep sequential models  
**Status:** Research concept / project brief v0.1  
**Literature scan current to:** 6 October 2026  
**Target:** ICML / NeurIPS main track (next realistic submission cycle)

---

## 1. Executive summary

This project asks a deceptively simple question:

> **When a pretrained/global time-series model encounters a new individual, how much should it trust population knowledge, and when should it begin trusting that individual's own data instead?**

Modern time-series foundation models (TSFMs) can make strong zero-shot forecasts across many domains. At the same time, real sequential systems are heterogeneous: patients, households, sensors, users and machines each have persistent individual dynamics. A global model is therefore attractive at cold start, when little entity-specific information is available, but it may eventually become suboptimal as observations from the same entity accumulate.

The proposed project studies the **population-to-individual transition** as a first-class machine-learning problem.

The intended paper is *not*:

- “we fine-tuned a time-series foundation model,”
- “we added LoRA to a forecasting Transformer,”
- “personalized forecasting improves eczema prediction,” or
- “local models can outperform global models.”

All of those ideas already have substantial prior art.

The stronger goal is to contribute a combination of:

1. **A formal problem definition:** sequential personalization for an unseen entity as its own evidence accumulates.
2. **Theory:** characterize when global, partially personalized and strongly personalized prediction should be optimal as a function of individual sample size, entity heterogeneity, observation noise, representation mismatch and possibly temporal dependence.
3. **A method:** an uncertainty-aware lightweight personalization mechanism, ideally derived from the theory rather than bolted onto a foundation model.
4. **A strict evaluation protocol:** held-out entities, chronological adaptation, no future leakage, multiple domains and multiple TSFM backbones.
5. **A general empirical phenomenon:** identify and measure the “personalization crossover” and negative-transfer regimes in modern deep sequence models.

A useful mental picture is:

```text
same-entity evidence →
0       2       4       8       16       32       64 observations/windows
|-------|-------|-------|-------|--------|--------|

GLOBAL / POPULATION                            INDIVIDUAL / PERSONAL
<---------------- adaptive transition ----------------------------->

cold start:
borrow heavily from population

later:
use increasing evidence from this entity

if the entity is unusual:
personalize earlier

if observations are noisy:
borrow from the population for longer
```

The project is deliberately positioned at the intersection of **deep learning, time-series foundation models, Bayesian/hierarchical modelling, transfer learning and online adaptation**.

---

# 2. Core research question

Assume there is a population of related entities

\[
i = 1,\ldots,N,
\]

with each entity producing a sequence

\[
D_i = \{(x_{i,t},y_{i,t},t)\}_{t=1}^{T_i}.
\]

A global model has learned parameters \(\theta\) from a large collection of sequences:

\[
f_\theta.
\]

A previously unseen entity \(i^\star\) arrives. At first, perhaps only one or two observations are available. As time passes, the model obtains

\[
D_{i^\star}^{(1)}, D_{i^\star}^{(2)},\ldots,D_{i^\star}^{(n)}.
\]

The central question is:

> **How should prediction transition from a population model to an individual-specific model as \(n\) increases?**

More specifically:

- Is there a predictable sample-size threshold at which personalization becomes beneficial?
- How does this threshold depend on population heterogeneity?
- How does it depend on observational noise?
- How does it depend on how atypical the new entity is relative to pretraining?
- How does temporal autocorrelation change the effective amount of evidence?
- Can uncertainty tell us **how strongly** to personalize?
- Can we predict when fine-tuning will cause **negative transfer**?
- Do the same laws appear in deep Transformers / TSFMs as in tractable probabilistic models?

---

# 3. The hypothesis

The project starts from the following hypothesis.

For an unseen entity, there are at least three regimes:

### Regime A — Population-dominated / cold start

There is too little same-entity evidence to estimate individual dynamics reliably.

\[
n \ll n^\star.
\]

A global model should dominate.

---

### Regime B — Partial personalization

There is enough information to identify some systematic individual deviation, but not enough to discard the population prior.

\[
n \approx n^\star.
\]

A shrinkage / Bayesian / gated solution should dominate both extremes.

---

### Regime C — Individual-dominated

There is enough reliable same-entity evidence that blindly retaining population assumptions introduces bias.

\[
n \gg n^\star.
\]

A strongly personalized model should dominate.

The key scientific object is therefore not merely prediction error. It is the **transition surface**

\[
n^\star = f(
\sigma^2,
\tau^2,
\rho,
r,
d,
N,
\text{temporal dependence},
\ldots
),
\]

where, for example:

- \(\sigma^2\): observation/noise level;
- \(\tau^2\): inter-entity heterogeneity;
- \(\rho\): similarity/representation overlap between the entity and population;
- \(r\): dimensionality or rank of personalization;
- \(d\): representation dimension;
- \(N\): amount of population/pretraining information.

A compelling paper would show that this transition can be **predicted**, not just observed after the fact.

---

# 4. Why this is interesting for ICML / NeurIPS

The project becomes much stronger if framed as a general learning problem rather than a medical forecasting application.

The same structure appears in:

- a new patient entering a health-monitoring system;
- a new household joining an electricity forecasting platform;
- a new road sensor being deployed;
- a new machine being monitored;
- a new user entering a recommender system;
- a new school learner interacting with an educational model;
- a new financial asset or account;
- a new wearable/device.

The recurring tension is:

> **How much should a learner borrow from the population versus specialize to an entity?**

This links several mainstream ML topics:

- foundation-model adaptation;
- representation transfer;
- parameter-efficient fine-tuning;
- meta-learning;
- hierarchical Bayes / empirical Bayes;
- online learning;
- test-time adaptation;
- personalization;
- uncertainty calibration;
- negative transfer;
- domain shift.

The target paper should therefore be written so that *patient* is only one example of an **entity**.

---

# 5. What has already been done?

## 5.1 Time-series foundation models are already mature

The project cannot claim that pretrained deep models for time series are novel.

Representative work includes:

### TimesFM

Das et al., **“A decoder-only foundation model for time-series forecasting”**, ICML 2024.

TimesFM showed that large pretrained decoder-style models can perform zero-shot forecasting across datasets.

Reference:  
https://proceedings.mlr.press/v235/das24c.html

### MOMENT

Goswami et al., **“MOMENT: A Family of Open Time-series Foundation Models”**, ICML 2024.

MOMENT targets a broader range of time-series tasks and provides openly available pretrained models.

Reference:  
https://proceedings.mlr.press/v235/goswami24a.html

### Recent TSFMs

By 2025–2026, the ecosystem includes families such as:

- Chronos / Chronos-2;
- TimesFM / TimesFM 2.x / TimesFM 3;
- Moirai / Moirai 2;
- Toto / Toto 2;
- specialized medical TSFMs such as MIRA.

Therefore the contribution should **use** these models as experimental objects, rather than compete by simply creating another huge pretrained forecasting model.

---

# 6. Closest prior work — and why the initial idea must be sharpened

This section is critical. Several recent papers already occupy parts of the space.

## 6.1 Lightweight online adaptation already exists

Lee et al., **“Lightweight Online Adaption for Time Series Foundation Model Forecasts”**, ICML 2025.

The paper combines a frozen/pretrained TSFM prediction with a lightweight online forecasting component and adaptively weights them using incoming feedback.

Reference:  
https://proceedings.mlr.press/v267/lee25ag.html

### Implication for us

We **cannot** claim:

> “No one has adapted a TSFM online before.”

That would be false.

The distinction we need is that our main object is **entity-level personalization**:

- a new entity has an identity;
- repeated observations belong to the same entity;
- the population acts as a prior;
- evidence accumulates specifically for this entity;
- the question is how personalization strength should evolve with entity-specific evidence;
- the transition itself is theoretically characterized.

The ICML 2025 online-adaptation work should be a direct baseline.

---

## 6.2 Fine-tuning criteria for TSFMs already exist

Na et al., **“Time-PEFT: Temporal and Multichannel Complexity-Based Fine-Tuning for Time-Series Foundation Models”**, ICML 2026.

This work analyzes when datasets benefit from fine-tuning using temporal and multichannel complexity measures and proposes specialized adapters.

Reference:  
https://proceedings.mlr.press/v306/na26b.html

### Implication for us

We **cannot** frame the project simply as:

> “When should a TSFM be fine-tuned?”

Time-PEFT already asks a version of this at the **dataset/task level**.

Our sharper question is:

> **For a specific unseen entity, how should personalization evolve as the amount of same-entity evidence changes?**

That is a different axis:

```text
Time-PEFT:
dataset characteristics → should this dataset/model be fine-tuned?

Proposed project:
entity evidence over time → how much should this individual entity deviate from the population model?
```

---

## 6.3 In-context adaptation already exists

Faw et al., **“In-Context Fine-Tuning for Time-Series Foundation Models”**, ICML 2025.

The method adapts forecasts using related time-series examples supplied in context, in some settings approaching explicit fine-tuning.

Reference:  
https://proceedings.mlr.press/v267/faw25b.html

### Implication

Retrieval/in-context adaptation should be treated as an important comparison.

However, the proposed project focuses on **the same entity accumulating its own evidence**, rather than merely providing examples from related series.

---

## 6.4 Personalized federated TSFM adaptation has appeared

A 2026 preprint, **“Personalized Federated Sparse Adaptation of Time-Series Foundation Models”**, studies personalized sparse adaptation of TSFMs for building-energy forecasting.

arXiv:  
https://arxiv.org/abs/2608.04695

### Implication

“Personalizing a TSFM” itself is **not new**.

Our paper needs a deeper contribution than personalized adapters:

- a formal population-to-individual problem;
- theory about the amount of same-entity evidence;
- an adaptive mechanism tied to that theory;
- a cross-domain held-out-entity study.

---

## 6.5 A very recent paper is especially important: RACE

Shi et al., **“RACE: Residual-Aware Test-Time Adaptation for Neighbor-Rich Time-Series Foundation Model Forecasting”**, arXiv:2610.00405, submitted 30 September 2026.

Reference:  
https://arxiv.org/abs/2610.00405

This is only days old at the time of this research brief and should be read carefully before committing to the final method.

RACE uses neighboring/retrieved historical series to estimate a residual correction for a frozen TSFM and includes a gate that determines when that correction should be trusted. It is evaluated in domains including continuous glucose monitoring and web/cloud forecasting.

### Why this is close

It contains:

- a frozen TSFM;
- test-time adaptation;
- a correction term;
- a gate deciding whether correction is useful;
- health time series among the experiments.

### Why there may still be a gap

The proposed project is intended to differ along several axes:

**RACE**
- primarily neighbor/retrieval-based evidence;
- predicts residual correction from related examples;
- test-time correction of a pretrained forecaster.

**Proposed work**
- same-entity evidence accumulates sequentially;
- explicit population prior → entity posterior;
- studies sample size and heterogeneity as fundamental variables;
- seeks a theoretical phase diagram / risk characterization;
- asks when and how strongly the entity should personalize;
- aims to quantify negative transfer and crossover points.

Because RACE is so recent, it should be one of the first papers reproduced or at least carefully audited.

---

# 7. Relevant theory already exists elsewhere

The theoretical novelty also needs careful positioning.

## 7.1 Transfer-learning phase diagrams

Tahir, Ganguli & Rotskoff, **“Features are fate: a theory of transfer learning in high-dimensional regression”**, ICML 2025.

Reference:  
https://proceedings.mlr.press/v267/tahir25a.html

This work develops a theory of transfer in a high-dimensional setting and studies phase behavior as target sample size and source/target feature overlap vary.

### Why this is useful

This is almost a methodological template for our desired style of paper:

1. construct a tractable theoretical setting;
2. identify regimes;
3. derive a phase diagram;
4. connect it to deep-learning behavior.

Our problem would replace generic source→target transfer with a specifically **sequential population→entity transition**.

---

## 7.2 Adaptive personalization exists in federated learning

Fermanian, Le Bars & Bellet, **“Adaptive Personalized Federated Learning via Multi-task Averaging of Kernel Mean Embeddings”**, ICML 2026.

Reference:  
https://proceedings.mlr.press/v306/fermanian26a.html

This work learns adaptive global/local weighting and provides finite-sample excess-risk guarantees.

Other relevant personalized federated learning theory includes:

- Yu et al., **“Understanding the Statistical Accuracy-Communication Trade-off in Personalized Federated Learning with Minimax Guarantees”**, ICML 2025:  
  https://proceedings.mlr.press/v267/yu25h.html

- Thapa & Li, **“Harnessing Heterogeneous Statistical Strength for Personalized Federated Learning via Hierarchical Bayesian Inference”**, ICML 2025:  
  https://proceedings.mlr.press/v267/thapa25a.html

### Implication

We **cannot** claim to invent the concept of transitioning from global to personalized learning.

The gap must be more specific:

- sequential temporal data;
- online accumulation of observations from one held-out entity;
- TSFM representations;
- temporal dependence;
- forecasting;
- a personalization crossover law;
- deep-model validation.

---

# 8. Older global-versus-local forecasting literature also matters

There is established forecasting literature comparing one **global model** trained across many series versus **local models** trained independently.

For example:

Hewamalage, Bergmeir & Bandara, **“Global models for time series forecasting: A Simulation study”**, *Pattern Recognition*, 2022.

DOI:  
https://doi.org/10.1016/j.patcog.2021.108441

This literature already shows that global models can exploit information across related series and that the relative advantage of global versus local models depends on dataset structure.

### Consequence

The paper must not make a historical claim such as:

> “We are the first to ask whether global or local forecasting is better.”

Instead, we should say:

> We study **adaptive, entity-specific movement between the two regimes in modern pretrained deep sequence models**, and aim to derive when that transition should occur.

---

# 9. The gap after the literature scan

A defensible current gap is a **conjunction of properties**, rather than one completely untouched keyword.

As of 6 October 2026, I have not found a paper that jointly does all of the following:

1. **Defines forecasting for a new entity as a sequential population-to-individual transition.**
2. **Varies the amount of same-entity evidence explicitly and treats it as a central independent variable.**
3. **Derives a theory predicting optimal personalization strength or a crossover regime from quantities such as sample size, heterogeneity, noise and representation mismatch.**
4. **Uses that theory to design an uncertainty-aware lightweight personalization rule for modern deep TSFMs.**
5. **Evaluates under strict held-out-entity chronological protocols across multiple unrelated domains.**
6. **Measures negative transfer, adaptation speed and the complete personalization curve rather than only final forecasting accuracy.**

This is the research gap to test.

It is deliberately stated cautiously. The project should *not* proceed under the assumption that novelty has already been proven; the first weeks should continue to search for adjacent work.

---

# 10. Proposed formal model

A useful bridge between tractable theory and a large frozen TSFM is:

\[
h_{i,t} = \Phi_\theta(x_{i,1:t}),
\]

where \(\Phi_\theta\) is a pretrained sequence representation.

Suppose the global prediction head is

\[
\hat y^{\text{global}}_{i,t+1}
=
h_{i,t}^{\top} w_0.
\]

Allow an entity-specific deviation in a low-dimensional subspace:

\[
w_i = w_0 + U a_i,
\]

with

\[
a_i \sim \mathcal{N}(0,\tau^2 I_r).
\]

Then

\[
y_{i,t+1}
=
h_{i,t}^{\top}(w_0 + U a_i)
+
\epsilon_{i,t},
\]

where

\[
\epsilon_{i,t}\sim\mathcal{N}(0,\sigma^2).
\]

Interpretation:

- \(w_0\): population knowledge;
- \(a_i\): individual deviation;
- \(U\): low-rank personalization directions;
- \(\tau^2\): population heterogeneity;
- \(\sigma^2\): observation uncertainty;
- \(r\): personalization complexity.

This model is already rich enough to ask:

> How many observations are required before estimating \(a_i\) is better than setting \(a_i=0\)?

---

# 11. The simplest intuition: adaptive shrinkage

In a highly simplified isotropic Gaussian setting, the posterior weight given to an individual effect often takes a form qualitatively like

\[
\gamma_n
\approx
\frac{n\tau^2}
{n\tau^2+\sigma^2}.
\]

This is **not yet a claimed result for the full project**; it is the intuition we should formalize.

It implies:

### With little individual data

\[
n \rightarrow 0
\quad\Rightarrow\quad
\gamma_n \rightarrow 0.
\]

Stay close to the population model.

### As individual evidence grows

\[
n \rightarrow \infty
\quad\Rightarrow\quad
\gamma_n \rightarrow 1.
\]

Trust the entity-specific estimate.

A crude midpoint occurs when

\[
n^\star
\sim
\frac{\sigma^2}{\tau^2}.
\]

This immediately generates interpretable hypotheses.

### Higher noise

\[
\sigma^2 \uparrow
\quad\Rightarrow\quad
n^\star \uparrow.
\]

You need more observations before personalizing.

### Greater population heterogeneity

\[
\tau^2 \uparrow
\quad\Rightarrow\quad
n^\star \downarrow.
\]

Individual differences are larger, so personalization becomes worthwhile sooner.

This simple relationship is exactly the kind of phenomenon we should seek to generalize and test in deep models.

---

# 12. Possible theoretical progression

The theory should develop in stages rather than attempting the hardest model immediately.

## Theory A — Gaussian hierarchical regression

Start with a closed-form model.

Questions:

- posterior personalization weight;
- Bayes risk;
- global-versus-local risk;
- exact/approximate crossover point;
- effect of feature dimension;
- uncertainty calibration.

Goal: obtain the first clean theorem and phase diagram.

---

## Theory B — Low-rank personalized representation

Use

\[
w_i=w_0+Ua_i
\]

with rank \(r\ll d\).

Study:

- sample complexity as a function of \(r\);
- bias-variance tradeoff;
- when a low-rank adapter is preferable to full fine-tuning;
- negative transfer when the true entity deviation falls outside the adapter subspace.

This gives a direct theoretical analogue of LoRA/adapters.

---

## Theory C — Representation mismatch

Let \(\rho\) describe the alignment between the frozen TSFM representation and entity-specific predictive directions.

Study the interaction:

\[
(n,\tau^2,\sigma^2,\rho,r).
\]

A possible phase diagram could identify:

```text
                         entity observations n
                  few                         many
              ┌──────────────────────┬──────────────────────┐
 high         │ global / shrinkage   │ partial personalize  │
 overlap      │                      │                      │
              ├──────────────────────┼──────────────────────┤
 low          │ uncertain / retrieval│ strong adaptation /  │
 overlap      │ or abstain           │ representation update│
              └──────────────────────┴──────────────────────┘
```

This would connect naturally to modern transfer-learning theory.

---

## Theory D — Temporal dependence

The observations are not IID.

One tractable extension is to assume a stable AR/LDS process or a mixing process.

Then the relevant quantity may become an **effective sample size**

\[
n_{\mathrm{eff}} < n,
\]

rather than raw observations.

This produces another useful hypothesis:

> 32 highly autocorrelated measurements may contain substantially less personalization evidence than 32 diverse measurements.

This is genuinely important for health and sensors.

---

# 13. Proposed deep-learning method

The method should ideally fall out of the theory.

A good first candidate is:

## Hierarchical Bayesian Personalization Adapter

Use a frozen or mostly frozen TSFM backbone:

\[
h_t = \Phi_\theta(x_{1:t}).
\]

Add a lightweight individual adapter:

\[
g_{\phi_i}(h_t).
\]

Prediction:

\[
\hat y_{i,t+1}
=
f_\theta(h_t)
+
\alpha_{i,t} g_{\phi_i}(h_t).
\]

The personal parameters have a population prior:

\[
\phi_i \sim p(\phi \mid \eta_{\mathrm{pop}}).
\]

As observations arrive, update

\[
p(\phi_i \mid D_{i,1:t}).
\]

The personalization strength

\[
\alpha_{i,t}
\]

should be determined from posterior uncertainty / evidence rather than manually scheduled.

---

# 14. Three implementation versions

## V0 — Frozen backbone + Bayesian residual head

This should be built first.

- Freeze TSFM.
- Extract hidden features.
- Fit a very small per-entity residual model.
- Learn a hierarchical prior from development entities.
- Update the residual posterior online.
- Combine global and personal predictions analytically.

Advantages:

- fast;
- mathematically tractable;
- strongly connected to theory;
- easy to test across multiple TSFMs;
- avoids huge compute;
- gives calibrated uncertainty.

This version alone could reveal whether the research hypothesis is real.

---

## V1 — Prior-centred LoRA / adapter

If V0 works, replace the simple head with a parameter-efficient adapter.

Instead of standard unconstrained PEFT, regularize the adapter toward the population state:

\[
\phi_i \sim \mathcal{N}(0,\tau^2I).
\]

Update only the adapter online.

Question:

> Does theory derived for a low-rank personal subspace predict when LoRA-style adaptation becomes beneficial?

---

## V2 — Uncertainty-aware mixture

Maintain:

\[
\hat y^{G}_{i,t}
\]

from the global model and

\[
\hat y^{P}_{i,t}
\]

from the personal component.

Combine them:

\[
\hat y_{i,t}
=
(1-\alpha_{i,t})\hat y^{G}_{i,t}
+
\alpha_{i,t}\hat y^{P}_{i,t}.
\]

Rather than learn \(\alpha\) as a black-box gate alone, derive or constrain it using:

- posterior variance;
- estimated entity heterogeneity;
- residual evidence;
- representation distance;
- number/effective number of observations.

This is where the method could become scientifically distinctive.

---

# 15. Synthetic experiments — essential for the theory

Synthetic data should not be an afterthought.

We need a data-generating process where the ground truth is known.

A minimal hierarchical dynamic process could be:

\[
a_i \sim \mathcal{N}(0,\tau^2),
\]

\[
y_{i,t}
=
(\phi_0+a_i)y_{i,t-1}
+
\epsilon_{i,t},
\]

\[
\epsilon_{i,t}
\sim
\mathcal{N}(0,\sigma^2).
\]

Sweep:

- heterogeneity \(\tau\);
- noise \(\sigma\);
- number of entities \(N\);
- same-entity sample size \(n\);
- autocorrelation;
- amount of population training data;
- missingness;
- irregular sampling;
- distribution shift.

Later use a nonlinear generator:

\[
z_{i,t+1}=F_{\theta_0+\Delta_i}(z_{i,t})+\epsilon_t.
\]

The key output should be **phase diagrams**, not only tables.

For example:

```text
x-axis: observations from new entity
y-axis: population heterogeneity
colour/state: best strategy

GLOBAL ─── PARTIAL PERSONALIZATION ─── LOCAL/PERSONAL
```

Then compare theoretical regime boundaries with the empirical boundaries produced by deep models.

---

# 16. Real datasets

The dataset suite should intentionally include several unrelated domains.

## Dataset 1 — ElectricityLoadDiagrams20112014

UCI ElectricityLoadDiagrams20112014.

- hundreds of electricity clients;
- repeated measurements for each client;
- 15-minute resolution in the original dataset;
- naturally defines **client = entity**;
- long history per entity.

Reference:  
https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

### Why useful

This is probably the easiest development dataset for testing the population→entity adaptation protocol.

### Caveat

It is an old, widely used forecasting dataset and may have appeared in TSFM pretraining corpora. It should not be the only final benchmark.

---

## Dataset 2 — PEMS-BAY

Traffic readings from hundreds of road sensors in the San Francisco Bay Area.

Typical setup:

- roughly 325 sensors;
- 5-minute measurements;
- each sensor is a persistent entity.

### Why useful

It provides a very different entity structure:

\[
\text{person} \rightarrow \text{electricity client} \rightarrow \text{road sensor}.
\]

If the same personalization law appears across these, the contribution becomes much more general.

### Caveat

Also widely used, so contamination/pretraining concerns apply.

---

## Dataset 3 — ShanghaiT1DM / ShanghaiT2DM CGM

Zhao et al., *Scientific Data* 2023.

The datasets contain continuous glucose-monitoring trajectories and associated clinical information.

ShanghaiT2DM contains substantially more individuals than the T1 cohort and is particularly relevant for entity-level personalization.

Paper:  
https://www.nature.com/articles/s41597-023-01940-7

Data collection:  
https://doi.org/10.6084/m9.figshare.c.6310860

### Why useful

- true person-level sequential data;
- strong between-person heterogeneity;
- high-frequency signal;
- directly tests whether the phenomenon extends to health.

### Important caution

The very recent RACE preprint operates in the CGM domain, so we must avoid relying on CGM as the sole demonstration of novelty.

---

## Dataset 4 — PhysioNet/CinC Challenge 2012

Approximately 12,000 ICU records covering the first 48 hours after admission, with irregular and heavily missing physiological measurements.

Reference:  
https://physionet.org/content/challenge-2012/1.0.0/

### Why useful

This is not the ideal primary dataset for long-term personal adaptation because each patient trajectory is relatively short.

However, it is valuable as a **stress test** for:

- irregular sampling;
- missingness;
- multivariate observations;
- uncertainty.

It should therefore be secondary.

---

## Dataset 5 — private longitudinal eczema datasets

Use the existing longitudinal eczema datasets as a final external validation setting.

They are valuable because they contain:

- real individuals;
- repeated measures;
- heterogeneous trajectories;
- sparse and/or irregular observations;
- a natural population-versus-patient tension.

Crucially, they are unlikely to have been used in pretraining of public TSFMs.

### Role in the paper

They should be **one real-world application**, not the source of the methodological claim.

That keeps the paper general while exploiting a dataset where we understand the observation process very well.

---

## Dataset 6 — fresh datasets from modern benchmarks

Qiao et al., **“It’s TIME: Towards the Next Generation of Time Series Forecasting Benchmarks”**, ICML 2026.

Reference:  
https://proceedings.mlr.press/v306/qiao26g.html

TIME was explicitly created to improve evaluation of modern forecasting models, including concerns around stale benchmarks and leakage/contamination.

The final study should inspect which TIME tasks have a meaningful entity structure and include at least one suitable **fresh/post-pretraining** benchmark if possible.

This would materially strengthen an ICML/NeurIPS submission.

---

# 17. Experimental protocol

The protocol is one of the most important parts of the project.

## 17.1 Split by entity FIRST

Never randomly mix observations from the same entity across train and test.

Use:

```text
population training entities
development entities
held-out test entities
```

A test entity must never be seen during model/method selection.

---

## 17.2 Reveal each test entity chronologically

For a test entity:

```text
history available: 0
forecast

reveal observations
history available: 1
adapt
forecast

history available: 2
adapt
forecast

history available: 4
adapt
forecast

history available: 8
adapt
forecast

...

history available: 64
adapt
forecast
```

Possible checkpoints:

\[
n \in \{0,1,2,4,8,16,32,64,\ldots\}.
\]

For high-frequency data, use windows rather than single timestamps.

The important requirement is that **the amount of same-entity evidence is explicit and controlled**.

---

## 17.3 Strict causal forecasting

At time \(t\), personalization can use only:

\[
D_{i,1:t}.
\]

No later value from that entity may affect the update.

This must be enforced programmatically.

---

# 18. Baselines

A strong paper needs surprisingly strong baselines.

## Classical / local

- persistence;
- seasonal naive;
- ETS / ARIMA where appropriate;
- local autoregression;
- local ridge regression.

## Deep models trained from scratch

- global Transformer / PatchTST;
- possibly DLinear/NLinear as simple strong baselines.

## Foundation-model baselines

At least two different TSFM families.

Good candidates in 2026:

- Chronos-2;
- TimesFM 2.5 or another accessible TimesFM checkpoint;
- Moirai 2;
- Toto 2.

Avoid designing the paper around one proprietary/restricted backbone.

---

## Adaptation baselines

Must include, where reproducible:

1. Frozen zero-shot TSFM.
2. Full fine-tuning.
3. Standard LoRA.
4. Standard adapter/PEFT.
5. Pure local model.
6. In-context fine-tuning / retrieval-based adaptation.
7. ICML 2025 lightweight online adaptation.
8. Time-PEFT or its most relevant components.
9. RACE if code/resources make reproduction possible.
10. Static mixtures:
   \[
   (1-\alpha)f_{\mathrm{global}} + \alpha f_{\mathrm{local}}.
   \]
11. Oracle hindsight mixture as an upper/reference bound.

The proposed method should beat or meaningfully improve the **adaptation trade-off**, not merely one zero-shot model.

---

# 19. Metrics

Final forecasting MAE is not enough.

## Forecast quality

Depending on domain:

- MAE;
- RMSE;
- MASE;
- RMSSE;
- pinball loss / quantile loss;
- CRPS for probabilistic prediction.

---

## Calibration

- empirical interval coverage;
- interval width;
- NLL if probabilistic density is available;
- calibration error.

---

## Adaptation curve

Define

\[
L(n)
\]

as expected test loss after \(n\) same-entity observations.

Plot:

\[
n \mapsto L(n).
\]

This should be a central figure.

---

## Personalization crossover

For each entity or dataset, estimate:

\[
n^\star
=
\min\{n:
L_{\mathrm{personal}}(n)
<
L_{\mathrm{global}}(n)
\}.
\]

Study its relationship to:

- noise;
- heterogeneity;
- entity distance;
- uncertainty;
- effective sample size.

---

## Negative-transfer rate

\[
\Pr(
L_{\mathrm{personal}}(n)
>
L_{\mathrm{global}}(n)
).
\]

This may be more scientifically interesting than average improvement.

---

## Area under the adaptation curve

A method that becomes excellent after 200 observations may be much worse than one that adapts safely after 5.

Use something like

\[
\mathrm{AUC}_{\mathrm{adapt}}
=
\sum_n w_n L(n).
\]

The exact naming/definition can be finalized later.

---

## Personalization regret

An online-learning style metric:

\[
R_T
=
\sum_{t=1}^T
\ell_t(\hat y_t)
-
\min_{m\in\mathcal{M}}
\sum_{t=1}^T
\ell_t(\hat y_t^m),
\]

where the comparator could be:

- always global;
- always local;
- best static mixture;
- hindsight oracle.

This could give the project a stronger theoretical/online-learning angle.

---

## Compute

Also measure:

- update latency;
- memory;
- trainable parameters;
- number of gradient steps;
- GPU cost.

A lightweight personalization rule is much more compelling if it is also deployable.

---

# 20. Critical ablations

The final paper should isolate *why* the method works.

Ablate:

- no hierarchical prior;
- fixed prior variance;
- learned heterogeneity;
- fixed \(\alpha\);
- learned black-box \(\alpha\);
- uncertainty-derived \(\alpha\);
- full-rank personal update;
- low-rank update;
- frozen versus partially trainable backbone;
- chronological sample size;
- entity-distance signal;
- representation mismatch;
- calibration mechanism.

Also ask:

> Does the theoretically predicted \(n^\star\) correlate with the empirical deep-model crossover?

That correlation may become one of the headline results.

---

# 21. What would make the paper genuinely strong?

A weak paper would be:

> “Our adapter improves MAE by 2% on four datasets.”

A much stronger ICML/NeurIPS story would look like:

### Finding 1

There is a reproducible population→individual transition across domains and architectures.

### Finding 2

The position of that transition is systematically controlled by heterogeneity and noise.

### Theorem

A tractable model predicts the transition / optimal shrinkage rule.

### Method

Use that theory to construct an uncertainty-aware personal adapter.

### Empirical result

The method avoids negative transfer and adapts faster than generic PEFT/online baselines.

### Generalization

The predicted regime structure survives across:

- synthetic systems;
- electricity;
- traffic;
- CGM;
- another fresh dataset;
- private longitudinal health data;
- several TSFM backbones.

That is much closer to a main-track contribution.

---

# 22. What could make the project fail?

We should define kill criteria early.

## Failure A — No stable crossover

If global TSFMs remain best even after substantial same-entity data, there may be no meaningful personalization phenomenon.

That itself would be interesting, but the current proposed method would need rethinking.

---

## Failure B — Standard LoRA already solves it

If generic PEFT dominates across all \(n\), with no meaningful negative-transfer regime, then a sophisticated Bayesian transition rule may add little.

---

## Failure C — Existing online adaptation solves it

If the ICML 2025 online-adaptation baseline or RACE already matches the proposed mechanism across the target protocol, novelty becomes weak.

---

## Failure D — Theory predicts synthetic systems only

If the derived transition has no relationship to deep TSFM behavior, the paper may split into two disconnected stories.

That is a major warning sign.

---

## Failure E — Gains are health-specific

If the phenomenon appears only in eczema or CGM, the work should probably target an ML-for-health venue rather than ICML/NeurIPS main track.

---

# 23. Useful pivots if the original hypothesis fails

A failed initial hypothesis does not necessarily kill the broader research direction.

Possible pivots:

### Pivot 1 — A negative-result paper

> **When personalization hurts time-series foundation models**

Systematic analysis of negative transfer could itself be valuable if broad and surprising.

### Pivot 2 — Uncertainty-calibrated online adaptation

If adaptation helps but crossover theory is weak, focus on safe adaptation with risk/coverage guarantees.

### Pivot 3 — Informative observation processes

Extend the model so the *timing of observations* itself carries information.

This is particularly relevant to health and would connect deep time-series models to non-random missingness.

### Pivot 4 — Irregular-time personalization

Study whether adaptation rules based on the number of observations fail when observation intervals differ, and replace raw \(n\) with information/effective sample size.

---

# 24. First 8-week research sprint

The initial objective is **not to write the full paper**. It is to decide whether the core phenomenon is real and sufficiently novel.

## Week 1 — Infrastructure + first benchmark

Implement a clean entity-held-out protocol on Electricity.

Run:

- zero-shot TSFM;
- local model;
- frozen TSFM + simple residual head.

Produce:

\[
L(n)
\]

for multiple same-entity sample sizes.

**Decision:** Does personalization ever overtake the global model?

---

## Week 2 — Simple personalization baselines

Add:

- fixed mixture;
- ridge residual head;
- standard LoRA/adapter if feasible.

Measure:

- crossover \(n^\star\);
- negative transfer;
- adaptation curve.

Repeat with a second TSFM backbone.

**Decision:** Is the phenomenon architecture-independent?

---

## Week 3 — Synthetic hierarchical generator

Implement controlled simulations.

Sweep:

\[
\tau,
\sigma,
n.
\]

Generate the first empirical phase diagram.

At the same time derive the simplest closed-form Bayesian result.

---

## Week 4 — Theory-versus-experiment test

Predict the crossover from the simple theory.

Test whether the empirical relationship behaves approximately like:

\[
n^\star \propto \frac{\sigma^2}{\tau^2}.
\]

Do **not** force this exact law if the experiments disagree.

The goal is to learn what variables are missing.

---

## Week 5 — Second domain

Add either:

- PEMS-BAY, or
- ShanghaiT2DM.

Repeat the exact held-out-entity protocol.

---

## Week 6 — First proposed method

Build V0:

> Frozen backbone + hierarchical Bayesian residual personalization.

Compare against:

- global zero-shot;
- local;
- LoRA;
- online adaptation baseline.

---

## Week 7 — Stress tests

Vary:

- noise;
- missingness;
- irregularity;
- entity shift;
- amount of population data.

Ask whether uncertainty predicts personalization failure.

---

## Week 8 — Go / no-go decision

Proceed to a full ICML/NeurIPS project only if at least three of these are true:

- crossover is robust;
- crossover varies predictably with heterogeneity/noise;
- standard PEFT suffers meaningful negative transfer;
- proposed shrinkage rule improves early adaptation;
- phenomenon holds in multiple domains;
- simple theory predicts at least the direction/order of deep-model behavior.

If these conditions fail, pivot early rather than spending six months polishing a weak adapter.

---

# 25. Minimum viable experiment to run FIRST

Before implementing LoRA, Mamba, neural SDEs or anything sophisticated:

## Data

Electricity clients.

## Backbone

One accessible pretrained TSFM, preferably Chronos-2 or another model with easy feature/output access.

## Test design

Hold out entities completely.

For each held-out entity, expose:

\[
n=
0,2,4,8,16,32,64
\]

history windows.

Compare:

### A. Global

Frozen pretrained/global forecast.

### B. Personal-only

A small model trained only from the entity's available history.

### C. Shrinkage

Global forecast + regularized personal residual.

Then create **one plot**:

```text
forecast error
^
|  personal-only
|  \
|   \
|    \_________
|
| ----------- global
|       \
|        \_____ shrinkage
|
+----------------------------> same-entity observations
       2  4  8 16 32 64
```

If there is no interesting structure in this plot, do not build a large architecture yet.

---

# 26. A possible theoretical paper skeleton

## 1. Introduction

Foundation models bring strong population-level priors, but real sequential entities are heterogeneous.

Key question:

> When should a model personalize?

## 2. Sequential entity personalization

Formal problem and online protocol.

## 3. Theory

Risk decomposition and personalization regimes.

Derive:

- optimal shrinkage;
- sample-size dependence;
- effect of heterogeneity/noise;
- representation mismatch;
- temporal effective sample size.

## 4. Deep personalization method

Translate the theory into a lightweight adapter/gate.

## 5. Controlled experiments

Synthetic phase diagrams.

## 6. Deep-model experiments

Multiple TSFMs and multiple domains.

## 7. Real-world longitudinal health validation

CGM + eczema / another longitudinal dataset.

## 8. Analysis

Negative transfer, calibration, compute, uncertainty, regime prediction.

---

# 27. Candidate paper titles

Do **not** settle on a title yet, but these capture the intended level of contribution.

### Theory-forward

**When Should a Foundation Model Personalize? A Theory of Population-to-Individual Adaptation for Time Series**

### More concise

**When to Personalize Time-Series Foundation Models**

### Method + theory

**From Population Priors to Individual Dynamics: Adaptive Personalization of Time-Series Foundation Models**

### Phenomenon-forward

**The Personalization Crossover in Time-Series Foundation Models**

### Slightly broader

**How Much Individual Data Is Enough? Adaptive Personalization of Deep Sequential Models**

My preference at this stage:

> **The Personalization Crossover in Time-Series Foundation Models**

because it names a potentially new empirical/theoretical object rather than merely another algorithm.

---

# 28. Novelty statement — draft

A future paper could say something approximately like:

> Existing work has studied global versus local time-series forecasting, parameter-efficient fine-tuning of time-series foundation models, online adaptation, retrieval-based test-time correction, and personalized learning in federated settings. However, comparatively little attention has been paid to the *sequential population-to-individual transition* faced by a pretrained forecaster when it encounters a previously unseen entity. We study how optimal personalization evolves as entity-specific observations accumulate, characterize its dependence on heterogeneity, noise and representation mismatch, and derive an uncertainty-aware adaptation rule that interpolates between population and individual prediction.

This statement should be revised after the full related-work audit.

---

# 29. What NOT to claim

Avoid these claims unless later searches establish them rigorously:

- “first personalized TSFM” — false/unsafe;
- “first online adaptation of TSFMs” — false;
- “first global-local forecasting analysis” — false;
- “first Bayesian personalization method” — false;
- “first method deciding whether to fine-tune TSFMs” — false;
- “first health personalization with foundation models” — highly unlikely.

The novelty should emerge from the **specific conjunction of theory + sequential entity evidence + TSFMs + adaptive shrinkage + held-out-entity evaluation**.

---

# 30. Why this direction fits a transition from Bayesian state-space modelling to deep learning

This project does not require abandoning structured probabilistic modelling.

Instead, it provides a natural progression:

```text
hand-specified Bayesian dynamics
            ↓
deep learned representations / TSFM
            +
hierarchical probabilistic personalization
            ↓
deep probabilistic sequential learning
```

The foundation model handles complex representation learning.

The probabilistic layer handles:

- uncertainty;
- population priors;
- partial pooling;
- individual evidence;
- online updating.

This is arguably more distinctive than replacing a state-space model with a standard Transformer and treating the deep architecture itself as the contribution.

---

# 31. Immediate reading list

Read these first, in roughly this order.

## A. Closest adaptation papers

1. **Lee et al. (ICML 2025)**  
   *Lightweight Online Adaption for Time Series Foundation Model Forecasts*  
   https://proceedings.mlr.press/v267/lee25ag.html

2. **Na et al. (ICML 2026)**  
   *Time-PEFT: Temporal and Multichannel Complexity-Based Fine-Tuning for Time-Series Foundation Models*  
   https://proceedings.mlr.press/v306/na26b.html

3. **Faw et al. (ICML 2025)**  
   *In-Context Fine-Tuning for Time-Series Foundation Models*  
   https://proceedings.mlr.press/v267/faw25b.html

4. **Shi et al. (arXiv, 30 Sep 2026)**  
   *RACE: Residual-Aware Test-Time Adaptation for Neighbor-Rich Time-Series Foundation Model Forecasting*  
   https://arxiv.org/abs/2610.00405

---

## B. Theory templates

5. **Tahir et al. (ICML 2025)**  
   *Features are fate: a theory of transfer learning in high-dimensional regression*  
   https://proceedings.mlr.press/v267/tahir25a.html

6. **Fermanian et al. (ICML 2026)**  
   *Adaptive Personalized Federated Learning via Multi-task Averaging of Kernel Mean Embeddings*  
   https://proceedings.mlr.press/v306/fermanian26a.html

7. **Thapa & Li (ICML 2025)**  
   *Harnessing Heterogeneous Statistical Strength for Personalized Federated Learning via Hierarchical Bayesian Inference*  
   https://proceedings.mlr.press/v267/thapa25a.html

---

## C. Foundation-model context

8. **Das et al. (ICML 2024)**  
   *A decoder-only foundation model for time-series forecasting*  
   https://proceedings.mlr.press/v235/das24c.html

9. **Goswami et al. (ICML 2024)**  
   *MOMENT: A Family of Open Time-series Foundation Models*  
   https://proceedings.mlr.press/v235/goswami24a.html

10. **Qiao et al. (ICML 2026)**  
    *It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks*  
    https://proceedings.mlr.press/v306/qiao26g.html

---

# 32. Final assessment

## Is the idea worth testing?

**Yes.**

But the initial broad idea—

> “personalize a deep time-series model as more patient data become available”

—is **not sufficiently novel by itself**.

There is already work on:

- online TSFM adaptation;
- PEFT for TSFMs;
- in-context adaptation;
- personalized federated TSFMs;
- global-versus-local forecasting;
- adaptive global/local weighting in federated learning;
- very recent neighbor-based test-time TSFM correction.

The project becomes much more interesting if the scientific target is:

> **the personalization crossover itself.**

That means asking:

1. **When** does individual evidence become preferable to population knowledge?
2. **Why** does the transition occur?
3. Can the transition be predicted from measurable quantities?
4. Can a deep model adapt at the correct rate automatically?
5. Can we prove something meaningful in a simplified model?
6. Does that theory survive contact with real foundation models?

If the answer to those questions is yes, this has the ingredients of a genuinely general ML project rather than a new application of deep learning.

---

# 33. One-sentence project brief

> **Develop a theory and deep-learning method for deciding when and how strongly a pretrained time-series foundation model should personalize to a previously unseen entity as entity-specific observations accumulate, and test whether the predicted population-to-individual transition holds across synthetic systems, public forecasting domains and longitudinal health data.**

---

# 34. First concrete action

Before building a new architecture, implement the **held-out-entity adaptation curve** on one clean dataset and one TSFM.

The first research question is not:

> “Can we beat the benchmark?”

It is:

> **“Does the personalization crossover actually exist, and is it systematic?”**

If yes, the next step is to explain it mathematically.

If no, that result should change the project immediately.

