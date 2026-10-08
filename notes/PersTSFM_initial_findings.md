# PersTSFM — Initial Experimental Findings

**Project:** Personalization of Time-Series Foundation Models  
**Current stage:** Initial validation experiments  
**Model:** Chronos-2  
**Dataset:** ElectricityLoadDiagrams20112014  
**Status:** Validation set only; held-out test set remains untouched

## 1. Research question

The original motivation of PersTSFM was to investigate:

> **When should a pretrained time-series model begin to personalize to an individual entity as entity-specific observations accumulate?**

The initial hypothesis was that a population-level model should perform best at cold start, while an increasingly personalized model should become preferable after enough individual data have been observed.

The first experiments investigated two possible personalization mechanisms:

1. explicit correction of a frozen foundation model using an entity's previous forecast errors;
2. implicit personalization through increasing amounts of same-entity input context.

The experiments produced a clear result: **explicit residual correction was generally unsuccessful, whereas increasing same-entity context substantially improved the frozen foundation model.**

---

# 2. Dataset and experimental protocol

The ElectricityLoadDiagrams20112014 dataset contains:

- 370 electricity customers;
- 140,256 timestamps;
- 15-minute resolution;
- no missing numeric observations.

Customers enter the dataset at different dates, producing long structural leading-zero periods. Each series was therefore considered active from its first non-zero observation onward.

Entities were required to have at least **365 active days**, leaving:

\[
351/370 = 94.9\%
\]

of customers.

These were permanently split at the entity level into:

- **245 population-training entities**
- **53 validation entities**
- **53 held-out test entities**

The split was stratified by year of first activity.

The held-out test set has not been used in any experiment so far.

For validation, each entity contributes 12 forecast origins separated by seven days:

\[
53\times12 = 636
\]

forecast cases.

Each forecast predicts the next:

\[
96\text{ timestamps}=24\text{ hours}.
\]

---

# 3. Frozen Chronos-2 baseline

Chronos-2 was first evaluated with four days of same-entity context:

\[
384\text{ observations}
\rightarrow
96\text{-step forecast}.
\]

Across 636 validation forecasts:

| Metric | Result |
|---|---:|
| Mean MAE | 22.3207 |
| Median MAE | 9.0912 |
| Mean RMSE | 32.8892 |
| 80% interval coverage | 77.54% |
| Negative prediction fraction | 0.41% |

Quantile crossing was extremely rare:

\[
16/61,056 = 0.0262\%.
\]

The large difference between mean and median MAE indicates substantial heterogeneity across customers.

---

# 4. Explicit residual personalization did not work

A residual bank was generated containing Chronos-2 forecasts for the 64 days preceding each validation forecast.

This produced:

- 7,473 historical forecast days;
- 717,408 individual residual observations.

Residuals were defined as:

\[
r_{i,t}
=
y_{i,t}
-
\hat y^{G}_{i,t}.
\]

The first personalization model estimated one scalar customer-specific correction from the preceding \(n\) days:

\[
b_i^{(n)}
=
\frac{1}{96n}
\sum r_{i,t},
\]

with

\[
n\in\{1,2,4,8,16,32,64\}.
\]

The personalized forecast was

\[
\hat y^{P}
=
\hat y^{G}
+
b_i^{(n)}.
\]

This did **not** improve average forecasting performance.

For example:

| Personal days | Mean MAE | Δ MAE vs Chronos |
|---:|---:|---:|
| 0 | 22.3207 | 0 |
| 1 | 25.9398 | +3.6190 |
| 8 | 22.8958 | +0.5751 |
| 16 | 22.6767 | +0.3559 |
| 64 | 22.5359 | +0.2152 |

More personal history reduced the damage caused by the correction, but never produced an average crossover below the frozen Chronos baseline.

Only 24/53 entities improved at any tested history length.

---

# 5. More expressive residual profiles also failed

A second approach estimated an entire 96-dimensional personal residual profile:

\[
\mathbf b_i
=
[b_{i,1},\ldots,b_{i,96}].
\]

This allowed each customer to have different systematic corrections at different forecast horizons/times of day.

However, it performed even worse at small sample sizes because of increased variance.

For example:

\[
\Delta MAE_{n=1}=+8.6310
\]

and even with 64 previous days:

\[
\Delta MAE_{n=64}=+0.2847.
\]

The more expressive personal model therefore displayed a clear bias–variance problem: it strongly overfit when little individual evidence was available and converged toward the global model as more observations accumulated.

---

# 6. Historical residuals are not reliably persistent

Residuals from individual previous days were then tested directly.

Neither recent nor weekly historical residuals provided useful corrections.

The best scalar residual lag among 1–28 days was lag 13:

\[
\Delta MAE = +2.2530.
\]

Even the intuitively attractive weekly lag:

\[
t-7\text{ days}
\]

gave:

\[
\Delta MAE = +2.6046.
\]

Using the complete previous-day residual profile was even worse.

Therefore the failure of residual personalization is not simply caused by averaging incompatible weekdays.

---

# 7. Shrinkage confirmed that explicit scalar personalization contains little transferable signal

A shrinkage experiment considered:

\[
\hat y
=
\hat y^G
+
\alpha b_i,
\qquad
\alpha\in[0,1].
\]

Five-fold entity-level cross-validation was used to choose \(\alpha\) without evaluating it on the same entities used for selection.

The selected personalization weight was essentially:

\[
\boxed{\alpha^\star\approx0}
\]

for almost every amount of personal history.

Even an oracle allowed to choose the best \(\alpha\) separately for every entity produced only very small improvements.

This strongly suggests that a persistent additive residual is **not** the appropriate representation of personal information.

At this point, the scalar residual direction was discontinued.

---

# 8. Key conceptual realization: Chronos already personalizes through context

The important realization was that the supposedly "global" frozen Chronos model was already given four days of observations from the new individual.

Therefore Chronos was not making predictions from population knowledge alone.

It was already performing a form of:

> **in-context personalization.**

This changed the central experimental question from:

> Does an explicit residual correction personalize Chronos?

to:

> **How much same-entity context does a pretrained foundation model require in order to personalize effectively?**

---

# 9. Increasing personal context produced a strong improvement

The exact same 636 validation targets were forecast while varying only the amount of previous same-entity context.

The first context experiment showed:

| Context | Mean MAE |
|---:|---:|
| 1 day | 37.7235 |
| 2 days | 23.1101 |
| 4 days | 22.3207 |
| 8 days | **19.8390** |
| 16 days | 20.0662 |
| 21 days | 20.0805 |

Moving from four to eight days reduced mean MAE by:

\[
22.3207-19.8390
=
2.4817,
\]

corresponding to approximately an:

\[
\boxed{11.1\%}
\]

improvement.

Furthermore:

\[
92.5\%
\]

of validation entities had lower average MAE with eight days than with four days.

This was the first strong positive personalization result.

---

# 10. Context-threshold experiment

Because electricity has strong weekly seasonality, a finer context sweep was conducted to determine whether the improvement occurred specifically when the previous week's trajectory became visible.

The resulting mean MAE curve was:

| Context days | Mean MAE |
|---:|---:|
| 3 | 22.5941 |
| 4 | 22.3207 |
| 5 | 21.7072 |
| 6 | 21.6734 |
| 7 | **20.0710** |
| 8 | 19.8390 |
| 9 | 19.8223 |
| 10 | 19.8538 |
| 11 | 19.8092 |
| 12 | **19.7611** |
| 13 | 19.8748 |
| 14 | 20.0662 |
| 16 | 20.0662 |
| 21 | 20.0805 |
| 28 | 20.0419 |
| 31 | 20.0826 |

The best mean MAE occurred at approximately **12 days of context**:

\[
19.7611.
\]

Compared with the original four-day context:

\[
\frac{22.3207-19.7611}{22.3207}
\approx
\boxed{11.5\%}
\]

lower mean MAE.

At 12 days:

- 71.5% of individual forecast cases improved relative to four days;
- **96.2% of entities** had improved mean performance.

---

# 11. Evidence for a weekly threshold

A particularly large improvement occurs between six and seven days:

\[
21.6734
\rightarrow
20.0710.
\]

This represents a decrease of:

\[
1.6024
\]

MAE in a single additional context day.

Therefore access to approximately one complete weekly cycle is clearly important.

However, the phenomenon is **not purely a discontinuity at seven days**.

There is already improvement from:

\[
4d\rightarrow5d\rightarrow6d,
\]

and performance continues improving after seven days until approximately 9–12 days.

The current interpretation is therefore:

> Weekly structure explains a substantial part of the context threshold, but increasing same-entity evidence provides additional useful information beyond simply revealing the immediately previous corresponding weekday.

---

# 12. Too much context does not continue helping

Performance does not improve monotonically with history length.

After roughly:

\[
8-12\text{ days},
\]

additional history provides essentially no gain and causes a small deterioration.

The approximate regimes currently look like:

\[
\boxed{
\begin{array}{ll}
1-2\text{ days} & \text{insufficient personal context} \\
3-6\text{ days} & \text{gradual improvement} \\
7\text{ days} & \text{large weekly-information transition} \\
8-12\text{ days} & \text{best regime} \\
>12\text{ days} & \text{saturation / mild degradation}
\end{array}
}
\]

This suggests that **more personal data are not automatically better**.

Older information may become redundant or less relevant.

---

# 13. Main finding so far

The strongest finding from this first experimental phase is:

> **For a frozen time-series foundation model, personalization can occur through the amount of individual history supplied in context, without any parameter updates.**

For Chronos-2 on electricity, increasing same-entity context from four days to approximately 8–12 days improves forecasting by roughly 11%, with the improvement observed across almost all entities.

At the same time:

> **Explicitly correcting the model using historical personal forecast residuals does not improve generalization.**

Thus personalization appears to depend substantially more on **how the foundation model consumes individual context** than on persistent additive entity-specific biases.

---

# 14. Emerging research question

The project therefore appears to be evolving from:

> When should a global model become an individual model?

toward the more foundation-model-specific question:

> **When is in-context personalization sufficient, and when is explicit parameter adaptation necessary?**

This suggests three distinct personalization mechanisms:

\[
\text{Population pretrained model}
\]

\[
+\quad\text{individual context}
\]

\[
+\quad\text{optional parameter adaptation}.
\]

A future comparison could therefore hold the amount of personal evidence fixed and compare:

1. **in-context personalization** — give the frozen model more history;
2. **parameter-efficient adaptation** — LoRA/adapters using the same history;
3. **probabilistic/hierarchical adaptation** — infer a personal latent variable or posterior using the same history.

This would allow the project to ask whether explicit adaptation actually provides information beyond what a modern sequence foundation model can already exploit at inference time.

---

# 15. What has NOT yet been established

These experiments are still exploratory.

At present we cannot claim that:

- the phenomenon generalizes beyond electricity;
- the observed threshold is a universal personalization threshold;
- the result is independent of weekly seasonality;
- Chronos-2 behaves similarly on health or other longitudinal datasets;
- 8–12 days is optimal on unseen test entities;
- explicit parameter adaptation is unnecessary;
- the phenomenon constitutes a novel theoretical result.

All model and method development so far has used only the **53 validation entities**.

The 53 held-out test entities remain untouched.

---

# 16. Current conclusion

The initial PersTSFM hypothesis has **not failed**, but its interpretation has changed.

The experiments do not support a simple model in which accumulated personal observations are summarized into an additive individual correction.

Instead, they provide strong evidence that:

\[
\boxed{
\text{the amount and relevance of individual context itself controls
the effectiveness of personalization in a pretrained TSFM}
}
\]

and that there appears to be a **finite useful-context regime** rather than a simple "more individual data is always better" relationship.

The most interesting empirical pattern so far is therefore:

> **Insufficient context → rapid improvement → weekly-information transition → optimal context window → saturation/slight degradation.**

This provides a stronger foundation for the next stage of PersTSFM than the original residual-correction experiments.