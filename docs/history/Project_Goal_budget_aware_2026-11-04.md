# PersTSFM — Research Status and Proposed Direction

**Working title:** *Towards a Self-Configuring, Budget-Aware Framework for Time-Series Foundation Models*

**Status:** 4 November 2026  
**Stage:** B0–B3 exploratory research and zero-shot benchmarking completed; scientific-method development pending.

---

## 1. Executive summary

**The objective of PersTSFM is to build an nnU-Net-like system for time-series foundation models (TSFMs).**

Given a new forecasting dataset, a forecast horizon and a computational budget, the framework should automatically determine:

1. **Model selection:** Which pretrained TSFM should be used?
2. **Context selection:** How much historical data should be provided?
3. **Adaptation:** Is zero-shot forecasting sufficient, or would fine-tuning improve predictions?
4. **Budget allocation:** How much computation should be spent evaluating, selecting and adapting the model?

The user should not need to manually benchmark dozens of models, test context lengths or design a fine-tuning procedure.

### Current position

We have established a working evaluation infrastructure for six TSFM checkpoints, successfully benchmarked six datasets and collected preliminary evidence that model rankings and useful context lengths vary considerably.

**Completed B3 experiments:**

- Six datasets covering energy, traffic and healthcare.
- Six pretrained TSFM checkpoints.
- 159 validated model/context/baseline actions.
- 710 evaluation forecasting cases.
- Frozen cohorts, provenance checks, reproducible inference and latency measurements.

However, we have primarily built the **benchmarking infrastructure**, not the final automated system.

The next scientific challenge is to develop a method that makes good decisions **without having to execute the entire benchmark**.

---

## 2. What we have done so far

### Phase 0–1: Personalised forecasting and feasibility

The initial project explored whether TSFMs can be personalised to individual time series using limited historical data.

We investigated the relationship between the amount of available personal history, model configuration and forecasting accuracy.

The early work used electricity forecasting and subsequently expanded into multiple domains.

Key observations:

- Zero-shot foundation models are often already strong.
- Increasing context length can produce substantial gains.
- Fine-tuning, including LoRA, did not consistently improve performance in the initial experiments.
- Model selection based on limited historical observations is not necessarily stable.

These findings motivated a shift towards **choosing the appropriate forecasting strategy before adapting model parameters**.

### Phase 2: Model and context selection

We developed a multi-model evaluation pipeline across six checkpoints:

| Model family | Checkpoint |
|---|---|
| Chronos | Chronos-2 |
| TimesFM | TimesFM3 |
| TiRex | TiRex-2 |
| Moirai | Moirai2 Small |
| Toto | Toto2 22M |
| Toto | Toto2 313M |

The infrastructure supports different model APIs, context lengths, prediction horizons and Conda environments.

Early experiments investigated forecasting accuracy, inference cost and the ability to choose a model from limited calibration observations.

Some inexpensive forecasting strategies were competitive, but learned selection and routing methods did not consistently outperform strong fixed alternatives across evaluation settings.

**Lesson:** A complex routing algorithm is not necessarily justified when a simple model or longer context already works well.

### Phase B0–B2: Budget-aware forecasting

We investigated selecting forecasting strategies under inference constraints.

This included:

- Fixed-model and fixed-context baselines.
- Calibration-based model selection.
- Computational latency measurements.
- Simple learned routing strategies.
- Budget-aware selection and accuracy–latency comparisons.

One exploratory ridge-based strategy produced an encouraging improvement in an earlier setting, but subsequent experiments did not establish consistent superiority across the accuracy–compute frontier.

This led us to prioritise a stronger and more generalisable benchmark before developing another selection algorithm.

### Phase B3: Systematic zero-shot benchmarking

We then built a more rigorous framework with:

- Frozen forecasting origins and ground-truth targets.
- Exact comparison across candidate actions.
- Calibration/evaluation separation.
- Model-specific inference environments.
- GPU inference profiling.
- Context-length sweeps.
- Prediction manifests and data hashes.
- Explicit validation of label availability.

For the healthcare datasets, we additionally introduced **entity-local chronological selection**, because patient-specific or relative recording timestamps cannot always be compared using a global calendar boundary.

This completed the foundation for reproducible comparisons.

Importantly, **B3 contains zero-shot inference, not newly trained versions of the six foundation models**.

---

## 3. Final B3 benchmark results

All six dataset reports passed validation.

| Dataset | Evaluation cases | Best observed zero-shot strategy | Best MAE | Calibration-selected strategy | Selection regret |
|---|---:|---|---:|---|---:|
| Electricity | 103 | Moirai2 Small, 14d | 35.2932 | Chronos-2, 31d | 5.26% |
| PEMS-BAY | 147 | TiRex-2, 5d | 1.5685 | TimesFM3, 7d | 7.89% |
| Solar | 56 | Chronos-2, 4d | 0.0217 | Chronos-2, 4d | 0% |
| HUPA-UCM | 40 | Toto2 313M, 7d | 11.6604 | TiRex-2, 3d | 11.37% |
| VitalDB | 86 | Moirai2 Small, 1h | 6.5209 | Moirai2 Small, 2h | 1.45% |
| PhysioNet 2019 | 278 | TimesFM3, 2d | 7.4216 | TimesFM3, 2d | 0% |

**Total: 710 evaluation cases and 159 validated actions.**

Selection regret here is the percentage by which the calibration-selected configuration's evaluation MAE exceeds the best observed evaluation MAE.

**Important qualification:** The best observed strategy is a retrospective oracle over the configurations tested. It is not a deployable policy, and the MAE values are not directly comparable across datasets because target scales differ.

### Finding A — No single checkpoint wins everywhere

Different architectures achieved the lowest observed evaluation error on different datasets.

This supports the motivation for model selection, but it does not yet establish that a learned router is better than simple validation-based selection.

### Finding B — More context does not always help

For electricity, longer contexts showed diminishing or negative returns.

| Electricity model | 7 days | 14 days | 21 days | 31 days |
|---|---:|---:|---:|---:|
| Moirai2 Small | 40.548 | **35.293** | 35.437 | 36.115 |
| Chronos-2 | **36.222** | 36.996 | 36.884 | 37.148 |

For PEMS-BAY, TiRex-2 achieved a lower MAE with five days of context than with seven.

This suggests context length should be treated as an explicit decision variable—not automatically maximised.

### Finding C — Limited calibration sometimes selects the wrong configuration

The largest observed gaps were:

- HUPA-UCM: 11.37%.
- PEMS-BAY: 7.89%.
- Electricity: 5.26%.

On Solar and PhysioNet 2019, calibration selected the best observed evaluation configuration.

Thus, calibration selection works in some settings but is not uniformly reliable.

The next step is to understand **when calibration is informative, when it is uncertain and when spending more computation on selection is justified**.

### Finding D — Inference cost differs substantially

The models have heterogeneous measured inference latencies.

For example, in PEMS-BAY:

- TiRex-2 5d: approximately 20 ms.
- TimesFM3 7d: approximately 117 ms.

The faster configuration also achieved lower evaluation error.

This indicates that some configurations may be dominated in both accuracy and runtime.

However, B3's warm single-case inference measurements do not represent full deployment cost. Model loading, calibration runs, training, GPU memory and repeated predictions must also be included.

---

## 4. What the actual research problem should be

The project should no longer be described simply as:

*Find the best TSFM for each dataset.*

Instead, the proposed problem is:

**How can we automatically configure a time-series foundation model for a previously unseen forecasting task, including its context length and optional adaptation, while respecting a finite computational budget?**

The distinction is crucial because identifying the best configuration by evaluating everything is expensive.

Our framework must make a good decision using only a small fraction of that information.

### The nnU-Net analogy

nnU-Net combines fixed design choices, rules derived from dataset properties and empirical configuration decisions.

PersTSFM could follow the same philosophy.

| nnU-Net principle | PersTSFM equivalent |
|---|---|
| Dataset fingerprint | Statistical and structural properties of time series |
| Automated preprocessing | Frequency handling, missingness, scaling and temporal validation |
| Architecture configuration | Selecting a TSFM checkpoint |
| Resolution/patch configuration | Selecting historical context length |
| Training configuration | Selecting zero-shot, lightweight adaptation or fine-tuning |
| Empirical model selection | Limited calibration or validation |
| Resource constraints | GPU memory, time and inference/training budget |
| Automatic prediction pipeline | Reproducible configured forecasting model |

The objective is a **self-configuring forecasting pipeline**, not a new universal neural-network architecture.

---

## 5. Proposed architecture

A practical PersTSFM system could contain five modules.

### Module A — Dataset fingerprinting

Before running an expensive foundation model, extract inexpensive characteristics of the forecasting task:

- Sampling frequency, horizon and available history.
- Number of entities and observations.
- Missingness and irregularity.
- Trend, autocorrelation and seasonality.
- Noise, variability and nonstationarity.
- Differences between entities.
- Available training labels and recent distribution changes.

These features should be computed using only information available at decision time.

**Output:** A compact representation of the dataset or forecasting regime.

### Module B — Model and context planner

Given the fingerprint, select promising combinations of:

- Pretrained TSFM.
- Context length.
- Available inference hardware.
- Forecast horizon.

Rather than evaluating all 24–30 combinations, the planner might initially recommend only two or three candidates.

This is the first genuinely automated component we should develop.

### Module C — Budgeted calibration

The system decides whether additional model evaluations are worth their cost.

For example:

- Under a very small budget, select using fingerprints alone.
- Under a moderate budget, evaluate two candidate configurations.
- Under a larger budget, evaluate more candidates or multiple calibration origins.

The scientific challenge is deciding **which evaluation to perform next and when to stop**.

### Module D — Adaptation decision

After selecting a promising zero-shot configuration, determine whether additional adaptation is worthwhile.

Candidate actions could include:

- Keep the pretrained model frozen.
- Apply inexpensive calibration or a simple residual correction.
- Fine-tune a small adapter using LoRA.
- Consider more expensive task-specific adaptation only when justified.

The decision must consider the history available for training, expected improvement, risk of overfitting and training cost.

**Fine-tuning should be an option, not an assumed requirement.**

### Module E — Final configuration and reporting

The system returns:

- Selected checkpoint.
- Context length.
- Adaptation method, if any.
- Expected inference and setup costs.
- Calibration evidence and uncertainty.
- A reproducible forecasting configuration.

An eventual user-facing interface could resemble:

`PersTSFM.fit(data, horizon, budget, objective)`

The framework would handle the configuration choices internally.

---

## 6. What could be scientifically novel?

A generic TSFM router is not sufficient by itself.

TimeRouter (2026) already investigates lightweight expert routing and selective gating. AutoGluon-TimeSeries already offers automated model evaluation, time budgets, ensembles and TSFM fine-tuning.

Our proposed contribution needs to be more specific.

### Proposed central contribution

**Joint, budget-aware selection of pretrained model, context length and adaptation strategy, with uncertainty-aware allocation of calibration and training computation.**

Potential methodological elements:

1. **Cost-aware hierarchical decision-making.** Select model and context first, then evaluate whether adaptation is worth investigating.
2. **Adaptive computational allocation.** Spend additional compute only when its estimated benefit exceeds the cost.
3. **Learning whether to fine-tune.** Predict when LoRA is likely to provide enough improvement to justify its cost.
4. **Safe fallback and uncertainty.** Retain a robust zero-shot option when selection evidence is weak.

These are research hypotheses, not completed contributions.

The strongest direction would be a method that makes the full decision more efficiently and reliably than independently optimising each component.

---

## 7. Do we need more datasets?

**Yes, but not more of the same kind.**

The current six datasets were sufficient to establish the infrastructure and reveal model heterogeneity.

They are **not sufficient for strong cross-task meta-learning or claims that the planner generalises to previously unseen datasets**.

We currently have hundreds of forecasting origins but only six dataset-level environments. Forecast origins from the same dataset do not constitute hundreds of independent meta-learning tasks.

### Main gaps in the existing benchmark

- Limited numbers of genuinely independent datasets.
- Only one forecasting horizon per current dataset.
- Primarily univariate target forecasting.
- Limited coverage of covariates and multivariate relationships.
- Limited retail, weather, industrial and macroeconomic data.
- Limited testing of genuinely new domains.
- Healthcare cohorts primarily measure within-entity forecasting rather than unseen-patient adaptation.

### Recommended additions

| Priority | Dataset type | Why we need it | Candidate sources |
|---|---|---|---|
| High | Retail and demand forecasting | Intermittency, promotions, irregular patterns | FEV tasks, M5-style demand |
| High | Weather and environmental | Strong seasonality, multiple scales | Jena Weather, FEV/GIFT-Eval |
| High | Industrial sensors | Nonstationarity, heterogeneous operating regimes | ETT, industrial benchmark tasks |
| High | Diverse frequencies and horizons | Understand context/horizon interactions | M4 Hourly, FEV |
| Medium | Finance and macroeconomic | Noisy and weakly predictable series | FEV/GIFT-Eval |
| Medium | Additional medical/wearable data | Independent clinical/physiological regimes | Public wearable or physiological forecasting datasets |

The best approach is to use established benchmark protocols rather than construct many new datasets manually.

### Recommended external benchmarks

**FEV / fev-bench:** Particularly useful because it defines standard forecasting tasks, evaluation windows and reproducible task-level metrics across heterogeneous domains.

**GIFT-Eval:** Valuable for testing robustness across frequencies, horizons and different forecasting regimes.

**TIME (2026):** Potentially useful as a newer task-centric benchmark, especially for evaluating generalisation beyond heavily reused datasets.

### Practical dataset target

Start with **8–12 carefully chosen external task configurations** across at least four additional domains or data regimes.

If the method shows promise, expand towards **20–40 distinct task configurations**, with genuinely unseen source datasets reserved for final testing.

We should not immediately run all six TSFMs across hundreds of benchmark tasks.

The new datasets should be chosen for statistical diversity and experimental coverage, not simply for increasing the total sample count.

---

## 8. Proposed experiments: B4–B6

### B4 — Learn to select model and context

**Question:** Can inexpensive dataset features predict promising model/context configurations on unseen tasks?

Develop a fingerprint-based planner and compare it against:

- One fixed strong TSFM.
- The best fixed configuration selected from development tasks.
- Selecting the model using calibration MAE.
- Selecting the context using calibration MAE.
- Simple heuristic rules.
- A lightweight learned router.
- Full-grid retrospective oracle, used only as an upper-bound reference.

The key metrics are forecasting regret, model-selection cost and performance on held-out tasks.

**Success condition:** The planner offers a reproducible improvement over strong, inexpensive baselines, not merely over random selection.

### B5 — Budget-aware adaptive evaluation

**Question:** Can the planner determine how much computation is worth spending before committing to a configuration?

Evaluate several budgets covering very cheap selection, limited calibration and more extensive model evaluation.

Investigate whether the system can:

- Identify when additional evaluations are useful.
- Stop when a configuration is sufficiently promising.
- Avoid costly candidates that are unlikely to improve the result.
- Produce an accuracy–compute frontier.

The full cost should include model loading, calibration, inference and memory usage, with costs reported both per forecast and amortised over repeated forecasts.

**Success condition:** Better accuracy at a matched budget, or lower cost at matched accuracy, than fixed evaluation schedules and standard model selection.

### B6 — Decide whether fine-tuning is worthwhile

**Question:** Given the chosen zero-shot configuration, can the system predict when adaptation will be beneficial?

Begin with one well-supported fine-tunable architecture, such as Chronos-2, rather than simultaneously implementing fine-tuning for all six families.

Test:

- No adaptation.
- Simple calibration or residual correction.
- LoRA fine-tuning with controlled training budgets.
- Always fine-tune versus never fine-tune.
- A learned or rule-based fine-tuning gate.

Vary available training history and data regime, always evaluating on future unseen targets.

The objective is not to demonstrate that LoRA always helps.

Instead, **the system should learn when fine-tuning is worth doing and when it should be avoided**.

**Success condition:** The adaptation decision improves net forecasting utility over both always-fine-tune and never-fine-tune policies, with training costs properly accounted for.

---

## 9. Experimental design requirements

Before claiming a generalisable automated system, we need several improvements.

**Cross-dataset generalisation:** Train the planner on development datasets and test it on genuinely unseen source datasets. Different forecast origins from the same dataset must not leak across the meta-learning split.

**Chronological validity:** Calibration and adaptation may use only historical observations and labels that have become available. Healthcare entity-local forecasting and pooled chronological deployment should be reported separately.

**Statistical uncertainty:** Use entity-level or appropriately clustered uncertainty estimates. Forecast origins within the same entity are correlated.

**Fair baselines:** Include classical forecasting methods and AutoGluon-style model selection, not just six pretrained checkpoints.

**Budget accounting:** Include selection overhead, model load time, memory, training and inference. Warm single-case latency is only one part of the cost.

**Task diversity:** Evaluate heterogeneous frequencies, context-to-horizon ratios, series lengths, covariates and forecast regimes.

**Ablations:** Separately measure contributions from fingerprints, context selection, adaptive calibration and fine-tuning decisions.

**Oracle separation:** Never use the best observed evaluation configuration as a training label or policy input for the final held-out task.

---

## 10. Proposed research contribution and paper direction

A possible future paper would be:

*PersTSFM: Self-Configuring Time-Series Foundation Models under Computational Budgets*

The paper would investigate three questions:

**RQ1 — Model/context configuration:** Can task characteristics predict which pretrained model and historical context are appropriate?

**RQ2 — Budget allocation:** Can the system efficiently decide which candidate configurations to evaluate and when to stop?

**RQ3 — Adaptation:** Can it identify when fine-tuning improves net forecasting utility and when zero-shot inference is preferable?

The overall optimisation objective is to minimise forecast error subject to a computation budget, allowing for selection and adaptation costs.

### What would make this compelling?

A method that:

- Works across previously unseen forecasting tasks.
- Requires substantially fewer expensive model evaluations.
- Selects context lengths automatically.
- Accounts for model-loading, inference and training costs.
- Avoids unnecessary fine-tuning.
- Matches or exceeds strong AutoML and routing baselines under comparable budgets.

### What would make it weak?

A system that merely runs all models, picks the best validation MAE and then optionally launches LoRA.

That would be useful engineering, but much of its core functionality already exists in AutoML frameworks.

---

## 11. Immediate priorities

We should now stop broad, exploratory zero-shot benchmarking and proceed in a controlled order.

| Step | Deliverable | Decision |
|---|---|---|
| 1 | Freeze and document B0–B3 experimental results | Establish reproducible baseline |
| 2 | Consolidate six-dataset statistics and uncertainty | Quantify what current evidence supports |
| 3 | Review TimeRouter, AutoGluon and related methods in detail | Identify a defensible methodological gap |
| 4 | Define 8–12 external forecasting tasks | Create a generalisation-development benchmark |
| 5 | Build a minimal fingerprint-based model/context planner | Test whether predictive selection is feasible |
| 6 | Compare against simple heuristics and existing routing approaches | Go/no-go on the selection contribution |
| 7 | Add adaptive calibration and explicit budgets | Establish the budget-aware contribution |
| 8 | Investigate LoRA and adaptation decisions | Extend to full self-configuration |
| 9 | Validate on untouched datasets | Assess publication readiness |

**We should not proceed to B6 fine-tuning policies unless the earlier experiments show that task-level configuration and budget allocation can be evaluated rigorously.**

---

## 12. Overall assessment

### What has been achieved

PersTSFM now has a functioning multi-model research infrastructure and six completed internal zero-shot benchmarks.

We have established:

- Meaningful variation in model rankings across datasets.
- Context-length sensitivity and diminishing returns.
- Imperfect model/context selection from limited calibration.
- Considerable differences in measured inference latency.
- Reproducible, leakage-checked forecasting cohorts across multiple domains.

This provides a strong experimental starting point.

### What has not been demonstrated

We have not yet shown:

- A generalisable automatic configuration policy.
- Superiority over strong existing routing or AutoML methods.
- Reliable prediction of fine-tuning benefits.
- A complete budget-aware adaptation strategy.
- Cross-dataset generalisation on a sufficiently diverse external benchmark.

These are the questions that should define the next phase.

### Recommended direction

**Keep the nnU-Net inspiration as the central vision, but organise the research around budget-aware automatic configuration rather than model routing alone.**

The near-term scientific priority is to demonstrate that a planner can predict useful model/context choices and allocate limited evaluation computation efficiently.

Fine-tuning should subsequently be integrated as another possible decision, rather than treated as the starting point.

**The long-term objective is a forecasting system that automatically knows what to run, how to configure it, how much to spend, and whether further training is worth the cost.**

---

## Key related work

1. [nnU-Net — Isensee et al., Nature Methods (2021)](https://doi.org/10.1038/s41592-020-01008-z) — Self-configuration using dataset fingerprints and automatic pipeline design.
2. [TimeRouter — Ning et al. (2026)](https://arxiv.org/abs/2606.11625) — Adaptive routing among pretrained TSFMs.
3. [AutoGluon-TimeSeries](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.fit.html) — Automated forecasting, validation, model selection, ensembles and training budgets.
4. [fev-bench — Shchur et al. (2025)](https://arxiv.org/abs/2509.26468) — Reproducible evaluation across heterogeneous forecasting tasks.
5. [GIFT-Eval — Aksu et al. (2024)](https://arxiv.org/abs/2410.10393) — Diverse TSFM evaluation across domains, frequencies and horizons.
6. [It's TIME — Qiao et al. (2026)](https://arxiv.org/abs/2602.12147) — Task-centric benchmarking using newer forecasting datasets.
7. [In-Context Fine-Tuning for Time-Series Foundation Models — Faw et al., ICML (2025)](https://proceedings.mlr.press/v267/faw25b.html) — Relevant alternative approach to time-series adaptation.

*Scope note: The B3 numerical results are taken from the validated terminal reports shared during this project. Earlier B0–B2 findings are retrospective summaries and should be checked against their original experiment artifacts before use in a manuscript. The proposed architecture and research contributions are future work, not completed experiments.*
