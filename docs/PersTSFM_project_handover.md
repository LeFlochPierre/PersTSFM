# PersTSFM — Complete Project Handover

## 1. Overall research goal

The project is developing an **nnU-Net-inspired self-configuring framework for time-series foundation models (TSFMs)**.

The core question is:

> Can we inspect a new forecasting dataset, derive a dataset/task fingerprint, and automatically restrict the expensive model/context search to a small number of promising configurations while retaining near-oracle forecasting performance?

The high-level pipeline is:

\[
\text{Dataset}
\rightarrow
\text{TS fingerprint}
\rightarrow
\text{forecasting plan}
\rightarrow
\text{small candidate set}
\rightarrow
\text{validation}
\rightarrow
\text{final forecaster}.
\]

The current configuration dimensions are:

\[
\boxed{\text{TSFM family}+\text{context length}}
\]

Current TSFMs:

```text
Chronos-2
TimesFM3
TiRex-2
```

The current framework does **not** directly predict which model is best.

Instead:

\[
\text{fingerprint}
\rightarrow
\text{small set of model/context candidates}
\rightarrow
\text{evaluate candidates}
\rightarrow
\text{choose best observed candidate}.
\]

This distinction is important.

The exhaustive/full search is performed **only for research evaluation**, so we know the oracle and can calculate regret. In actual deployment on a new dataset, exhaustive search would not be performed.

---

# 2. Repository / environment

Repo:

```bash
~/RDS/home/projects/PersTSFM
```

Conda environment:

```bash
PERTSFM_ENV
```

Main workstation GPU:

```text
NVIDIA RTX PRO 6000 Blackwell
```

Main self-configuration directories:

```text
configs/selfconfig/
src/selfconfig/
results/selfconfig/fingerprints/
results/selfconfig/benchmark_matrix/
results/selfconfig/plans/
results/selfconfig/evaluation/
results/selfconfig/execution/
tests/selfconfig/
docs/
```

Important existing selfconfig source files:

```text
src/selfconfig/regime.py
src/selfconfig/memory.py
src/selfconfig/rules.py
src/selfconfig/search_space.py
src/selfconfig/dataset_adapter.py
src/selfconfig/fingerprint.py
src/selfconfig/benchmark.py
src/selfconfig/planner.py
src/selfconfig/evaluation.py
src/selfconfig/capacity.py
src/selfconfig/selection.py
src/selfconfig/execute.py
src/selfconfig/policy_simulation.py
```

Relevant docs/config:

```text
configs/selfconfig/phase1.yaml
docs/selfconfig_protocol.md
docs/selfconfig_phase1_freeze.md
docs/README_SELFCONFIG.md
```

Known frozen Phase-1 commit:

```text
d27a6b3
```

That commit froze the original V1 planner before looking at forecasting-performance results.

Later executor/policy-simulation work was added after that; current Git commit status should be checked in the new chat.

---

# 3. Development datasets

Six datasets are currently used as the **development suite**:

```text
electricity
pems_bay
hupa_ucm
vitaldb
solar
physionet2019
```

They span very different domains:

```text
Electricity       electricity demand/load
PEMS-BAY          traffic
HUPA-UCM          glucose/physiology
VitalDB           anaesthesia / physiological monitoring
Solar             solar generation
PhysioNet2019     ICU / clinical trajectories
```

These six datasets must now be treated as **development datasets**, not future held-out evidence.

We have inspected their forecasting results and used them to design V1.1/V1.2.

Therefore any final generalisation claim must come from completely new datasets/tasks.

---

# 4. Forecasting protocols

Important protocol properties:

### Electricity

```text
resolution = 15 min
forecast horizon H = 96 steps = 24 h
53 validation entities
12 origins/entity
636 cases
```

### PEMS-BAY

```text
resolution = 5 min
H = 12 steps = 1 h
```

### HUPA-UCM

```text
resolution = 5 min
H = 12 steps = 1 h
development split
```

### VitalDB

```text
resolution = 1 min
H = 15 steps
```

### Solar

```text
resolution = 10 min
H = 6 steps = 1 h
```

### PhysioNet2019

```text
resolution = 60 min
H = 6 steps = 6 h
```

Existing forecasting infrastructure is reused rather than duplicated.

Important existing files:

```text
src/data/registry.py
src/evaluation/context_curve.py
src/experiments/run_context_curve.py
src/models/adapters.py
```

`context_curve.py` supports:

```text
chronos2
timesfm3
tirex2
```

---

# 5. Fingerprint design

The dataset fingerprint is computed using historical data only, respecting the forecasting evaluation boundary.

Important fix already made:

Originally `eval_index` could be interpreted as a raw-series position.

The adapter was fixed to prioritize:

```text
eval_origin
```

matched to the actual DatetimeIndex, with `eval_index` as fallback.

The leakage boundary is the earliest evaluation origin.

Only history before the first evaluation origin for each entity contributes to the fingerprint.

---

# 6. Base fingerprint features

Current features include:

```text
number of series
series-length statistics
sampling interval
forecast horizon
number of variables
missingness
zero fraction
constant fraction
median / IQR
skewness
kurtosis
coefficient of variation
lag-1 ACF
horizon ACF
trend strength
dominant period
seasonality strength
spectral entropy
between-series scale variability
between-series mean variability
between-series ACF variability
```

---

# 7. Temporal-memory fingerprint

`src/selfconfig/memory.py`

Computes ACF at:

\[
H,\;2H,\;4H,\;8H,\;16H,\;32H
\]

plus:

\[
P,\;2P
\]

where \(P\) is the dominant period.

It also computes:

```text
memory_lag_02_steps
memory_lag_01_steps
```

defined as the largest tested horizon-relative lag where median absolute ACF remains ≥0.20 or ≥0.10 respectively.

Important caveat:

These are persistence scales **among the tested lags**, not exact decorrelation times.

---

# 8. Regime fingerprint

`src/selfconfig/regime.py`

Features include:

```text
recent-vs-prior mean shift
scale shift
IQR shift
quantile shift
rolling mean variability
rolling scale variability
adjacent mean shift
window-mean persistence
usable fraction
median profile windows
```

Current windows:

```text
recent = 4H
prior = up to 28H before recent
rolling windows = non-overlapping 4H windows
history considered = latest max 64H
```

HUPA was especially important here:

```text
mean shift ≈ 1.002
adjacent mean shift ≈ 0.642
seasonality strength ≈ 0.080
```

which made it a regime-rich rather than periodic dataset.

---

# 9. History-capacity fingerprint

`src/selfconfig/capacity.py`

The planner computes:

```text
min
p10
p25
median
p75
p90
max
horizon
robust limit
resource limit
effective limit
```

Current rule:

```python
robust_limit = floor(p25_history_steps)
resource_limit = 4096
effective_limit = min(robust_limit, resource_limit)
```

`4096` is a compute-budget constraint, not a scientific optimality claim.

Observed effective limits:

```text
Electricity       4096
PEMS-BAY          4096
HUPA-UCM          2015
VitalDB            239
Solar              575
PhysioNet2019       47
```

---

# 10. Useful fingerprint observations

### Electricity

```text
H = 96
dominant period ≈ 96.38
seasonality ≈ 0.911
spectral entropy ≈ 0.344
ACF(P) ≈ 0.899
memory_lag_02 ≈ 3072
capacity = 4096
```

Very strong daily periodicity and very long persistence.

### PEMS-BAY

```text
H = 12
dominant period ≈ 292.57
seasonality ≈ 0.385
spectral entropy ≈ 0.556
ACF(P) ≈ 0.568
ACF(2P) ≈ 0.300
memory_lag_02 ≈ 24
capacity = 4096
```

Important observation:

The existing simple memory feature did **not** predict the huge long-context benefit seen empirically.

This is one reason not to simply replace the PEMS rule with a memory-lag rule.

### Solar

```text
H = 6
dominant period ≈ 143.75
seasonality ≈ 0.872
spectral entropy ≈ 0.257
ACF(P) ≈ 0.872
ACF(2P) ≈ 0.897
capacity = 575
```

### HUPA-UCM

```text
H = 12
dominant period ≈ 287.86
seasonality ≈ 0.080
spectral entropy ≈ 0.481
capacity = 2015
```

Weak periodicity, strong regime change.

### VitalDB

```text
H = 15
dominant period ≈ 47.8
seasonality ≈ 0.145
spectral entropy ≈ 0.789
capacity = 239
```

### PhysioNet2019

```text
H = 6
dominant period ≈ 11.75
seasonality ≈ 0.242
spectral entropy ≈ 0.740
capacity = 47
```

---

# 11. Original V1 planner

The original frozen planner proposed **three contexts per model**.

With three TSFMs:

\[
3\text{ models}\times3\text{ contexts}=9
\]

configurations per dataset.

Search branches included periodic, regime-rich and ordinary cases.

Important constants included:

```text
strong seasonality       0.50
moderate seasonality     0.30
periodic ACF             0.25
high horizon ACF         0.30
regime mean shift        0.50
regime minimum capacity  32H
regime long anchor max   128H
```

The planner was frozen before forecasting-performance results were inspected.

---

# 12. Frozen V1 proposed plans

### Electricity

```text
chronos2: 96, 192, 768
timesfm3: 96, 192, 768
tirex2:   96, 192, 768
```

### PEMS-BAY

```text
chronos2: 12, 293, 1172
timesfm3: 12, 293, 1172
tirex2:   12, 293, 1172
```

### HUPA-UCM

```text
chronos2: 12, 96, 1536
timesfm3: 12, 96, 1536
tirex2:   12, 96, 1536
```

### VitalDB

```text
chronos2: 15, 60, 120
timesfm3: 15, 60, 120
tirex2:   15, 60, 120
```

### Solar

```text
chronos2: 6, 144, 575
timesfm3: 6, 144, 575
tirex2:   6, 144, 575
```

### PhysioNet2019

```text
chronos2: 6, 24, 47
timesfm3: 6, 24, 47
tirex2:   6, 24, 47
```

All feasibility/model/history invariants passed.

---

# 13. Frozen-plan executor

Added:

```text
src/selfconfig/execute.py
tests/selfconfig/test_execute.py
```

CLI:

```bash
python -m src.selfconfig.execute \
  --dataset electricity \
  --mode proposed \
  --models chronos2 timesfm3 tirex2
```

Modes:

```text
proposed
full
```

Outputs:

```text
results/selfconfig/execution/<mode>/<dataset>/context/<model>/
results/selfconfig/benchmark_matrix/<mode>/<dataset>.csv
results/selfconfig/benchmark_matrix/<mode>/<dataset>_manifest.json
```

The executor reuses the existing context-curve runner.

---

# 14. Important executor compatibility issue

The old context-curve infrastructure requires a historical reference context.

If that context is absent from the planner candidates, `execute.py` temporarily adds it as an auxiliary evaluation.

Example Electricity:

Planner asks for:

```text
96
192
768
```

but the old runner internally executes:

```text
96
192
384   <- auxiliary reference only
768
```

The final benchmark matrix filters out `384`.

Therefore the scientific candidate count remains **3 contexts/model**, not four.

The manifest records this explicitly.

For final runtime/compute claims this should be cleaned up.

Recommended future change:

```text
allow context-curve evaluation without a mandatory reference context
```

or implement:

```text
run_context_candidates(...)
```

so exact candidate-only runtime can be measured.

---

# 15. V1 exhaustive-search results

We ran both the proposed V1 search and full exhaustive search for all six development datasets.

## Electricity

Self-config:

```text
Chronos-2
768 steps
192 h = 8 days
MAE = 19.838995
```

Oracle:

```text
Chronos-2
768
MAE = 19.838995
```

Result:

```text
relative regret = 0%
proposed configs = 9
full configs = 19
search reduction = 52.6%
```

Interesting full-search trend:

```text
Chronos 768   19.8390
Chronos 1536  20.0662
Chronos 3072  20.0784
Chronos 4096  20.1177
```

So 768 is genuinely an interior optimum rather than “longer is always better”.

---

## VitalDB

Self-config:

```text
TiRex-2
120 steps = 2 h
MAE = 6.444987
```

Oracle:

```text
TimesFM3
239 steps ≈ 3.98 h
MAE = 6.290378
```

Result:

```text
absolute regret = 0.154608
relative regret = 2.458%
proposed = 9
full = 15
search reduction = 40%
```

Failure mode:

The oracle is at the history-capacity boundary.

---

## PhysioNet2019

Self-config:

```text
TimesFM3
47 steps = 47 h
MAE = 7.259059
```

Oracle identical.

Result:

```text
regret = 0%
proposed = 9
full = 12
search reduction = 25%
```

---

## HUPA-UCM

V1 self-config:

```text
TiRex-2
1536 steps = 128 h
MAE = 13.520873
```

Oracle:

```text
TiRex-2
2015 steps ≈ 167.9 h
MAE = 13.384051
```

Result approximately:

```text
relative regret = 1.022%
proposed = 9
full = 27
search reduction = 66.7%
```

Again the oracle sits at the history-capacity boundary.

---

## Solar

Self-config:

```text
Chronos-2
575 steps ≈ 95.83 h
MAE = 0.025897
```

Oracle identical.

Result:

```text
regret = 0%
proposed = 9
full = 30
search reduction = 70%
```

---

## PEMS-BAY

This was the important V1 failure.

Self-config:

```text
TiRex-2
1172 steps ≈ 97.67 h
MAE = 1.532574
```

Oracle:

```text
TimesFM3
4096 steps ≈ 341.33 h
MAE = 1.318055
```

Result:

```text
absolute regret = 0.214518
relative regret = 16.275%
proposed = 9
full = 39
search reduction = 76.9%
```

The TimesFM3 curve continues improving strongly:

```text
1172   1.5471
1536   1.4497
2344   1.3493
3072   1.3261
4096   1.3181
```

This showed that the original periodic structural anchor stopped too early.

---

# 16. Overall V1 result

Across all six development datasets:

```text
54 proposed configurations
142 exhaustive configurations
```

Overall search reduction:

\[
1-\frac{54}{142}
=
61.97\%.
\]

V1 performance:

```text
mean relative regret   3.2926%
median relative regret 0.5111%
maximum regret        16.2754%
exact oracle datasets 3/6
```

The PEMS failure dominates the mean regret.

This was a useful result because it exposed a concrete weakness rather than making the system appear artificially perfect.

---

# 17. Key V1 insight

The main failure pattern was:

```text
VitalDB   oracle at capacity
HUPA      oracle at capacity
PEMS      oracle at capacity
```

For VitalDB/HUPA, missing the capacity endpoint caused only small regret.

For PEMS, it caused large regret.

This led to the idea that every candidate search should retain a **history-capacity sentinel**.

Rather than changing something arbitrary like:

```text
4P -> 8P
```

because PEMS failed, the more defensible principle is:

> A structural fingerprint guess may miss useful long-range information, therefore retain one candidate representing the maximum robustly usable/model-feasible history.

---

# 18. V1.1 — Three-anchor planner

We designed:

\[
\boxed{
C_{\text{short}},
C_{\text{structural}},
C_{\text{capacity}}
}
\]

Interpretation:

```text
SHORT       local/minimum forecasting information
STRUCTURAL  fingerprint-derived characteristic scale
CAPACITY    maximum robust/model-feasible usable history
```

For the offline simulation:

### SHORT

The smallest V1-proposed context.

### STRUCTURAL

The largest context proposed by V1 — the existing fingerprint/rule-derived long guess.

### CAPACITY

The largest model-feasible context represented in the frozen exhaustive search.

If STRUCTURAL = CAPACITY, retain the old V1 middle candidate as the third distinct context.

Examples from the replay:

### Electricity

```text
Chronos:
short      96
structural 768
capacity   4096

TimesFM:
96 / 768 / 4096

TiRex:
96 / 768 / 1536
```

### VitalDB

```text
15 / 120 / 239
```

### HUPA

```text
12 / 1536 / 2015
```

### PEMS

```text
Chronos: 12 / 1172 / 4096
TimesFM: 12 / 1172 / 4096
TiRex:   12 / 1172 / 1536
```

The model-specific TiRex capacity is lower because of its model context limit.

---

# 19. V1.1 offline replay result

Added:

```text
src/selfconfig/policy_simulation.py
tests/selfconfig/test_policy_simulation.py
```

Command:

```bash
python -m src.selfconfig.policy_simulation
```

The replay uses the already-computed exhaustive matrices, so no GPU inference is needed.

V1.1 results:

```text
exact oracle = 6/6 datasets
relative regret = 0% on all six
total evaluations = 54
total exhaustive = 142
overall search reduction = 61.97%
```

So simply replacing the old middle/long structure with:

```text
SHORT + STRUCTURAL + CAPACITY
```

makes the candidate set contain the exhaustive-search optimum on all six development datasets.

Important caveat:

This is a **development result**. V1.1 was designed after inspecting V1 failures.

It cannot yet be called generalisation.

---

# 20. V1.2 — Adaptive search

We then designed an adaptive policy intended to use fewer than nine configurations.

Current offline V1.2 algorithm:

## Stage 1

For every model evaluate:

\[
C_{\text{short}}
\quad\text{and}\quad
C_{\text{structural}}.
\]

Three models × two contexts:

\[
\boxed{6\text{ initial evaluations}}
\]

## Stage 2

Compare models at their STRUCTURAL context.

Let:

\[
L^*_{\mathrm{struct}}
=
\min_m L(m,C_{\mathrm{struct},m}).
\]

A model is considered competitive if:

\[
L(m,C_{\mathrm{struct},m})
\le
(1+\tau)L^*_{\mathrm{struct}}
\]

where \(\tau\) is the tie/competitive band.

Current default:

\[
\boxed{\tau=2\%}
\]

## Stage 3

Only competitive models receive a CAPACITY evaluation.

If STRUCTURAL already equals CAPACITY, nothing extra is required.

Then choose the best observed configuration among everything evaluated.

This gives a dataset-dependent number of evaluations rather than always nine.

---

# 21. Why V1.2 is useful

It avoids naïve early model pruning.

We learned that model ranking can change dramatically with context.

Example PEMS:

Around 1172 steps:

```text
Chronos ≈ 1.5325
TiRex   ≈ 1.5326
TimesFM ≈ 1.5471
```

TimesFM is not best.

But at 4096:

```text
TimesFM = 1.3181
```

and becomes the global oracle.

Therefore:

> Do not simply evaluate models at short context and kill the apparent losers.

V1.2 instead asks whether a model remains **competitive enough at the structural scale to justify a long-history probe**.

---

# 22. V1.2 per-dataset result at 2% band

At:

\[
\tau=2\%
\]

results are:

### Electricity

```text
best = Chronos-2 / 768
oracle = same
regret = 0
evaluations = 7
full = 19
search reduction = 63.16%
```

Only Chronos was sufficiently competitive for the capacity probe.

### VitalDB

```text
best = TimesFM3 / 239
regret = 0
evaluations = 8
full = 15
search reduction = 46.67%
```

TimesFM and TiRex were competitive.

### PhysioNet2019

```text
best = TimesFM3 / 47
regret = 0
evaluations = 6
full = 12
search reduction = 50%
```

Capacity equals structural, so no extra evaluations were needed.

### HUPA-UCM

```text
best = TiRex-2 / 2015
regret = 0
evaluations = 8
full = 27
search reduction = 70.37%
```

Chronos and TiRex were competitive.

### Solar

```text
best = Chronos-2 / 575
regret = 0
evaluations = 6
full = 30
search reduction = 80%
```

### PEMS-BAY

```text
best = TimesFM3 / 4096
regret = 0
evaluations = 8
full = 39
search reduction = 79.49%
```

TimesFM and TiRex were considered competitive at the structural scale, allowing TimesFM's 4096 capacity candidate to be evaluated.

---

# 23. Overall V1.2 result

At a 2% competitive band:

```text
43 configurations evaluated
142 exhaustive configurations
```

Overall search reduction:

\[
1-\frac{43}{142}
=
69.7183\%.
\]

Development-set regret:

```text
mean = 0%
median = 0%
max = 0%
exact oracle = 6/6
```

So the development result is:

\[
\boxed{
0\%\text{ regret with }69.7\%\text{ search reduction}
}
\]

using only 43 evaluations versus 142 exhaustive evaluations.

Again: this is a **development result**, not yet a held-out generalisation claim.

---

# 24. V1.2 tie-band sensitivity

We replayed:

```text
0%
1%
2%
3%
5%
10%
```

Results:

```text
band    mean regret   max regret   evaluations   search reduction

0%      1.8301%       9.6192%        40          71.8310%
1%      0.2269%       1.3613%        41          71.1268%
2%      0.0000%       0.0000%        43          69.7183%
3%      0.0000%       0.0000%        43          69.7183%
5%      0.0000%       0.0000%        44          69.0141%
10%     0.0000%       0.0000%        47          66.9014%
```

So 2% is the smallest tested band that recovers the oracle on all six development datasets.

This means 2% should currently be treated as a **development-tuned hyperparameter**.

Once frozen, it should not be changed based on held-out datasets.

3% is equally successful on these six but does not save additional configurations relative to 2%.

---

# 25. Policy-simulation outputs

Saved:

```text
results/selfconfig/evaluation/v1_v11_v12_policy_replay.csv
results/selfconfig/evaluation/v12_tie_band_sensitivity.csv
```

The CLI ran successfully and reproduced the results twice.

---

# 26. Current scientific interpretation

The project has now progressed through three useful stages.

## V1

Fixed fingerprint/rule-based candidate generation.

```text
9 evaluations/dataset
3/6 exact oracle
61.97% overall search reduction
mean regret 3.29%
PEMS major failure
```

## V1.1

Add an explicit capacity sentinel.

```text
9 evaluations/dataset
6/6 exact oracle on development data
61.97% search reduction
```

## V1.2

Adaptive two-stage search.

```text
6 initial evaluations
+ selective capacity evaluations
43 total evaluations across six datasets
6/6 exact oracle on development data
69.72% search reduction
```

This is currently the strongest deterministic planner.

---

# 27. Relationship to nnU-Net

The original inspiration is nnU-Net.

The conceptual similarity is:

```text
dataset
↓
dataset fingerprint
↓
rule-based configuration
↓
limited empirical choices
↓
validated configuration
```

Our current framework is therefore reasonably nnU-Net-like.

But our future direction could go beyond nnU-Net.

---

# 28. Future V2 — learned/meta-configurator

We discussed eventually learning:

\[
\text{dataset representation}
\rightarrow
\text{configuration ranking}.
\]

Possible design:

\[
z_D
=
[
\text{handcrafted fingerprint};
\text{learned latent task embedding}
].
\]

Candidate configuration:

\[
c=
[
\text{model identity},
\text{context length},
\text{horizon},
\ldots
].
\]

Then learn:

\[
f_\theta(z_D,c)
\rightarrow
\widehat{r}(D,c)
\]

where \(r\) is relative regret rather than raw MAE.

Relative regret is preferable because MAE scales differ enormously across datasets.

The learned system could rank:

```text
TimesFM / 4096
TimesFM / 3072
TiRex / 1536
Chronos / 3072
...
```

and evaluate only the predicted top configurations.

Potential eventual target:

```text
39 full configurations
→ 9 V1 configurations
→ 6–8 V1.2 configurations
→ perhaps 2–4 learned V2 configurations
```

However:

\[
N=6
\]

datasets is far too small for a credible learned meta-configurator.

Training one now would mostly memorize:

```text
PEMS -> TimesFM long context
Solar -> Chronos
etc.
```

Therefore **do not build the latent model yet**.

---

# 29. Dataset/task expansion needed before V2

The eventual meta-dataset should contain many heterogeneous forecasting tasks.

Target order of magnitude:

\[
30-100+
\]

tasks/datasets if feasible.

Potential domains:

```text
electricity/load
traffic
solar
wind
weather
economics
retail demand
web traffic
IoT
industrial sensors
wearables
glucose
physiology
anaesthesia
ICU
other clinical time series
```

Variation should include:

```text
sampling frequency
forecast horizon
number of entities
history length
missingness
seasonality
non-stationarity
scale
domain
```

Critical evaluation rule:

**split at the dataset/task level**, not configuration level.

Example:

```text
meta-train:
D1 ... D35

meta-validation:
D36 ... D42

held-out meta-test:
D43 ... D50
```

No held-out dataset should influence the meta-model or deterministic planner.

---

# 30. Immediate next steps

The next chat should continue from here.

## Step A — Freeze/document V1/V1.1/V1.2 results

Check Git status:

```bash
git status --short
```

Make sure these are present:

```text
src/selfconfig/execute.py
tests/selfconfig/test_execute.py
src/selfconfig/policy_simulation.py
tests/selfconfig/test_policy_simulation.py
```

Run:

```bash
python -m compileall -q src/selfconfig
python -m pytest -q tests/selfconfig
```

Note:

Before the latest policy-simulation patch, the observed suite was:

```text
11 passed
```

The policy patch added three additional tests, but the latest full pytest count after adding them was not pasted in the chat. The actual current count should therefore be checked rather than assumed.

---

## Step B — Do NOT overwrite the original V1 planner

Keep V1 as a frozen historical baseline.

Version the policies explicitly:

```text
V1
V1.1
V1.2
```

Do not silently modify `rules.py` and call it the same planner.

The development progression itself is useful experimental evidence.

---

## Step C — Turn V1.1 into a real planner

Currently V1.1 is an **offline replay** using exhaustive matrices.

Need to modify the actual planning layer so it can generate:

```text
SHORT
STRUCTURAL
CAPACITY
```

without knowing oracle results.

Capacity must come from:

```text
history capacity
model-specific max context
resource limit
```

not from reading the exhaustive benchmark matrix.

The offline implementation currently uses the frozen full search to identify the corresponding feasible capacity context; production implementation should compute it directly from the fingerprint/search-space logic.

---

## Step D — Turn V1.2 into a real sequential executor

V1.2 cannot be represented as a completely static plan because Stage 2 depends on Stage-1 forecasting performance.

Production flow should be:

```text
1. fingerprint dataset

2. generate:
   SHORT
   STRUCTURAL
   CAPACITY

3. run:
   SHORT + STRUCTURAL
   for all three models

4. compare structural MAE

5. models within 2% of best structural MAE
   receive CAPACITY evaluation

6. choose best observed configuration
```

This should become a separate sequential executor/policy rather than being buried in the static planner.

---

## Step E — Keep 2% explicitly labelled as development-tuned

Current default:

\[
\tau=0.02
\]

Reason:

It was the smallest tested band giving zero development regret.

Do not change this once held-out evaluation starts.

Potential sensitivity table should stay in the eventual paper/supplement.

---

## Step F — Fix compute-cost accounting

Current `execute.py` sometimes evaluates an auxiliary legacy reference context.

This contaminates actual wall-clock/GPU-cost comparisons.

Before final compute-efficiency experiments:

```text
remove mandatory auxiliary reference evaluation
```

or add:

```python
require_reference=False
```

to the context runner.

Final efficiency metrics should include:

```text
number of model/context evaluations
wall-clock time
GPU time if feasible
possibly peak memory
```

and must reflect only configurations the policy actually requested.

---

## Step G — Add genuinely unseen datasets

The current six are now development-only.

Before changing V1.2 based on new results:

1. Freeze the policy.
2. Select new datasets.
3. Fingerprint them.
4. Generate V1.2 candidates without seeing exhaustive performance.
5. Execute V1.2.
6. Separately execute exhaustive search solely to calculate oracle regret.
7. Never alter the planner based on held-out results.

This is the critical next scientific milestone.

---

# 31. Important future baselines

Eventually compare V1.2 against:

```text
full exhaustive search
fixed default model/context
largest-context-only policy
random search with same evaluation budget
V1
V1.1
V1.2
```

Potential later baselines:

```text
Bayesian optimization
successive halving / racing
learned meta-ranking
```

Do not add complexity prematurely.

---

# 32. Main scientific hypothesis now

The emerging claim is no longer merely:

> Longer context is better.

Instead:

> Cheap dataset-level fingerprints can identify a small number of structurally meaningful context candidates, while an adaptive validation policy selectively spends additional compute where model ranking remains uncertain.

V1.2 operationalizes this as:

\[
\boxed{
\text{Fingerprint}
\rightarrow
\text{SHORT + STRUCTURAL}
\rightarrow
\text{model competition}
\rightarrow
\text{selective CAPACITY probes}
\rightarrow
\text{winner}.
}
\]

On the six development datasets:

\[
\boxed{
43/142\text{ configurations}
}
\]

were sufficient to recover all six exhaustive-search optima.

That corresponds to:

\[
\boxed{
69.7\%\text{ fewer evaluations}
}
\]

with:

\[
\boxed{
0\%\text{ development regret}.
}
\]

This is the strongest current result, but it must be validated on unseen tasks before making a generalisation claim.

---

# 33. Critical methodological safeguards

Do not:

```text
tune V1.2 after seeing held-out test results
call six development datasets independent generalisation evidence
train a latent/meta-model on only these six datasets
prune models based only on short-context performance
count auxiliary reference contexts as planner configurations
overwrite V1 and lose the historical baseline
```

Do:

```text
version every planner
freeze before held-out testing
split future meta-learning at whole-dataset level
report regret and search cost together
retain exhaustive search only as an evaluation oracle
keep interpretable fingerprint features even if a learned embedding is added later
```

---

# 34. Recommended immediate continuation prompt for the next chat

Use this after pasting the handover:

> Continue the PersTSFM project from this handover. V1, V1.1 and V1.2 are now defined and the six development-dataset exhaustive matrices already exist. The current best deterministic policy is V1.2: SHORT+STRUCTURAL for all models, then CAPACITY only for models within a 2% structural-MAE band. It achieved zero development regret with 43/142 evaluations. I now want to (1) inspect the current Git/repo state, (2) implement V1.1 and especially V1.2 as real planner/executor logic rather than offline replay, (3) preserve V1 unchanged, (4) remove the legacy auxiliary-reference overhead for exact compute accounting, and then (5) design the held-out dataset expansion before considering a learned latent/meta-configurator. Give me exact code/commands and keep the methodology publication-quality.