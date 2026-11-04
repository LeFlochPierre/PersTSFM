# PersTSFM Direction A — Research and infrastructure reset

**Research memo (1 November 2026).** This is a proposal and implementation audit checklist, *not* a validated new method. Re-read the literature prior to any NeurIPS 2027 submission. The local RTX6 repository is the ground truth; the visible GitHub `main` branch does not yet contain all Phase 3 code.

## 1. Proposed thesis

**Working title:** *Forecasting Under a Computation Budget: Sequential Acquisition of Time-Series Foundation Model Predictions with Delayed Feedback.*

At each forecast origin `t`, a policy observes historical data and cheap diagnostics, chooses an affordable forecasting action, may purchase another model prediction, and eventually returns a forecast. The output label becomes observable only after the forecast horizon and may overlap with other outstanding forecasts. The policy must jointly consider accuracy, compute, feedback delay, and nonstationarity.

Research questions:

1. Is the *cost-constrained oracle gap* meaningfully large across unseen datasets, not just in isolated series?
2. Can **predecision** features and **observed, previously purchased** forecasts predict whether a second computation improves expected error enough to offset its cost?
3. Can a sequential stopping/selection policy outperform cheap or expensive single-model choices, static validation selection, TimeRouter-style selection, and standard contextual bandits at the same measured total compute?
4. Can we establish a new analysis or guarantee under delayed feedback, if and only if stronger existing theory does not already cover the chosen assumptions?

This is not a claim that bandit model selection, delayed feedback or TSFM routing are novel individually. A new contribution requires a well-defined *joint* problem/algorithm and defensible comparisons.

### Formalization

At origin `t`, history `H_t`, features `z_t` and available compute budget `B_t` are observed. An action `a` can run a model, increase context, acquire a probabilistic prediction, or stop. Each action incurs measured cost `c(a,t)` and reveals a prediction. The final prediction incurs loss `ell(y_t, prediction_t)` when outcome `y_t` becomes available (possibly delayed until `t+H`). The optimization can be framed as:

`min E[forecast loss + lambda * total cost]`, subject to per-origin or cumulative budget and a nonanticipating policy.

Cost may be GPU seconds, inference latency, energy, or memory; **do not mix them**. Begin with seconds and GPU-seconds, and report memory separately. Include model load time separately (cold-start vs warm resident cache).

## 2. Closest prior art to reproduce/check

| Work / reference | Closest overlap | Required control |
|---|---|---|
| [TimeRouter](https://arxiv.org/abs/2606.11625), [code](https://github.com/UConn-DSIS/TimeRouter), 2026 | Lightweight learned selection among frozen TSFMs with fallback | Routing, plus a cost-aware routing ablation; check repo dependencies and published setup |
| [PADRE, IJCAI 2026](https://ijcai.org/proceedings/2026/503) | Selective forecasting update decisions using **delayed** evidence of drift | Baseline or discussion for delayed-update setting; not identical to purchasing model predictions |
| [Impermanent](https://arxiv.org/abs/2603.08707), [code](https://github.com/TimeCopilot/impermanent) | Live, nonstationary, prequential forecasting | Truly chronological evaluation, where forecasts precede outcome availability |
| [FEV Bench](https://github.com/autogluon/fev), [tutorial](https://autogluon.github.io/fev/latest/tutorials/03-tasks-and-benchmarks/) | Forecast task and dataset contracts | Stable task registry, ground-truth isolation, covariate availability rules |
| [GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval) | Diverse probabilistic zero-shot benchmarks | External evaluation distribution/metrics, contamination policy |
| [Vowpal Wabbit contextual bandits](https://vowpalwabbit.org/tutorials/contextual_bandits.html) | Sequential action selection under partial observed losses | Epsilon-greedy, IPS/DR, contextual bandit baselines, logged action propensities |
| [MABWiser](https://fidelity.github.io/mabwiser/) | Simple contextual bandit algorithms | LinUCB, LinTS and random/greedy baseline implementations |
| [River ADWIN](https://github.com/online-ml/river/blob/main/river/drift/adwin.py) | Online change detection | Concept-drift diagnostic baseline only, avoid making drift detection itself the novelty |

Important novelty pressure: cost-sensitive bandits, active model evaluation and sequential acquisition exist broadly. A strong paper must demonstrate why TSFM heterogeneity, delayed multi-step outcomes, online changing regimes and compute costs require a genuinely different algorithm or formal guarantee.

## 3. Dataset foundation

### Task contracts / leakage controls

Use FEV as the primary **dataset contract**, not necessarily as the final experimental benchmark. Pin a specific version and per-dataset config/horizon/frequency/task hash. For each FEV task, `task.get_window(0).get_input_data()` exposes past observations and future *known* covariates, not future targets. The **official evaluation ground truth remains sealed** until the final preregistered evaluation. Generate development origins strictly inside past input data. The `get_ground_truth()` method must be prohibited in the development data adapter.

- `time_available`: input observation arrival time.
- `forecast_origin`: timestamp when prediction is made.
- `target_maturity`: earliest timestamp at which a scored forecast can be used to update policy (normally origin + H; handle partial feedback explicitly if choosing to support it).
- For entity grouping, do not leak overlapping patients/sensors/studies between train and held-out families.
- Treat origin overlap and shared entities as dependent observations. Bootstrap at dataset or entity (or time-block) level, not per forecast point.
- Inputs must obey covariate availability (known future calendars vs unknown future measurements).
- Mark data contamination/pretraining overlap per checkpoint/task; do not interpret contaminated runs as unbiased generalization.

### Pilot family inventory and realistic roles

| Development family | Datasets in current exploratory work or candidate registry | Use |
|---|---|---|
| Energy | M_DENSE_1H, LOOP_SEATTLE_1H, Electricity, solar, ercot_1H | Strong seasonality and context/scale heterogeneity |
| Traffic | pems_bay | Sensor heterogeneity; beware correlated sensor IDs |
| Wearable/activity | hupa_ucm | Person-specific dynamics; check series length and resolution |
| Clinical | vitaldb, physionet2019, hospital_admissions_1D | Short/irregular series; ensure valid tasks/horizon; split by patient |
| General | m4_hourly and other eligible M4 components | Classical baselines and different frequencies |
| Nonstationary streams | [Impermanent GitHub activity](https://github.com/TimeCopilot/impermanent) | Distinct drift-focused evaluation; preserve true temporal cutoffs |

These names are **candidate work items**, not a claim they are installed, all usable, or represent twelve independent datasets. Run an availability/feasibility audit before choosing a final cohort.

**Pilot:** 6–10 family/task combinations with >=8 rolling origins where feasible, rather than stretching one sensor series into pseudo-independent examples. **Final:** as many *independent dataset families* as practicably feasible (target >=15 datasets from several domains, not only 6 datasets and four tiny test folds). Keep some families untouched for final evaluation; one task from each family is not necessarily independent if the underlying data overlaps.

Pre-register horizons based on each benchmark's task and application; do not force 168-step horizons on short clinical series or compare MAE across unrelated target scales. Report within-task normalized skill (MASE/skill vs seasonal naive) and rank/aggregate by dataset. Keep raw MAE per task too.

## 4. Model portfolio and cost ladder

Use a deliberately **small, diverse** set before increasing model count. Existing PersTSFM pilot integrations: Chronos-2, Moirai2 Small, TimesFM3, TiRex2, Toto2 22M, Toto2 313M. Do not assume all are deployable in the same environment or all support the same context or exogenous covariates.

| Tier | Proposed initial action | Why |
|---|---|---|
| Zero/very cheap | last-value, seasonal naive, drift, historic mean | Essential stop-now action |
| Cheap trained | ridge lag regression / ETS / AutoTheta (task-specific) | Strong non-TSFM alternative; count fitting cost |
| Small TSFM | Chronos-Bolt tiny (9M), Chronos-2-small (28M), Toto2 22M | Tests efficient early prediction |
| Medium TSFM | Chronos-2 120M, Moirai2 Small, TiRex2 | Diverse architectural inductive biases |
| Heavy TSFM | Toto2 313M, TimesFM3 | Higher-cost expert; validate true measured cost vs medium |
| Optional adaptation | Audited Chronos-2 LoRA 100/500 steps | Expensive action in a later extension, **not** mandatory for first paper feasibility |

Official sources: [Chronos model sizes](https://github.com/amazon-science/chronos-forecasting), [Toto 2 models](https://www.datadoghq.com/blog/ai/toto-2/), [TimesFM3](https://github.com/google-research/timesfm), [TimesFM3 model card](https://huggingface.co/google/timesfm-3.0-pytorch). **TimesFM3 downloadable weights are non-commercial**; document licensing carefully. All weights and code must be pinned to a commit/revision, including tokenizer/preprocessing and evaluation libraries.

Do not call a model `cheap` based on parameter count alone: benchmark real end-to-end latency at correct batch/horizon/context under warm inference and cold-loading conditions. Cache cost must be charged to policy whenever a new expert is instantiated; policies comparing preloaded vs on-demand models must operate under equal memory rules.

Start with **4–6 actions including cheap baselines**, not 30 model-context variants. Introduce contexts and LoRA only after demonstrating a positive compute-selection feasibility gap.

## 5. Tooling: minimum stack

**Must have:** `fev` (pinned), `numpy`, `pandas`, `pyarrow`/Parquet (large result store), `torch` + installed model wrappers, `scikit-learn`, `statsforecast`, `pytest`, Git. Add a deterministic YAML/JSON config loader or use existing PersTSFM config patterns. No need to install another experiment framework just to configure the pilot.

**Strongly useful:** `river` for streaming features/drift baselines, `mabwiser` for LinUCB/LinTS reference policies, `optuna` if model/hyperparameter searches are subsequently required, `duckdb` to query large per-forecast Parquet tables, simple GPU timing/profiling via torch CUDA events and `time.perf_counter` (after synchronization). `Vowpal Wabbit` is an advanced later baseline for logged contextual bandits, not necessary for first smoke run.

**Avoid initially:** Ray/Ray Tune, Kubernetes/serving stacks, complex simulation frameworks, full LoRA sweeps on all models, and changing every dataset's horizon to make an existing script work. Adds more compatibility burden than scientific signal.

## 6. Three experiment gates

### Gate 0: capability and data audit (no GPU tuning)

- Pin model/environment versions and checkpoint revisions (include SHA/revision in artifacts).
- Inventory datasets with length, frequency, IDs, missingness, target scale, origins, horizon, pretraining-overlap status, and whether target labels can be guaranteed absent before origin.
- Build an `ActionSpec` table with model, context, horizon, cost semantics, cold/warm load status, supported covariates, batch support, GPU/CPU, license, and whether trained parameters update.
- Check model parity with existing Phase 2 pilot at fixed inputs, and verify Chronos-2's post-fine-tune mixed-precision cleanup when including LoRA.

### Gate 1: **cost-constrained oracle gap** (offline acquisition table)

For every development origin, precompute *candidate expert forecasts and measured inference costs*, retaining per-origin and per-entity predictions, **not just aggregated loss**. Compute an unreachable cost-constrained hindsight oracle as **headroom only**. Also compute best constant action, validation-selected action, cheapest action, best feasible static mixture, and a stratified simple feature-based rule.

Report Pareto frontiers: dataset-normalized loss vs inference GPU-seconds and latency, with confidence intervals grouped by dataset/entity. Include cold vs warm cost, and expert-residency constraints. If the oracle gap is tiny or temporally unpredictable, **stop** (do not invent a complex algorithm).

### Gate 2: **causal replay of sequential selection**

Replay predictions at historical origins in chronological order. Even though the full forecast matrix was precomputed offline, the simulated policy must reveal **only predictions for actions it actually purchases**. Never expose outcome losses of unqueried actions; outcomes for purchased actions unlock strictly at `target_maturity`. Maintain separate wall-clock compute charging for replay vs actual inference; use measured per-action costs as policy costs. Log every request, cost, reward reveal, and policy update.

Baselines: always-cheapest, always-best-development, best static budget-constrained action, random/epsilon-greedy, LinUCB/LinTS, cost-aware supervised router, optional TimeRouter-style learned static routing, delayed-feedback selector, and explicit oracle (unattainable upper bound). Honest uncertainty analysis and held-out family splits required.

### Gate 3: method (only after positive headroom and predictability)

Potential algorithm: uncertainty-aware **marginal value-of-compute** policy with calibrated upper/lower confidence bounds for expected *conditional gain* of the next expert, online updates when delayed labels arrive, and a budget-aware stopping criterion. Could use simple incremental estimates, context-dependent distributions, or drift-aware reset; test novelty against pre-existing sequential selection/bandit theory. Output should be both a forecast and action log trace.

Success criterion isn't a particular percentage: demonstrate a statistically supported superior *error–cost trade-off* across unseen domains, with fair cost normalization, ablations, and strong baselines. Pre-specify a target like reducing measured cost 30–50% with <=1% normalized accuracy loss as a motivating bar, not a guaranteed result.

## 7. Repository consolidation plan

**Do not rename or delete existing modules to match a new research idea.** Existing Phase 2/3 code and results are valuable provenance. First preserve them; then create a small new decision framework alongside the old experiment code:

```
PersTSFM/
  README.md
  environment.yml
  configs/
    phase2/ phase3/
    direction_a/                     # new research configs
  src/
    data/ evaluation/ models/ analysis/ experiments/
    decision/                        # NEW, only after Gate 1
      action_schema.py
      replay.py
      feedback_queue.py
      cost_model.py
      policies/
  scripts/
    phase2/ phase3/ maintenance/
    direction_a/                     # NEW: inventory, oracle gap, replay
  tests/
    phase2/ phase3/
    direction_a/                     # leakage and sequential policy tests
  docs/
    research_direction_a.md
    dataset_registry.md
    model_capabilities.md
  data/                               # remain local; no move
  results/                            # remain local; no move
  external/                           # audit before changes
  notes/ references/                  # leave as they are
  archive/
    patches/phase3/                   # root ZIPs only
    triage/                           # accidental root filename
    maintenance/                      # move manifests
```

**Current local tree risks:** scattered `PersTSFM_*.zip` archives, suspicious `tatus --short`, older `src/experiments/*.bak`, unknown `pertsfm_phase2_benchmark`, and `.gitignore` rules that must be examined before staging. `src/analysis` contains potentially useful historic synthesis code: do not move it merely because naming is inconsistent. The visible GitHub `main` tree is older than RTX6's local code and should not be used as a substitute for a `git status`/`git ls-files` local audit.

**Safe housekeeping:** the bundled script defaults to dry run, moves only **untracked root-level ZIPs** and the odd root-level typo *when it is an untracked regular file*, uses checksums, creates a manifest, and supports restoration. It deliberately does not touch data, results, source, caches, Git metadata or unknown directories.

Suggested local Git workflow:

```
cd ~/RDS/home/projects/PersTSFM
conda activate PERTSFM_ENV
python -m pytest -q tests/phase2 tests/phase3

git status --short
git branch --show-current
git ls-files | wc -l
python scripts/maintenance/housekeep_direction_a.py      # inspect only
python scripts/maintenance/housekeep_direction_a.py --apply
# review manifest and git status; then commit on a new research branch
```

**Do not commit datasets, patient records, model weights, generated caches, or huge logs.** If `.gitignore` isn't appropriate, update it after reviewing exact tracked files and the risk of inadvertently staging sensitive or licensed data. Avoid unconditional `git clean -fdx` and `rm -rf`.

## 8. Immediate deliverables and decision gates

1. Local repository audit: full inventory, Git branch/status, regression test health, exact dataset storage and model availability; safe root cleanup.
2. A **versioned dataset registry** with 10–20 feasible task candidates, supported origins and independent family groups.
3. A **model capability/cost table** for a small portfolio, with explicit warm/cold timing on RTX6.
4. A **blind oracle-gap feasibility table** on development-only data: cheap/static/oracle loss vs compute. No model tuning yet.
5. Decision: proceed to the online method only if the gap is large and demonstrably predictable from information available at decision time. If not, pivot early.

**What not to do:** do not launch Phase 3.7A LoRA atlas yet; it is a side-branch, not the foundation of Direction A.
