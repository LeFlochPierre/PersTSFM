# PersTSFM — Research question, evidence and next steps

**Research handover — 27 October 2026**  
**Status:** early-stage experimental programme; the empirical dataset and the proposed learned algorithm are not yet validated across unseen source families.

## 1. The research question

**Main question**

> Given a new forecasting dataset, limited historical observations and a finite computation budget, can we determine **which pretrained time-series foundation model to use, how much context it should receive, whether adapting it is worthwhile, how much training history and adaptation compute to allocate, and when to stop**—without exhaustively evaluating every combination?

The project started from an individual-personalisation question: *At what point does a previously unseen entity's personal history make adaptation preferable to a population/pretrained model?* Our recent work broadens the decision to a hierarchy of dataset, model and entity choices. We should preserve the personalisation objective while distinguishing **dataset-level adaptation** from **individual-level adaptation**, which has not yet shown reliable later-period gains.

Define:

- **Model** `m`: pretrained TSFM checkpoint / architecture.
- **Inference context** `C`: observations passed into **one forecast**. Longer context need not improve quality and may increase latency.
- **Adaptation history** `T`: observations available **to sample training windows** prior to the training cutoff. This may be far larger than `C`.
- **Adaptation budget** `S`: LoRA/full fine-tuning steps or actual GPU seconds (including zero-shot `S=0`).
- **Dataset signature** `φ(D)`: pre-evaluation descriptors (frequency, length, seasonality, autocorrelation, shifts, noise, missingness, heterogeneity, etc.).
- **Budget** `B`: compute available for probing and adaptation; for repeated deployment, also account for inference costs.

A potential learned decision system predicts losses and uncertainties over `m,C,T,S`, uses cheap sequential observations to update them, allocates compute, and stops when additional information or adaptation is not worth its expense.

**Critical distinction:** An extensive empirical benchmark showing adaptation sometimes fails is useful; it does **not** itself demonstrate an accurate or novel decision policy. "Good hindsight performance" is not the same as an actionable forward policy.

## 2. Two possible contributions

### A. Standalone empirical study

**Working title:** *When Should Time-Series Foundation Models Adapt? A Cross-Model Study of Context Saturation, Training History and Fine-Tuning Budgets.*

Questions:

1. How common is improvement or degradation after adaptation across independent datasets and TSFM architectures?
2. How do zero-shot context saturation, adaptation-data amount and adaptation steps interact?
3. Are negative/positive adaptation gains stable across seeds, training windows and later periods?
4. Which pre-training-cutoff dataset properties correlate with adaptation benefit, with proper source-family grouping and uncertainty?
5. How much GPU time, memory and model-loading overhead does adaptation consume relative to zero-shot defaults or simple residual calibration?

**Contribution threshold:** A reproducible benchmark and robust, previously unreported cross-model phenomena—not just a large collection of scores. Compare prior TSFM adaptation and context-saturation work directly before claiming novelty.

### B. Methodological paper (NeurIPS/ICML ambition)

**Working title:** *Budget-Aware Adaptation of Time-Series Foundation Models via Learned Response Curves.*

Train a transferable uncertainty-aware predictor or sequential policy that can choose `S=0`, select `m,C,T` and adaptively spend compute. Compare against fixed zero-shot, fixed context, fixed LoRA schedules, no-probe priors, simple selection heuristics, successive halving/Hyperband, and cost-aware optimisation. **Primary evidence:** lower *later-period loss at matched total GPU cost* on **held-out independent source families**. Algorithmic novelty must be established against existing multi-fidelity/meta-learning work.

The empirical study can be a legitimate outcome even if no learned policy succeeds. Do not assume the two automatically make separable papers; avoid double-counting contributions.

## 3. Evaluation protocol and leakage boundary

- Benchmark: FEV v0.10.0; distinct datasets/configurations can share underlying source families. Group by **independent original source family** for held-out evaluation.
- Never read official FEV test labels to select configurations, tune hyperparameters or develop policies.
- Historical pre-official replay: **training** on observations strictly before lag-2 target period; **select** adaptation/context on lag-2; freeze; **evaluate** on later lag-1 target period. Earlier Phase-2 `K=3` experiments sometimes select using lags 3+2, evaluate lag-1; make this distinction explicit.
- Final scientific conclusions require additional chronological origins and family-disjoint held-out tests. Our existing two-origin replay remains exploratory.
- Primary metrics: MAE and ultimately benchmark-compatible probabilistic scoring (official FEV SQL uses nine quantile levels); `mean_pinball_3` is diagnostic only. Monitor crossing quantiles, effective model context, entity coverage and seeds.
- Costs: measured *model loading + fine-tune compute + validation inference + deployment inference*. Forecast-call count is not equivalent to actual GPU time; for LoRA measure continuation cost rather than assuming independently restarted runs can be combined.

## 4. What we have actually established

### Phase 2 — zero-shot search and time-aware replay

Six pretrained checkpoints tested: Chronos-2, TimesFM3, TiRex-2, Toto 2 22M, Toto 2 313M, Moirai 2 Small. An initial context grid (96, 384, 768, 1536, 3072, subject to feasibility) generated **76 completed K=3 configurations** across LOOP Seattle, M-DENSE and Hospital Admissions. An ERCOT K=2 corpus adds 18 configurations at 96/192/384; longer continuous windows fail strict finite-input constraints for this fully matched cohort.

Selected configurations vary. LOOP Seattle often favours long contexts, M-DENSE intermediate contexts; ERCOT has strong chronological instability. ERCOT lag2-versus-lag1 MAE configuration Spearman correlation was **−0.426**; some regrets were driven heavily by COAST/SOUTH_C. These are signals for *uncertainty-aware evaluation*, not proof that our planned method solves it.

### Phase 3.0–3.1 — value of cheap probes

- Small entity-subsample probes reduce *forecast calls* but frequently degrade future selection quality.
- A simple six-model longest-only baseline beat our first `adaptive_top2` heuristic on all four families in the four-entity lag-2 experiment; our method also spent more forecast calls.
- Historical entity-specific configuration selection usually harmed later performance despite hindsight headroom. Entity personalisation remains **unproven**.

### Phase 3.2 — GPU cost and Pareto analysis

On M-DENSE, batch 8 and 16 entities, 17 feasible checkpoint/context combinations completed. At context 768: Toto 22M ~34.9 ms/16 forecasts, Moirai Small ~51.4 ms, Chronos-2 ~68.8 ms, TimesFM3 ~210.8 ms, for this measured setup. Loading costs were seconds. Warm inference can be fast enough that searching multiple checkpoints is not worthwhile for one-off predictions; repeated deployment has different economics.

Illustrative *hindsight* M-DENSE MAE/latency Pareto choices: Toto 22M@3072 (MAE 63.295, 34.8 ms), Moirai Small@768 (MAE 60.482, 51.4 ms), Chronos-2@768 (MAE 57.720, 68.8 ms). This is not a learned selection result.

### Phase 3.3A — Chronos-2 LoRA, M-DENSE, limited history

30 entities; inference context `C=768`, training pool `T=3072` pre-lag2 observations/entity, batch size 8, learning rate `1e-5`, seed 42. Independent restarts for budgets `0/20/100/500` steps:

| Steps | Lag2 validation MAE | Lag1 later MAE | Training wall seconds |
|---:|---:|---:|---:|
| 0 (zero-shot) | **53.89950** | **54.94869** | 0 |
| 20 | 56.17707 | 55.99026 | 4.24 |
| 100 | 55.64884 | 56.15802 | 10.62 |
| 500 | 56.37062 | 57.03708 | 45.12 |

The locked validation choice was zero-shot, and the lag1 hindsight winner was also zero-shot. This does **not** show no possible fine-tuning recipe could work. Adapted quantiles exhibited crossings; monitor uncertainty/calibration.

### Phase 3.3B — full pre-official history audit

| Family | Pre-lag2 training observations/entity | Maximum tested fully finite input context |
|---|---:|---:|
| M-DENSE 1H | 15,504 | 8,192 |
| LOOP Seattle 1H | 6,744 | 6,144 |
| Hospital Admissions 1D | ~1,113–1,115 | 768 |
| ERCOT 1H | 151,176 | 384 (no missing-value imputation) |

M-DENSE and LOOP have much more train history available than the earlier exported 3,072-step rolling buffers. Chronos-2 max input length 8,192 is a per-example context limit, **not** a cap on the adaptation-history training pool.

### Phase 3.3C — full-history Chronos-2 inference-context sweep, M-DENSE

| Context | Lag2 MAE | Lag1 MAE |
|---:|---:|---:|
| 96 | 71.95639 | 68.76970 |
| 384 | 56.76485 | 58.80250 |
| 768 | 53.89950 | **54.94869** |
| 1536 | **53.51631** | 57.35734 |
| 3072 | 53.69068 | 57.33118 |
| 4096 | 54.04133 | 56.85738 |
| 6144 | 54.35178 | 56.06186 |
| 8192 | 55.03536 | 55.91012 |

Validation selected 1,536 but later optimal was 768: **4.38% later regret** from the lag2-selected context. More context is not monotonically better; the exact minimum is not yet reliable across periods. Note the full-history snapshot was corrected for pandas microsecond-versus-nanosecond timestamps; use `_ns_v2.npz`, not the original invalid snapshot.

### Phase 3.3D — history-pool ablation, fixed context 768, M-DENSE

We varied adaptation training pool 3,072/6,144/15,504 observations per entity with steps 20/100/500, same LR and seed. Lag-2 MAEs were nearly invariant to the training pool (e.g., at 100 steps **55.64884 / 55.64912 / 55.63222**, respectively); every LoRA model lost to zero-shot (**53.89950**). Selection locked to zero-shot.

**Outstanding:** lag1 `--score-later` for Phase 3.3D has **not been provided** as of this handover. Do not claim later outcomes for the full-history ablation without that output. Also audit actual Chronos training-window sampling and unique origins before interpreting pool-length invariance as a biological or statistical effect.

## 5. Immediate next scientific actions

1. Complete locked lag1 replay for Phase 3.3D, preserving model selection from lag2.
2. Audit Chronos-2 sampling from the whole training pool: time-index distribution, recency bias, observed windows, unique-window coverage, and batch/seed behaviour; verify effective context length.
3. Generalise the **working single-model LoRA pipeline** to 6–10 genuinely independent dataset source families first, with safe per-task horizons/context feasibility and *strict missingness handling*. Do not assume all six zero-shot checkpoints are trainable; officially supported fine-tuning paths and parity must be checked individually.
4. Add **zero-shot, simple residual correction**, 20/100/500-step LoRA, appropriate fixed context, and modest per-family context sweeps. Collect multiple seeds and chronological replay windows; freeze evaluation sets.
5. Expand to a second independently trainable TSFM and subsequently a larger family-disjoint development/test corpus (aim for approximately 20–40 independent source families if feasible; this is a design goal, not a dataset guarantee).
6. Investigate whether pre-cutoff features predict **adaptation gain**, **context saturation** and **the value of incremental training**, comparing against simple seasonal/context heuristics and stronger meta-optimisation baselines. Do not fit feature predictors from just four source families.

**Decision gate:** Proceed to a complex learned active selector only if cheap signals predict cross-family adaptation returns better than simple priors and search baselines on held-out source families. Otherwise, prioritise a strong empirical/negative-results study.

## 6. Repository / environment handover

- Keep `results/` and `data/` intact; they contain frozen scientific outputs and potentially large non-Git assets.
- Keep research scripts and existing tests intact until import and CLI references can be migrated together.
- Active newer Phase 3 entry points already live in `scripts/phase3/` (`context_saturation.py`, `lora_history_saturation.py`). Older Phase 2/3 tools still exist in the repository root; several tests reference these by exact basename, so moving them casually can break imports.
- Keep the old cleanup manifest `archive/manifests/cleanup_20261027T180152554261Z.json`. The second cleanup creates an **independent** reversible manifest.
- Root-level archive ZIPs/patch scripts are packaging/install artifacts, not experiment results; move these conservatively without deletion.
- A broad `git add -A` would stage huge untracked `results/` and `archive/` trees; do not do this until `.gitignore` and a deliberate tracking policy are in place.
- Distinct envs: `PERTSFM_DATA_ENV` for FEV data; `PERTSFM_ENV` for Chronos-2 and PEFT. The Hugging Face dependency fix involved `transformers==4.57.6`, `huggingface-hub==0.36.2`, and `datasets==3.6.0` at that time; capture a current environment snapshot, not a guessed frozen package list.

## 7. Proposed near-term paper narrative

**Empirical paper:** "Choosing Not to Adapt": systematic, chronology-controlled study showing the joint effect of model, dataset structure, context saturation, available training history and LoRA budget on future forecasting and GPU cost. Important negative findings are valid if broadly reproducible and compared against prior work.

**Algorithm paper (conditional):** An uncertainty-aware policy choosing zero-shot vs adaptation and allocating model/context/history/steps under budget. Evaluate cross-source generalisation, later-period regret, and real end-to-end GPU cost. **Neither paper currently has sufficient evidence for a top-tier acceptance claim.**
