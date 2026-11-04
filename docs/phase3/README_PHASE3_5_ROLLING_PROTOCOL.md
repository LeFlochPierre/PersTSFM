# PersTSFM Phase 3.5 — Rolling-origin chronological evaluation and Chronos-2 sampler audit

**Goal:** Make the Phase 3.4 Chronos-2 context/LoRA benchmark causally sound across multiple historical origins, without replacing any working Phase 3.4 code or accessing official FEV targets.

## Architecture

- `rolling.py`: source-snapshot → oldest-to-newest historical prefix snapshots, same predeclared cohort across folds, disjoint lag-1 evaluation windows, provenance hashes, frozen suite manifest, safe resume and all-fold lock before scoring.
- `sampler_audit.py`: confirm complete pre-validation training pools; compute eligible windows; optionally **trace actual** Chronos-2 `np.random.randint` training crop cutoffs (opt-in, installed-source-code guarded).
- `uncertainty.py`: paired entity-bootstrap confidence intervals for candidate-vs-comparator gains (descriptive within-dataset only; sensors and adjacent folds are correlated).
- `reporting.py`: per-fold locked-strategy later MAE, validation-chosen zero-shot comparator, hindsight oracle diagnostic and fold summary.
- `scripts/phase3/rolling_adaptation.py`: thin, single CLI interface; core modules reusable by other code.
- `configs/phase3/*_rolling_{smoke,pilot}.json`: versioned experimental configurations.

## Important protocol definition

For each *fold*, use a new frozen **prefix** of the original preofficial history. Its last `2H` observations are deliberately withheld from LoRA training:

```
← past ... [TRAIN: through t-2H-1] [VALIDATE: lag2, H] [EVALUATE: lag1, H] →
```

Configuration selection uses lag2 only. All fold selections must be written and checksummed before any lag1 scores are revealed. The next (newer) fold may train on older holdout observations, exactly as a rolling refit would. A separate frozen-adapter-across-origins experiment remains future work and **is not equivalent**.

- Folds are ordered oldest → newest. With `stride_horizons=1`, lag1 evaluation blocks do **not** overlap (they are adjacent).
- Within each fold, different strategies share exactly the same entity cohort and horizon. Entities must also be feasible in **every** fold; global `invalid_policy=drop` deterministically intersects compatible IDs.
- End timestamps must align across entities by default. If not, align the dataset first; disabling this guard is only appropriate when asynchronous entities are an intentional design choice.
- Missing observations and too-short context/history combinations cause fail-fast QC or deterministic non-performance-based exclusion.
- The original snapshot contains only **FEV preofficial inputs**; no `get_test_data()` call or official FEV test label access. Historical lag1 is an *internal* chronological holdout, **not** the official FEV test.
- Features and model training use only each fold's prefix before lag2; selection cannot see lag1 targets. Saving forecasts on lag1 is allowed; scoring requires an explicit frozen-suite action.
- **M-DENSE and LOOP Seattle have already been explored and are development datasets.** Do not reuse them as untouched held-out dataset families for learning a predictive policy.

## Install and test

From the PersTSFM repository root:

```bash
unzip -n PersTSFM_phase3_5_rolling_protocol.zip
conda activate PERTSFM_ENV
python -m pytest -q tests/phase3/test_rolling_adaptation.py
python -m pytest -q tests/phase2 tests/phase3
```

All files in this ZIP are *new* and have no expected filename overlap with Phase 3.4. The only GPU dependency is in the opt-in `run` command. The config expects your existing corrected LOOP snapshot `results/phase3/context_saturation_inputs/loop_preofficial_full_ns_v2.npz`.

### Stage A: smoke-test the protocol (4 entities × 3 folds)

```bash
python -m scripts.phase3.rolling_adaptation plan \
    --config configs/phase3/adaptation_loop_rolling_smoke.json

python -m scripts.phase3.rolling_adaptation init \
    --config configs/phase3/adaptation_loop_rolling_smoke.json \
    --out results/phase3/rolling/loop_smoke_v1

# Provenance audit: eligible windows, without training.
python -m scripts.phase3.rolling_adaptation audit-pools \
    --suite results/phase3/rolling/loop_smoke_v1 \
    --fold 0 --context 768 --history full

set -o pipefail
python -m scripts.phase3.rolling_adaptation run \
    --suite results/phase3/rolling/loop_smoke_v1 \
    --trace-sampler \
    2>&1 | tee results/phase3/rolling_loop_smoke_v1.log

# Only after ALL folds have locked validation-selected strategies:
python -m scripts.phase3.rolling_adaptation score \
    --suite results/phase3/rolling/loop_smoke_v1

python -m scripts.phase3.rolling_adaptation report \
    --suite results/phase3/rolling/loop_smoke_v1
```

**GPU caveat:** actual Chronos-2 `--trace-sampler` requires the installed package's `_construct_slice()` implementation to contain the guarded random cutoff expression `np.random.randint(self.min_past, full_length - self.prediction_length + 1)`. If the upstream implementation changes, tracing refuses to proceed. First run one fold (`run --fold 0 --trace-sampler`) to validate package compatibility. You can run without `--trace-sampler` if your installed version is incompatible, but in that case **do not claim measured sampling coverage**.

A fold's LoRA training output will contain `sampler_draws.csv` and `sampler_trace_summary.json` in its saved adapter directory when tracing is enabled. The trial metadata still tracks normal Phase 3.4 predictions. The global NumPy random function is patched transiently to forward calls unchanged and is restored in a `finally` block; this is safe only for a single-process training invocation without other threads concurrently drawing from NumPy's global RNG.

### Stage B: full LOOP development pilot (323 compatible entities × 4 folds)

```bash
python -m scripts.phase3.rolling_adaptation plan \
    --config configs/phase3/adaptation_loop_rolling_pilot.json

python -m scripts.phase3.rolling_adaptation init \
    --config configs/phase3/adaptation_loop_rolling_pilot.json \
    --out results/phase3/rolling/loop_pilot_v1

python -m scripts.phase3.rolling_adaptation run \
    --suite results/phase3/rolling/loop_pilot_v1 \
    2>&1 | tee results/phase3/rolling_loop_pilot_v1.log

python -m scripts.phase3.rolling_adaptation score \
    --suite results/phase3/rolling/loop_pilot_v1
```

Pilot configuration: zero-shot contexts 384/768/1536/3072/4096; LoRA fixed context 768 and steps 20/100 with seeds 42/43/44; 4 chronological folds. This is intentionally cheaper than a maximal context×adaptation×history grid. **Do not start the full pilot before the CPU tests and the small GPU smoke pass.**

Run individual folds with `run --suite ... --fold 0`, `--fold 1`, etc. After an interruption, repeat the same command: the existing Phase 3.4 trial-cache will resume completed trials and reject inconsistent manifests. `run` refuses any new training after the suite has been frozen for later scoring. `score` is idempotent on completed folds, and refuses scoring unless *all* fold selection locks exist.

## Scientific interpretation and limitations

1. The reported `validation_mae` chooses a configuration **within each rolling fold**; the `later_mae` from the next chronological block evaluates that selection. These are historical backtests, not the official benchmark test.
2. LoRA seeds are averaged before selecting the strategy, and later strategy performance averages the same seed set. Avoid selecting a lucky seed.
3. A best-possible later strategy is printed as a *hindsight-only diagnostic* and cannot be fed back into selection.
4. The paired bootstrap resamples individual sensors across one fold; this is descriptive uncertainty, not a confidence interval over independent datasets. Strong conclusions need independent families, multiple seeds and temporally blocked analyses.
5. The rollout currently **re-fits LoRA at each origin** and compares all candidates. Its measured cost is thus full-sweep cost. A future budget-aware policy must compare its *actual decision cost*, including search and training, to baselines on held-out families.
6. Chronos-2's default sampler draws target cutoffs randomly between `min_past` (usually H) and `pool_length-H`, then truncates the input context to at most C. Hence the number of *fully C-length* eligible windows differs from the number of all possible training cutoffs. The optional tracer captures actual target cutoffs and entity indices and does not infer actual draw distribution from just the pool length.
7. We have not audited the internal sampler of any non-Chronos-2 model; only Chronos-2 LoRA is implemented here.

**Source for current Chronos code structure:** `amazon-science/chronos-forecasting`, `src/chronos/chronos2/dataset.py` (`_construct_slice`, `_generate_train_batches`) and `pipeline.py` (`fit`). Audit and record your installed source hash before scientific claims about training sample distributions. Never assume the GitHub default branch matches your installed package version.

## Future extension points

- `OriginPolicy` object for frozen-adapter versus rolling-refit comparisons;
- time-blocked uncertainty estimation across rolling origins;
- instrumented sampler trace coverage metrics, stratified by history quartile;
- cheaper residual bias/scale calibration baseline;
- additional model backends;
- final independent-family test split and official FEV benchmark after methodology freeze.
