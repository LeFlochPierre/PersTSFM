# PersTSFM Phase 2.13 — Unified cross-task historical experiment matrix

**Purpose:** Run one reproducible matrix of `task × TSFM checkpoint × context × K` using already frozen, pre-official calibration inputs. No dataset-by-dataset Python or shell loops. Existing Phase 2.10 inference and historical scorer remain unchanged. Only add the scheduler and 9 regression tests.

**Important:** This is a *historical calibration development corpus*, **not** official fev-bench scoring. The scheduler does not import `fev`, access official evaluation targets, or open historical labels itself. The existing scoring command opens historical calibration labels only *after* forecasts are written. Official benchmark test windows remain unopened.

## Installation

From the repository root:

```bash
cd ~/RDS/home/projects/PersTSFM
conda activate PERTSFM_DATA_ENV
unzip -n PersTSFM_phase2_13_cross_task_runner.zip
python -m pytest -q tests/phase2/test_cross_task_matrix.py
```

## First run: one manifest across three independent sources

```bash
# DRY RUN by default. NO GPU inference, NO scoring.
python -m scripts.phase2.run_cross_task_matrix \
  --tasks M_DENSE_1H hospital_admissions_1D LOOP_SEATTLE_1H \
  --models chronos2 moirai2_small \
  --contexts 96 384 768 1536 \
  --k 3
```

The scheduler checks all frozen cases for identical K and context feasibility. It rejects irregular dates, missing histories, unsupported contexts and partial outputs instead of dropping entities without warning. It recognises already completed results. Task order is the requested order.

**Small controlled execution (two new configurations maximum):**

```bash
set -o pipefail
python -m scripts.phase2.run_cross_task_matrix \
  --tasks M_DENSE_1H hospital_admissions_1D LOOP_SEATTLE_1H \
  --models chronos2 moirai2_small \
  --contexts 96 384 768 1536 \
  --k 3 --quantile-policy rearrange \
  --execute --max-jobs 2 \
  2>&1 | tee results/phase2/phase2_13_first_execution.log
```

The **same command** can be rerun with `--max-jobs 5` or without it to finish; it verifies and skips already scored runs. It uses `conda run --no-capture-output -n` to launch each checkpoint in its required environment and scoring in `PERTSFM_DATA_ENV`. Keep your normal Conda installation available on PATH. There is **no overwrite option**.

## All six checkpoints, across all full exported tasks

```bash
# This is a planning command ONLY, and could generate a large matrix:
python -m scripts.phase2.run_cross_task_matrix \
  --models chronos2 timesfm3 tirex2 toto2_22m toto2_313m moirai2_small \
  --contexts 96 384 768 1536 3072 \
  --k 3 --quantile-policy rearrange
```

Without `--tasks`, the scheduler discovers **all existing full-cohort exports**, but refuses to invent or silently export new ones. Model-specific TiRex 2048-step context limit is enforced. Calendar-irregular and other incompatible datasets are marked `SKIP_INFEASIBLE`, without deleting them. The full output will be very expensive; inspect and throttle using `--max-jobs N` first. To recover forecasts that have not been scored, use `--execute --score-only` with the same model/context/task selection.

### Output

Each invocation creates a dated directory under:

`results/phase2/cross_task_matrix/<timestamp>_{plan|execute}/`

- `plan.csv`: every requested combination and its state (`DONE`, `PENDING`, `NEEDS_SCORE`, `SKIP_INFEASIBLE`, `BLOCKED`).
- `manifest.json`: input/registry checksums and scheduler settings.
- `logs/`: complete inference and scoring stdout/stderr per actual job.
- `executed_jobs.csv`, `final_plan.csv`: results of a permitted execution, including failures.
- `score_catalog.csv`: one row per completed model/context/task/K, including diagnostic MAE, three-quantile pinball and exploratory load/inference seconds.
- `case_catalog.csv`: one row per matched scored case with task, model, context, K, entity, historical lag and losses. Suitable for later budgeted replay design, **not for random entity/window train/test splitting**.

No score catalog is written if no completed configurations are in the plan.

### Safety and scientific limitations

- Pre-official calibration is strictly historical. **Do not use official task outcomes for algorithm selection.**
- The *same* case IDs are used for all configurations of a task/context/K; if a full cohort is not supported the job is skipped rather than quietly reducing its size.
- Compare scores only within a task at a common K and matched cohort. Raw MAEs are not comparable across dataset units. The three-quantile pinball diagnostic is **not** native fev SQL.
- The manifest's `family_id_exact` is merely an exact dataset identifier, not a verified independent source family; group across frequencies and overlapping benchmark collections before meta-training/test splits.
- `--quantile-policy rearrange` matches the policy used in your existing Chronos/Moirai historical runs; the unmodified point forecast is preserved. The scheduler blocks inconsistent existing quantile policies. Retain raw quantiles and correction metadata from each run.
- Timing is **exploratory**: separate Conda processes reload checkpoints; repeated warmup and GPU memory benchmarking are future work. Counting configurations is not equivalent to measuring GPU-seconds.
- Initial runner supports K=1..3, finite regular-cadence histories and **full** exported cohorts. K=0 priors, calendar-aware monthly/quarterly/yearly models, missing data, covariates/multivariate targets, native fev SQL and larger benchmark-family splits remain subsequent work.
- To avoid accidental expensive GPU campaigns, `--execute` is required and `--max-jobs 2` is recommended initially.
