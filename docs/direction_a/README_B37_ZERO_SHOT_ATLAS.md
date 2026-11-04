# PersTSFM B3.7 — zero-shot model × context × dataset baseline atlas

## Scope

Reuses B3 frozen cohorts and adapters; no training, no new source protocols, no new forecasting horizons. Produces a reproducible **matched-origin** comparison of 6 TSFM checkpoints, plus last-value and seasonal-naive baselines, across 4 registered context lengths in each of the three *already frozen* datasets.

**Not yet FEV.** FEV tasks, scoring rules, covariate handling, horizon configuration and leakage restrictions are different. B3.8 will use FEV's own Task / EvaluationWindow / Benchmark contracts and evaluate comparable **univariate zero-shot** slices before attempting multivariate/covariate tasks. Do not claim FEV scores from this B3.7 grid.

## Install

From repository root:

```bash
unzip -n PersTSFM_B37_zero_shot_matrix.zip
python -m pytest -q tests/direction_a
python -m pytest -q
```

## Audit and plan (read-only with respect to the repository data)

```bash
conda activate PERTSFM_ENV
python -m src.budgeted_forecasting.b37_suite inspect \
  --config configs/direction_a/b37_matrix.json
python -m src.budgeted_forecasting.b37_suite plan \
  --config configs/direction_a/b37_matrix.json \
  --output-dir results/direction_a/b37_plan
```

`inspect` reads dataset registry and protocol metadata only. `plan` writes an audit JSON, per-action manifest CSV, and shell commands. Before marking an existing output complete, it validates its model/context identity, frozen cohort hash, shard hash, exact cases, truth hashes, split, and costs. The planner refuses to overwrite an existing plan directory.

The config intentionally sets `active: false` for HUPA-UCM, VitalDB, PhysioNet 2019 pending protocol audit. They are included in inspection, not in automatic GPU workloads. **VitalDB and PhysioNet time axes in the existing registry are relative indices; a shared chronology across patients is not a real-world date split.** Audit the split before freezing.

## Preflight the two not-yet-tested checkpoints

The configured TimesFM3 environment is provisional; verify the module and exact checkpoint/API first. Toto 313M uses the already configured Toto environment.

```bash
conda run -n PERTSFM_ENV python -c "import timesfm3; print('PASS timesfm3')"
conda run -n PERTSFM_TOTO_ENV python -c "from src.models.toto2_adapter import Toto2ForecastModel; print('PASS Toto adapter import')"
```

For **solar** (112 cases), first test these actions with `--dry-run`, then execute without `--dry-run` after checking dependency/VRAM requirements:

```bash
OUT=results/direction_a/b35_solar_28
python -m src.budgeted_forecasting.capability_atlas run-action \
  --cohort "$OUT/cohort.json" --model timesfm3 --context-hours 96 \
  --output "$OUT/shards/timesfm3_96h.csv" --dry-run
python -m src.budgeted_forecasting.capability_atlas run-action \
  --cohort "$OUT/cohort.json" --model toto2_313m --context-hours 96 \
  --output "$OUT/shards/toto2_313m_96h.csv" --dry-run

# Use the correct env for each real model:
conda run --no-capture-output -n PERTSFM_ENV python -m src.budgeted_forecasting.capability_atlas run-action \
  --cohort "$OUT/cohort.json" --model timesfm3 --context-hours 96 \
  --output "$OUT/shards/timesfm3_96h.csv"
conda run --no-capture-output -n PERTSFM_TOTO_ENV python -m src.budgeted_forecasting.capability_atlas run-action \
  --cohort "$OUT/cohort.json" --model toto2_313m --context-hours 96 \
  --output "$OUT/shards/toto2_313m_96h.csv"
```

## Complete missing actions and report

Inspect `results/direction_a/b37_plan/actions.csv`, then:

```bash
bash results/direction_a/b37_plan/run_missing.sh
bash results/direction_a/b37_plan/build_reports.sh
```

No command silently overwrites a shard or report. Missing actions are created in their **existing** dataset shard directories; completed shards are verified and skipped. To regenerate a particular report, choose a new report directory, do not overwrite an old report.

**Do not interpret evaluation-hindsight winner as an actionable selection policy.** For model/context comparison, report entity-macro MAE, per-origin and per-entity spreads, and calibration-selected action on evaluation. Prefer scale-free relative scores *within datasets*; do not pool raw MAEs.

## Design / caveats

- six TSFMs: Chronos-2, TiRex-2, Moirai2 Small, Toto 2 22M, Toto 2 313M, TimesFM3
- four context lengths per existing dataset (electricity 3/4/7/14d; traffic 1/3/5/7d; solar 6/24/48/96h). Contexts are in the dataset registry.
- two classical baseline actions when eligible: last-value and seasonal-naive; seasonal-naive is omitted when the frozen maximum context is shorter than a full day
- solar uses a 28-entity pooled-cutoff per-entity-origin cohort, **not** the previously used 8-entity cohort
- the three development datasets use distinct horizons (electricity 24h, traffic 1h, solar 1h); the results are per-task, not horizon-matched
- TiRex-2's 2048-point context ceiling is enforced by the planner; other checkpoint limits must be verified with adapter/model tests
- present TSFM adapters run univariate inputs without covariates; FEV multivariate/covariate tasks require new controlled adapters and metrics
- in the registry, train/test chronological eligibility and availability of labels must be audited before activating HUPA, VitalDB, or PhysioNet
- different environments mean absolute warm-runtime comparisons are only exploratory
- this phase only measures **zero-shot** capability. B4 adds training later.
