# PersTSFM Phase 2.7 — Official fev-bench first-window historical feasibility

Additive patch; does not touch Phase 2.6, existing model adapters, original pilots, or held-out forecasts.

## Workflow

1. `conda activate PERTSFM_DATA_ENV`
2. `python -m scripts.phase2.audit_fev_history_feasibility --list-candidates`
3. `python -m scripts.phase2.audit_fev_history_feasibility --preset starter 2>&1 | tee results/phase2/benchmark_feasibility_starter.log`
4. Optional: `python -m scripts.phase2.audit_fev_history_feasibility --configs ercot_1H LOOP_SEATTLE_1H`

This downloads/loads source time-series datasets, which may be large. It accesses only the prediction-time past of the first official benchmark window; it never loads future ground truth and never scores official benchmark outcomes. The starter preset audits six diverse target-only tasks.

Outputs: `results/phase2/benchmark_feasibility/fev_bench_v0_10_0/`:

- `task_profiles/<task_id>.json` — individually saved, resumable input-only profiles
- `task_history_profiles.csv` — inspected population, history sizes, horizon
- `calibration_context_feasibility.csv` — feasibility at first official window and earliest of 3 pre-official calibration origins
- `provisional_family_groups.csv` — frequency-normalized family hints, **not verified source lineage**
- `audit_manifest.json` — registry SHA and limitations

**Scientific protocol:** the benchmark's official 10/20 windows are held back for testing. To select model/context without contaminating official outcomes, build *three separate, disjoint calibration horizons* ending strictly before the first official benchmark cutoff, rather than scoring the official windows 0-3 to tune a policy. A calibration window with cutoff `n-k*h` has target indices `[n-k*h,n-(k-1)*h)` and thus never reaches the first official target `[n, n+h)`.

The feasibility grid includes short contexts (8/16/32/64) for monthly and yearly tasks; these are **candidate probes only**, not fixed model configurations.

**Limitations:** These are only historical feasibility probes, not completed calibration windows. Benchmark definition SHA is verified but Hugging Face underlying dataset revision is not yet pinned. Family grouping is heuristic; manual source-level lineage checks are mandatory. SQL (primary fev-bench metric) is probabilistic; native quantile scoring must be implemented and validated before leaderboard comparisons. The results are not official fev-bench scores.
