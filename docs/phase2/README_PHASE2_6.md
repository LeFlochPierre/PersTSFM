# PersTSFM Phase 2.6 — official fev-bench registry

An additive metadata-only patch. Extract at the repository root. No datasets are downloaded, no GPU is used, and no held-out targets are accessed.

## Prerequisites
- `PERTSFM_DATA_ENV` (fev 0.10.0 + PyYAML).
- Official benchmark tasks are pinned to `autogluon/fev` Git tag `v0.10.0` (`benchmarks/fev_bench/tasks.yaml`).

## Run
```bash
cd ~/RDS/home/projects/PersTSFM
conda activate PERTSFM_DATA_ENV
python -m scripts.phase2.audit_official_benchmarks \
  2>&1 | tee results/phase2/benchmark_registry_audit.log
```

Produces `results/phase2/benchmark_registry/fev_bench_v0_10_0/`:
- `official_tasks.yaml`: exact upstream task definitions, version-pin and raw SHA256 recorded in manifest.
- `task_catalog.csv`: 100 official task definitions and input requirements.
- `family_groups.csv`: exact dataset-identifier grouping for grouped cross validation; requires human review of lineage.
- `feasibility.csv`: potential contexts marked UNKNOWN until input-history/adapter audits.
- `overlap_audit.csv`: flags names matching previous M4, ETT1, and ERCOT pilots or alias collisions.
- `benchmark_manifest.json`: counts, run environment, official schema caution, source SHA256.

## Scientific caveats
- **Metadata-only:** no series lengths or frequencies verified. A context is never automatically declared feasible.
- **Not official results yet:** many tasks have covariates/multivariate targets or native `SQL` scores with quantiles. A target-only MAE/MASE experiment is a research ablation, not equivalent to official fev-bench comparison.
- **Leakage:** selecting a configuration using the official test windows invalidates official benchmark comparability. For learned selection, add selection-only inner validation windows before any official evaluation cutoff.
- **Split:** divide whole independent source families, not individual tasks/windows. Alias flags are hints, not proof of common sources. Perform an explicit source lineage and TSFM pretraining overlap audit.
- **Provenance:** pinning the YAML does not pin the underlying Hugging Face dataset snapshots; that comes next.
