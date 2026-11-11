# Repository reset audit — 11 November 2026

Baseline: clean `research/persistence` at checkpoint `a31f1c5`; 392 tests passed in `PERTSFM_ENV` before changes. No `AGENTS.md`, requirements file, `pyproject.toml`, or active dependency manifest was found. Root `environment.yml` is empty; multiple Conda environments exist, so dependency removal cannot be inferred from it.

## Target architecture

Keep genuinely shared functionality in `src/data`, `src/models`, and `src/evaluation`; put the new experiment in `research/persistence` without copying old modules. Keep legacy research packages and entry points with their tests until a reproduction manifest establishes exact commands and outputs. Current identity lives at root and `docs/research_proposal`; legacy scientific plans remain in existing phase docs and `docs/history`.

| Path | Purpose | Class and recommendation | Risk and validation |
|---|---|---|---|
| `src/data/registry.py`, `src/data/prep/` | Dataset metadata, loading, protocol builders | D: retain; reuse selectively | Audit entity IDs, time units, missingness, and chronology. |
| `src/models/` | TSFM forecast interface/adapters | D/C: retain; optional for first pilot | Model-specific environments; preserve adapter tests. |
| `src/evaluation/metrics.py`, `aggregation.py`, `case_sampling.py` | Metrics, aggregation, sampling | D: retain; inspect semantics | Quantile sorting and entity weighting require checks. |
| `src/experiments/adaptation_benchmark/` | Phase 3 LoRA and rolling evaluation | C: historical; reuse provenance ideas | Direct script/test imports; preserve as a group. |
| `src/selfconfig/`, `src/budgeted_forecasting/` | Previous self-configuration/budget research | C: later archive candidates | Many test/script/doc dependencies; moving now risks reproducibility. |
| `src/analysis/`, `src/experiments/run_context_*` | Previous context and routing analysis | B/C: inspect per entry point | Some scripts write on import; audit before relocation. |
| `scripts/phase2/`, `scripts/phase3/`, `scripts/direction_a/` | Previous research CLIs | C: preserve with source/config/tests | Exact test imports and documented commands. |
| `scripts/maintenance/` | Migration and diagnostic tools | B/C: inspect for archival | Tests depend on path and behaviour. |
| `configs/`, matching `tests/` | Frozen settings and regression evidence | C: preserve together | Moving alone breaks paths and parity. |
| `docs/phase2/`, `docs/phase3/`, `docs/direction_a/` | Historical protocols/findings | C: preserve and index later | Unique provenance. |
| `data/`, `results/`, `archive/` | Data, outputs, backups | C: preserve; generated files excluded from routine commits | Some result CSVs are already tracked; never bulk-delete. |
| `__pycache__/`, `.pytest_cache/`, `*.pyc` | Regenerable caches | A: safe cleanup candidate | Ignored by Git; no scientific value expected. |
| `environment.yml` | Empty placeholder | B: leave until environment audit | Existing model-specific environments are not represented. |

A = safe disposable; B = potentially obsolete; C = historical reproducibility; D = active reusable.

## Stage completed and next gate

This stage rewrites `README.md` and `Project Goal.md`, preserves the old goal as `docs/history/Project_Goal_budget_aware_2026-11-04.md`, and adds `research/persistence/README.md`. No source, tests, dataset, result, or old experiment configuration is removed. Exact deletion list: **none**.

Before archiving legacy code, make a per-entry-point map of source/config/doc/test references and a reproduction command for each completed phase. Potential archive groups are `src/selfconfig/` with its dependents and `src/budgeted_forecasting/` with its dependents; these are proposals requiring review because moves could compromise reproducibility. The active package stays small by keeping the new persistence work separate rather than adding wrappers around old research code.
