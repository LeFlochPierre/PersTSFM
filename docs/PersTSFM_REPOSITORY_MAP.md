# PersTSFM repository map and cleanup policy

## Live layout (after safe Stage 1)

```text
PersTSFM/
  README.md                       main project description
  configs/                        reproducible experiment configurations
  src/                            maintained model adapters/evaluation logic
  scripts/
    phase2/                       original phase-2 pipeline
    phase3/                       new full-history context/LoRA scripts
    maintenance/                  cleanup and audit utilities
  tests/                          phase-2 and phase-3 regression tests
  docs/
    PersTSFM_RESEARCH_RECAP_2026-10-27.md
    PersTSFM_REPOSITORY_MAP.md
    phase2/                       earlier methods and protocol notes
    phase3/                       recent experimental notes
    research_snapshots/           older date-specific handovers
  archive/
    packages/                     installation ZIPs (not imported by runners)
    installers/                   superseded one-shot patches/installers
    manifests/                    reversible cleanup logs
  data/                           never cleaned automatically
  results/                        never cleaned automatically
  references/                     papers and bibliography
  external/                       external resources
```

## Why active `PersTSFM_phase*.py` scripts are still in the root

Several tests import these scripts *directly by name* (`from PersTSFM_phase3_1_adaptive_racing ...`) or by literal file path (`ROOT/'PersTSFM_phase3_3a_chronos_lora.py'`). Moving them without updating imports/tests and source path assumptions breaks the suite. The cleanup utility therefore **moves only packaging/docs/superseded one-shot patches**, and generates a references audit for all remaining root-level runners.

## Stage 2: code migration (future separate change)

1. Create a Git checkpoint of source and tests, without adding binary/results directories.
2. Choose a single canonical module per phase; consolidate overlapping phase runners only after parity/regression checks.
3. Migrate one script at a time to `scripts/phase2/` or `scripts/phase3/`, updating test imports, subprocess invocations, README commands, and project-root resolution using `Path(__file__).resolve().parents[2]` where appropriate.
4. Run phase-specific pytest, frozen-result parity checks and dry runs **before removing any old entry point**.
5. Keep a move/compatibility map under `docs/`; only archive genuinely superseded implementations.

## Git recommendations

Do **not** commit generated prediction catalogs, checkpoints, calibration snapshots, `.npz`, `.parquet`, `.csv` result bundles, or `archive/packages/` ZIPs without a deliberate lightweight/large-file strategy. Do commit reproducible code, tests, curated documentation and environment specifications. Always inspect `git diff --cached --stat` before any commit. A Git deletion paired with a new path often represents a move, not lost content.

Cleanup is never a scientific deletion: preserve historical negative results and make restoration manifests permanent.
