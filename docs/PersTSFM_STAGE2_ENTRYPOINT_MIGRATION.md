# PersTSFM — Stage 2: move live root scripts into `scripts/`

This package **does not delete experiments or refactor their numerical methods**.

## What moves

The old root-level analysis scripts `PersTSFM_phase2_11_*`, `_12_*`, `_14_*`, `PersTSFM_phase2_fev_preflight.py`, `PersTSFM_phase2_temporal_policy_audit.py`, `PersTSFM_temporal_selection_diagnostics.py` move to `scripts/phase2/`.

`PersTSFM_phase3_0_*`, `_1_*`, `_2_*`, `_3a_chronos_lora.py`, and `_3b_history_audit.py` move to `scripts/phase3/`.

Two **one-off installers only** (`PersTSFM_phase2_15_ercot_missing_guard.py` and `PersTSFM_phase3_3a_fix_rolling_history.py`) move to `archive/installers/`. They are not experimental runners. Their fixes should already have been applied, and their working files are preserved unchanged.

The script updates direct Python imports from these root modules and literal quoted `.py` references within `src/`, `scripts/`, `tests/`, and root-level Python files. Existing modern `scripts/phase3/context_saturation.py` and `scripts/phase3/lora_history_saturation.py` are **not changed** unless they explicitly reference a legacy entry point.

**No modifications to `data/`, `results/`, `external/` or historical archived packages.** It does not rewrite old results/manifests or historical documentation.

## Run safely

Work from `~/RDS/home/projects/PersTSFM`:

```bash
python -m pytest -q tests/maintenance/test_migrate_root_entrypoints.py
python scripts/maintenance/migrate_root_entrypoints.py              # preview only
python scripts/maintenance/migrate_root_entrypoints.py --apply      # moves + import/path rewrites
python -m pytest -q tests/phase2 tests/phase3                      # regression verification
ls *.py                                                           # should show no migrated Phase 2/3 scripts
```

Migration writes a checksum-protected manifest under `archive/manifests/entrypoints_<UTC>/manifest.json`; use the EXACT manifest printed on your machine to restore. Restore refuses to overwrite files changed since migration.

If a test fails, retain the output and restore using the printed command. Migration tests are synthetic; your real repo may have additional dynamic references that need attention.

## Execute the moved scripts

Use `python -m` **from the repository root** so that `src` and `scripts` imports remain resolvable (don't use `python scripts/phase3/...` for modules importing `src`):

```bash
conda activate PERTSFM_ENV
python -m scripts.phase3.PersTSFM_phase3_3a_chronos_lora --help

conda activate PERTSFM_DATA_ENV
python -m scripts.phase3.PersTSFM_phase3_0_probe_value --help
python -m scripts.phase2.PersTSFM_phase2_12_joint_temporal_replay --help
```

Existing `scripts/phase3/context_saturation.py` and `scripts/phase3/lora_history_saturation.py` can continue to run with their established commands.

## Important Git warning

The repository has uncommitted modified source and many untracked research files; **don't use `git clean -fd`, `git reset --hard`, or a blanket `git add -A`**. The migration only moves/references explicit script names and saves backups; Git might report moves as deletions and new files until the relevant files are staged.

Suggested inspection:

```bash
git status --short
git diff -- src/
git diff --check
find . -maxdepth 1 -type f -name 'PersTSFM_*.py' -print
```

Keep generated result CSV/parquet/npz, model checkpoints and archival ZIPs out of ordinary Git staging. Stage selected code and tests only after regression tests pass.

## Remaining root directories

`pertsfm_phase2_benchmark/` is a historical directory with its own README; it is intentionally **not automatically moved**. Verify whether that README describes a still-active protocol before archiving the directory. `notes/`, `references/`, `data/`, and `results/` are not clutter to remove and remain untouched.
