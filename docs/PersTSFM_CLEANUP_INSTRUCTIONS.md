# Safe PersTSFM cleanup — Stage 1

## Goal

Reduce repository-root clutter without relocating active runnable scripts, breaking test imports, touching datasets/results, deleting files, or staging Git changes.

## Installation

Unzip the supplied package from the repo root:

```bash
cd ~/RDS/home/projects/PersTSFM
unzip -n PersTSFM_recap_and_safe_cleanup.zip
```

## Preview and archive (no deletion)

```bash
python scripts/maintenance/PersTSFM_root_tidy_stage1.py --show-references
python -m pytest -q tests/maintenance/test_root_tidy_stage1.py
python scripts/maintenance/PersTSFM_root_tidy_stage1.py --apply
```

The tool moves root-level `.zip` files into `archive/packages/`, old known one-shot patch installers into `archive/installers/` **only if not referenced in runnable code/tests**, a timestamp-fix README into `docs/phase3/`, and the earlier research-status snapshot into `docs/research_snapshots/`. It keeps every active `PersTSFM_phase*.py` runner in the root, alongside `src/`, `scripts/`, `tests/`, `data/`, and `results/`.

It will print a new manifest under `archive/manifests/root_tidy_<timestamp>.json`. To reverse Stage 1:

```bash
python scripts/maintenance/PersTSFM_root_tidy_stage1.py --restore archive/manifests/root_tidy_<ACTUAL_TIMESTAMP>.json
```

Restore refuses modified or occupied paths rather than silently overwriting files. Keep the earlier cleanup manifest (`cleanup_20261027T180152554261Z.json`) separately.

## Git: inspect before committing

Your current `git status` shows tracked modifications in `src/` and moved tracked docs, plus many untracked results and scripts. None of these are deleted or staged by the tool. Do **not** run a blanket `git add -A` here.

The remote `.gitignore` already excludes many binary results, but may not ignore `results/**/*.csv` and `.json` and older archive ZIPs. To keep local runs out of normal Git status, consider adding, **after inspecting existing tracked files and Git policy**:

```gitignore
# Local, reproducible run outputs; opt in explicitly for curated tables
/results/
/archive/packages/
/archive/installers/
/archive/script_backups/
```

Because `gitignore` does not untrack existing tracked files, check the tracked situation first:

```bash
git ls-files results archive | head -50
git status --short -uno
git diff --stat
```

If `results/` contains already-tracked content you wish to preserve, choose a more selective ignore rule. Commit source/tests/docs in a deliberate, inspected set of changes; do not auto-commit huge experiment artifacts.

## Stage 2 (not automatic)

Root runnable scripts should be migrated to `scripts/phase2/` and `scripts/phase3/` only in a dedicated code-refactor change. Existing regression tests import several root scripts by path/name. The tidy script shows these references for migration planning. Run all affected tests and representative CLI dry runs after each move.

## Research handover

Read `docs/PersTSFM_RESEARCH_RECAP_2026-10-27.md`. It supersedes older summaries for the current research question and includes Phase 3.3D results, history-saturation findings, negative LoRA results and next actions.
