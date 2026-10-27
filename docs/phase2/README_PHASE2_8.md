# PersTSFM Phase 2.8 — adaptive historical calibration budget audit

This additive patch contains only a CSV/manifest analyser and offline contract tests. It has no FEV dependency and never reads datasets or future targets.

## Install

```bash
cd ~/RDS/home/projects/PersTSFM
conda activate PERTSFM_DATA_ENV
unzip -n PersTSFM_phase2_8_calibration_budget.zip
python -m pytest -q tests/phase2/test_calibration_budget.py
```

## Run

```bash
python -m scripts.phase2.audit_calibration_budget \
    2>&1 | tee results/phase2/calibration_budget_audit.log
```

The output lives under `results/phase2/calibration_budget/fev_bench_v0_10_0/`:

- `task_calibration_budget_matrix.csv`: task-by-K-by-context feasibility from shortest inspected history
- `calibration_budget_summary.csv`: counts for K=0,1,2,3 with provisional family hints
- `manifest.json`: source hashes and limitations

Interpretation: K>0 is the number of non-overlapping historical calibration horizons, K=0 is no validation probing. Calculations apply to the first official evaluation input window, not benchmark future targets. A row with `coverage != FULL` cannot establish whole-dataset feasibility. This does not build calibration forecasts, verify observed time indices, implement official SQL scoring, or establish independent family train/test splits.
