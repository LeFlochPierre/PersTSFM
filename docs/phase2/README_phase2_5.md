# PersTSFM Phase 2.5 — Selection-only external FEV runner

**Scope:** M4 Hourly, ETT1 OT, ERCOT — FEV 0.10.0, frozen external protocols created earlier.

This additive patch never changes the six-dataset V1/V1.2 results, local FEV protocol files, or model adapters. Inference reads only `selection_inputs.parquet` and `manifest.json`. It is impossible to request evaluation-window histories through the inference CLI.

## Three scripts

- `run_fev_selection.py`: runs one existing PersTSFM model checkpoint at one frozen context against selection histories. Creates per-case forecasts and a runtime audit; blocks overwriting previous runs.
- `score_fev_selection.py`: separate, selection-only label-aware scorer. Runs in `PERTSFM_DATA_ENV` and calls `fev.Task.get_window(w).get_ground_truth()` **only for selection windows**. Reports matched-case MAE and a seasonal scaled MAE (MASE) estimate. Publication-grade native FEV parity is still to be verified.
- `summarise_fev_selection.py`: prints matched full-cohort selection results. Does not access any ground truth.

## Installation

From repository root on `research/phase2-tsfm`:

```bash
unzip -n PersTSFM_phase2_5_fev_selection_runner.zip
conda activate PERTSFM_ENV
python -m pytest -q tests/phase2/test_fev_external_selection.py
```

## Recommended workflow (ETT first)

```bash
conda activate PERTSFM_MOIRAI_ENV
python -m scripts.phase2.run_fev_selection --task ett1_ot --model moirai2_small --context 96 --windows 0 --max-entities 2
# Tiny smoke output in .../subset_2_w0, no scoring yet.

python -m scripts.phase2.run_fev_selection --task ett1_ot --model moirai2_small --context 96
# Full eight SELECTION cases (two entities x four windows).

conda activate PERTSFM_DATA_ENV
python -m scripts.phase2.score_fev_selection \
  --predictions results/phase2/external_forecasts/ett1_ot/selection/moirai2_small/context_96/all/predictions.parquet

python -m scripts.phase2.summarise_fev_selection
```

## Running additional models

```bash
conda activate PERTSFM_TOTO_ENV
python -m scripts.phase2.run_fev_selection --task ett1_ot --model toto2_313m --context 96

conda activate PERTSFM_DATA_ENV
python -m scripts.phase2.score_fev_selection \
  --predictions results/phase2/external_forecasts/ett1_ot/selection/toto2_313m/context_96/all/predictions.parquet
```

Other supported model keys (from prior installs): `chronos2`, `timesfm3`, `tirex2`, `toto2_22m`, `toto2_313m`, `moirai2_small`. Existing original 3 models use `PERTSFM_ENV`.

Frozen context grids:
- M4 Hourly: 96, 192, 384 — common 414-series cohort across all windows.
- ETT1 OT: 96, 192, 384, 768, 1536, 3072 — two entities.
- ERCOT: 96, 192, 384, 768, 1536, 3072 — eight entities.

## Safeguards

1. No target labels in model inference; scorer is explicitly separate.
2. No evaluation-window history or ground truth loaded by any patch script.
3. Every frozen input Parquet hash checked; forecast files hashed and non-overwriting.
4. Every output ID, forecast horizon, finite value, quantile order and timestamp checked.
5. Comparability requires full selection cohorts. The scorer refuses partial smoke-test outputs.
6. Recorded times are model-load and batched inference times separately; they do not include all orchestration overhead and are **not yet end-to-end GPU cost**.
7. M4 relative timestamps are not a globally meaningful calendar. Dataset family and pretrained-data overlap are development-only cautions.
8. The seasonal scale in selection MASE uses the entire available past from FEV; parity with official `fev` aggregation must be independently validated before using MASE in the paper.
9. Do not run evaluation-window inference or inspect its labels until a policy and protocol are frozen. These three tasks remain **development tasks**, never final held-out evidence.
