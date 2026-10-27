# PersTSFM Phase 3.3C — Full-history Chronos-2 context saturation

## Why this experiment?

The Phase 2 exporter froze only the most recent 3,072 input values per historic forecast origin. The actual pre-official FEV histories are often much longer. Phase 3.3B verified M_DENSE_1H has 15,504 training observations before lag 2 (per entity) and is feasible through context 8,192.

This experiment isolates **inference context length** only. There is **no fine-tuning** and no official FEV evaluation-label access. Later, vary LoRA *training-history pool length* independently.

## Exact chronology

FEV v0.10.0 `get_window(0).get_input_data()` returns pre-official history `x` of length `N`. For forecasting horizon `H`:

- Model input at lag 2: `x[N-2H-C:N-2H]`
- Validation target at lag 2: `x[N-2H:N-H]`
- Model input at lag 1: `x[N-H-C:N-H]`
- Later historical target at lag 1: `x[N-H:N]`

The complete grid is predicted before later targets are scored, with the **context locked on lag-2 mean MAE** and later scoring requiring a separate command. The pre-official historical test is exploratory (the particular datasets have already informed development). Never claim official FEV benchmark performance from these results.

The model uses `src.models.adapters.Chronos2Model.predict_batch` (identical to existing zero-shot adapter), and reports point MAE, 3-quantile diagnostic pinball and raw crossing diagnostics. Quantiles are rearranged for diagnostic pinball, as in earlier phases. No native nine-quantile FEV SQL here.

## Two Conda environments — keep them separate

FEV and Chronos-2 need not be installed in the same environment. Export a reproducible
non-pickle `.npz` containing **only pre-official history** in `PERTSFM_DATA_ENV`,
then load it in `PERTSFM_ENV` for prediction. This does not modify either environment.

The snapshot contains all history, including pre-official historical lag-2 and lag-1
periods. Only lag-2 *scores* may determine context selection; lag-1 scores are
written separately after `selection.json` is locked.

## Install

Copy `scripts/phase3/context_saturation.py` and `tests/phase3/test_context_saturation.py` into the same relative repo paths. No prior scripts are modified.

```bash
cd ~/RDS/home/projects/PersTSFM
conda activate PERTSFM_DATA_ENV
python scripts/phase3/context_saturation.py \
  --task M_DENSE_1H \
  --export-snapshot results/phase3/context_saturation_inputs/mdense_preofficial_full.npz

conda activate PERTSFM_ENV
python -m pytest -q tests/phase3/test_context_saturation.py
python scripts/phase3/context_saturation.py --task M_DENSE_1H \
  --snapshot results/phase3/context_saturation_inputs/mdense_preofficial_full.npz
```

## Pilot, then full cohort

```bash
mkdir -p results/phase3
set -o pipefail
python scripts/phase3/context_saturation.py \
  --task M_DENSE_1H --snapshot results/phase3/context_saturation_inputs/mdense_preofficial_full.npz \
  --max-entities 4 --contexts 768 3072 8192 \
  --batch-size 4 --execute \
  2>&1 | tee results/phase3/phase3_3c_context_smoke.log

python scripts/phase3/context_saturation.py \
  --task M_DENSE_1H --snapshot results/phase3/context_saturation_inputs/mdense_preofficial_full.npz \
  --max-entities 0 \
  --contexts 96 384 768 1536 3072 4096 6144 8192 \
  --batch-size 8 --execute \
  2>&1 | tee results/phase3/phase3_3c_mdense_context_full.log
```

Each run prints a saved directory under `results/phase3/context_saturation/`. Once validation has finished and the choice is locked:

```bash
python scripts/phase3/context_saturation.py \
  --score-later results/phase3/context_saturation/<ACTUAL_RUN_DIRECTORY>
```

## Saved outputs

- `manifest.json`: exact matched entity IDs, contexts, cadence, data/snapshot hashes and hardware.
- `predictions.npz`: raw point forecasts and quantile-rearranged q10/q50/q90 for both historical origins; official test labels excluded.
- `validation.csv`: lag-2 metrics only.
- `selection.json`: lag-2 best context; hashes of prediction, validation and manifest files.
- `later.csv`: lag-1 metrics, written only by `--score-later`, without GPU inference.

## Scientific limitations

1. The comparison uses one model, one seed and one chronological replay. Extend to further origins/datasets for robust saturation claims.
2. Quantile metrics are not official nine-quantile FEV SQL.
3. The entire pre-official input is available to the script but *lag-1 historical targets are never used for selection*. The later diagnostic is separate.
4. Context length is capped by the loaded Chronos-2 model's `model_context_length`. This is not the amount of full history usable for LoRA training.
5. No full fine-tuning, model retraining or changes to Phase 2 results. Scientific preflight tests run without a GPU.
