# PersTSFM Phase 2.10 — Pre-official historical calibration inference (starter)

**Purpose:** connect Phase 2.9 pre-official calibration cases to the existing PersTSFM model adapters, while keeping calibration labels isolated from the inference process.

The inference runner (`run_preofficial_calibration.py`) reads only `calibration_inputs.parquet` plus its manifest. The separate scorer (`score_preofficial_calibration.py`) verifies file hashes and matched case/timestamp IDs before opening `historical_calibration_labels.parquet`. Neither script imports `fev`, opens any official evaluation window or calls `get_ground_truth`.

**Supported for now:** target-only, regular-cadence, finite input histories, with requested context feasible for **every** chosen calibration case. Model output must include point forecasts and q10, q50, q90. Monthly, quarterly, yearly and other irregular calendar cadences are **deliberately rejected**, pending a proper calendar-aware adapter. This is not official fev-bench scoring. Measured runtime is exploratory; no warmup/repeated timing.

**Inputs:** existing Phase 2.9 task folders containing manifest, inputs, historical labels, and entity inventory. This patch **does not** change them.

## 1. Install and test

```bash
cd ~/RDS/home/projects/PersTSFM
conda activate PERTSFM_DATA_ENV
unzip -n PersTSFM_phase2_10_preofficial_runner.zip
python -m pytest -q tests/phase2/test_preofficial_calibration_inference.py
```

## 2. Start on M_DENSE_1H (30 entities, horizon=168)

The task ID from your Phase 2.9 export is `fev_0_10_0_044`. First make a 2-entity smoke prediction with 1 calibration origin:

```bash
conda activate PERTSFM_MOIRAI_ENV
python -m scripts.phase2.run_preofficial_calibration \
  --task-id fev_0_10_0_044 --model moirai2_small \
  --context 96 --k 1 --max-entities 2
```

If this succeeds, do the full 30-entity K=1 run:

```bash
python -m scripts.phase2.run_preofficial_calibration \
  --task-id fev_0_10_0_044 --model moirai2_small \
  --context 96 --k 1
```

## 3. Score *historical* calibration labels, not official evaluation targets

```bash
conda activate PERTSFM_DATA_ENV
python -m scripts.phase2.score_preofficial_calibration \
  --predictions results/phase2/preofficial_forecasts/fev_bench_v0_10_0/fev_0_10_0_044/moirai2_small/context_96/k1/full/predictions.parquet
```

You should see `n_cases=30`, MAE and a **three-quantile mean pinball diagnostic**. It is NOT official SQL (Scaled Quantile Loss). The smoke forecast is in `.../k1/smoke_e2/` and is never used as a full benchmark score.

## 4. Test the K=3 calibration path

```bash
conda activate PERTSFM_MOIRAI_ENV
python -m scripts.phase2.run_preofficial_calibration \
  --task-id fev_0_10_0_044 --model moirai2_small \
  --context 96 --k 3
conda activate PERTSFM_DATA_ENV
python -m scripts.phase2.score_preofficial_calibration \
  --predictions results/phase2/preofficial_forecasts/fev_bench_v0_10_0/fev_0_10_0_044/moirai2_small/context_96/k3/full/predictions.parquet
```

This should cover all 90 historical cases. **Never directly compare average K=1 and K=3 losses as though they were the same test cohort:** they include different forecasting origins. Their purpose is to test the acquisition budget, and decisions should ultimately be evaluated on an identical later locked target window.

## Guardrails
- All original Phase 2.9 artifacts are write-once and SHA-verified; output directories are also write-once.
- The runner refuses per-context reductions in case count, missing inputs or irregular time steps. Future work can implement explicitly labelled matched subcohorts and missing data imputation.
- The scorer refuses incomplete/smoke runs by default. Use `--allow-smoke` only for debugging.
- This stage has no FEV native SQL parity, no official evaluation access, no repeated timing calibration and no meta-learned policy.
- Task family independence and immutable dataset revisions still need verification before final scientific evaluation.
