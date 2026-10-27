# PersTSFM Phase 2.9 — Pre-official historical calibration cases

Phase 2.7 and 2.8 only computed history-length *feasibility*. This patch exports actual
model inputs and corresponding historical calibration targets before the **first**
official `fev-bench` evaluation period. It accesses ONLY `get_window(0).get_input_data()`
and only its historical `past` portion; it never opens official benchmark test labels.

## Install

```bash
cd ~/RDS/home/projects/PersTSFM
conda activate PERTSFM_DATA_ENV
unzip -n PersTSFM_phase2_9_preofficial_calibration.zip
python -m pytest -q tests/phase2/test_preofficial_calibration_export.py
```

## Validate end-to-end on one small, full-cohort task

```bash
python -m scripts.phase2.export_preofficial_calibration --configs us_consumption_1M \
  2>&1 | tee results/phase2/phase2_9_smoke.log
```

You should see 31 entities and up to three calibration cases per entity (93 cases).
The exported cases use the first official input window's historical observations;
label horizons are pre-official, disjoint, and strictly later than their own inputs.

## Then export independent development families

```bash
python -m scripts.phase2.export_preofficial_calibration \
  --configs M_DENSE_1H hospital_admissions_1D ecdc_ili \
  2>&1 | tee results/phase2/phase2_9_families.log
```

Outputs for each task under
`results/phase2/preofficial_calibration/fev_bench_v0_10_0/<task_id>/`:

- `calibration_inputs.parquet`: univariate model histories and forecast timestamps, no labels.
- `historical_calibration_labels.parquet`: calibration truth from historical observations only; to be read *after* inference by a **separate calibration scorer**, which is not provided yet.
- `entity_inventory.json`: per-entity supported K, context eligibility, missing observations.
- `manifest.json`: task identity, counts, hashes and provenance.

Files are write-once. Existing task directories are refused, even if partial.

## Scientific limitations

- This is a *historical calibration corpus*, **not** published fev-bench leaderboard inference.
- No native SQL/quantile scoring is provided yet; no model forecasts are run by this patch.
- A `K=0` policy makes no calibration forecasts. Such a policy needs its own dataset-features-only selection method.
- A fixed K may be impossible for some entities; the inventory records exact eligible cohorts.
- The context feasibility set may be different at each origin; compare configurations only on matched cases.
- Missing targets are not imputed or dropped. Task `hierarchical_sales_1D` has already shown many missing observations and requires a separate data-quality/adapter audit.
- Series are individually chronological, with **no global calendar alignment guarantee**.
- Dataset revisions are not yet pinned; source-family independence is unverified.
- The historical labels are intentionally available for *calibration*, but must not be used as official evaluation test data.
