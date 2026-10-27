# PersTSFM Phase 2.4 — Frozen fev input-only protocol

This development-only protocol is derived from the existing `fev_preflight.py` manifest and fev 0.10.0. It does not load future target values and never calls `window.get_ground_truth()`.

Run in the **PERTSFM_DATA_ENV**:

```bash
cd ~/RDS/home/projects/PersTSFM
conda activate PERTSFM_DATA_ENV

for task in m4_hourly ett1_ot ercot; do
    python -m scripts.phase2.freeze_fev_protocol --task "$task" \
      2>&1 | tee "results/phase2/external_preflight/${task}_freeze.log"
done
```

First inspect using `--check-only` if desired. To deliberately overwrite a previously frozen protocol, use `--force`, but preserve the old manifest beforehand.

Main grids:
- M4 Hourly: [96, 192, 384], which all 414 series can support at each window. A 768-step *separate sensitivity cohort* of 245 series must not be compared to MAEs computed on the 414-series cohort.
- ETT1 OT, ERCOT: [96, 192, 384, 768, 1536, 3072]. Context feasibility must be fixed before forecasting.

The output for each task is `results/phase2/external_protocol/<task>/selection_inputs.parquet`, `evaluation_inputs.parquet`, and `manifest.json`. These files are physically separate: **do not give `evaluation_inputs.parquet` to the configuration-selection policy**.
The table contains case_id, entity_id, window_index, role, truncated history values and timestamps, and *known future timestamps only*. It contains **no future target values**. The later scoring stage must obtain ground truth separately and must never expose it to the model or the selection policy while configurations are selected.

By default, windows 0–3 are selection windows and 4–7 evaluation windows. Strict per-entity non-overlap is verified. For ETT1/ERCOT, global calendar separation is checked; M4 uses per-series chronology only. The same common cohort is used for every model, context and window of a given main task.

This is development data, **not held-out meta-test**. All checkpoints' pretraining-source overlap must be recorded in subsequent reporting. The fev task's own preferred metric (often MASE) should be calculated alongside PersTSFM mean-entity MAE and not conflated with it.
