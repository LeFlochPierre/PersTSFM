# PersTSFM Direction A: B0 feasibility scaffold

**Status:** offline benchmark implementation, not a trained routing policy and not GPU benchmarking.

This addition is deliberately separate from the existing `src/models/adapters.py` (`ForecastInput`, `ForecastOutput`, `ForecastModel.predict_batch`) and `src/data/registry.py`. It cannot modify a checkpoint, inference wrapper, dataset, or experiment result.

## Quick start (from the PersTSFM root)

```bash
conda activate PERTSFM_ENV
python -m pytest -q tests/direction_a
python -m src.budgeted_forecasting.cli demo --output results/direction_a/demo.csv
python -m src.budgeted_forecasting.cli validate --input results/direction_a/demo.csv
python -m src.budgeted_forecasting.cli analyze \
  --input results/direction_a/demo.csv \
  --output-dir results/direction_a/demo_report \
  --budget-ms 2 --budget-ms 10 --budget-ms 30
```

**The demo uses synthetic predictions and fake illustrative latencies. Never report the demo as empirical evidence.** The `results/` folder is already ignored by the PersTSFM `.gitignore`.

## Cache schema

One CSV row per `(case_id, action_id, step)`:

| Column | Meaning |
| --- | --- |
| `dataset_id` | Dataset identifier; used for dataset-macro metrics |
| `series_id` | Entity/sensor/patient identifier within dataset |
| `case_id` | Unique forecasting origin (dataset/series/origin) |
| `split` | `calibration` or `evaluation` |
| `origin_time` | ISO 8601 timestamp with timezone |
| `label_available_at` | When **all** ground-truth horizon targets are observed, with timezone |
| `action_id` | Configuration ID, e.g. model+context |
| `step` | Target horizon step, starting at 1 |
| `truth` | Observed target (for evaluation; inaccessible to future online policy until label time) |
| `prediction` | Predicted target |
| `latency_ms` | **Measured per-origin action runtime**, repeated identically for each horizon step |

The converter must export *every action* for each case; this first version rejects missing actions, incomplete horizons, duplicate steps, inconsistent truths or costs, unzoned timestamps, and calibration labels that have not matured before the first evaluation origin anywhere in the cache (strict pooled-policy temporal split). Strictly separate calibration and evaluation **globally in time**, not just within each series. For datasets with non-aligned historical calendars, run separate cache experiments or define an explicitly documented training cutoff; this B0 implementation rejects mixed-timeline pooling rather than silently leaking. Cached outcomes must never be exposed to future policies before label availability.

## Baseline semantics

For each compute budget (ms):

1. Estimate action eligibility using **calibration median** execution latency. This does **not** guarantee an evaluation inference stays under budget; actual overruns are reported.
2. Choose `cheapest` based on calibration median runtime.
3. Choose `validation_selected` based on calibration **dataset-macro MAE** among eligible actions.
4. Calculate a **hindsight oracle** selecting the action with lowest per-case MAE on evaluation labels among the same eligible actions. The oracle is *not deployable*. Its performance is an optimistic bound, not proof that a selector can achieve the gap.
5. Report evaluation dataset-macro MAE and mean runtime, budget overruns, and the oracle gap.

This is a **single-action** feasibility test; it does not yet implement sequential acquisition, cumulative cost, forecast ensembling, confidence policies, delayed-feedback updates, or tests of cross-domain generalization.

## Connecting real PersTSFM inference (next step)

1. Preserve the existing `src/data/registry.py` dataset protocols and timestamps; do not regenerate train/evaluation splits ad hoc.
2. For each forecast origin and `action_id`, use the existing model adapters' `predict_batch` API to produce the horizon array, and record the **exact checkpoint, context, batch size, device, warm/cold state, and wall time** in experiment metadata.
3. Make sure context ends no later than `origin_time`; obtain truth only from the forward forecast horizon. `label_available_at` is the last target's actual observation time, **not** the time an offline exporter ran.
4. Write rows matching the schema above and validate them before analysing. Do not fabricate runtime using parameter counts or assumed speed tiers.
5. Run one real dataset and two fully verified actions before expanding the action registry.

Future improvements: separate `costs` and `predictions` tables for multi-batch profiling, dataset-held-out selectors, richer sequence features, sequential actions, and late-label prequential replay.

## Non-destructive installation

Do not blindly unzip over existing files. `unzip -n` avoids overwriting but still creates any missing files; review proposed names with `unzip -l` and Git first. All files in this package live under `src/budgeted_forecasting/`, `scripts/direction_a/`, `tests/direction_a/`, `configs/direction_a/`, and `docs/direction_a/`.
