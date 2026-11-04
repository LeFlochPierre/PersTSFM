# PersTSFM Direction A — B2.1 exact origin matching fix

## Symptom

`extend_atlas run-action` fails before TiRex-2 inference with:

```
ValueError: Case mismatch: missing=['electricity::MT_092::...', ...], extra=[]
```

The existing B1.2 `load_matched_dataset()` attempts to invert the B1.1 cohort selection by counting distinct calibration/evaluation timestamps and rerunning `load_electricity_pilot()`. This is fragile: B1.1 selects **global timestamps**; different entities need not be present at every selected timestamp, and a reconstructed sampling operation should not be treated as the source of truth for matching an existing cache.

## Patch

`patch_b12_exact_origins.py` makes only two edits to `src/budgeted_forecasting/extend_atlas.py`:

1. Import `prepare_cases` instead of `load_electricity_pilot`.
2. Replace the matched-dataset reconstruction function with an exact lookup of every `(entity_id, forecast_origin)` key from the B1.1 cache against the frozen electricity protocol. Extract raw histories using the existing registry and compare *every* case ID, split, timestamp and ground-truth target against the original cache.

It does **not** modify the B1.1 cache, model adapters, tests, datasets, or existing results. It does not run GPU inference.

## Install

From the PersTSFM repo root:

```bash
unzip -n PersTSFM_DirectionA_B21_exact_origin_fix.zip
python scripts/maintenance/patch_b12_exact_origins.py
python scripts/maintenance/patch_b12_exact_origins.py --apply
python -m pytest -q tests/direction_a
python -m pytest -q
```

The patch refuses unrecognised versions of the B1.2 file. Applying it saves the original bytes in `archive/maintenance/extend_atlas_before_exact_origin_patch_<UTC>.py`.

## Run (no new Chronos-2 inference)

```bash
BASE=results/direction_a/b21_electricity_53
SHARD=results/direction_a/b21_shards/tirex2_4d_exact_v2.csv
python -m src.budgeted_forecasting.extend_atlas run-action \
  --base-cache "$BASE/forecast_cache.csv" \
  --base-manifest "$BASE/manifest.json" \
  --model tirex2 --context-hours 96 --output "$SHARD" --dry-run

# ONLY after dry-run passes:
python -m src.budgeted_forecasting.extend_atlas run-action \
  --base-cache "$BASE/forecast_cache.csv" \
  --base-manifest "$BASE/manifest.json" \
  --model tirex2 --context-hours 96 --output "$SHARD"

python -m src.budgeted_forecasting.extend_atlas merge \
  --base-cache "$BASE/forecast_cache.csv" \
  --shard "$SHARD" \
  --output-dir results/direction_a/b21_electricity_53_multimodel
```

## Caveats

- A passing dry run does not imply that TiRex-2 is installed or that its inference will succeed on the GPU.
- The successful B1.1 run reported 206 cases over 53 entities. The anticipated 318 was an incorrect assumption of full entity-by-time coverage. Check the per-entity/split origin count distribution from the cache before using case-level statistics.
- B1.1's global-time origin selection is still a **design limitation** for subsequent scaled experiments; the patch fixes exact reconstruction only. We should develop a new, independently frozen protocol with balanced per-entity coverage for final evaluation.
- All evaluation results in this cohort are exploratory, not untouched confirmatory findings.
