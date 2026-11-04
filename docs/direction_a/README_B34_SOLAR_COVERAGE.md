# B3.4 solar sparse-coverage patch

B3.0's default freeze selects hashed entities, then global calibration/evaluation
origin timestamps. On sparse solar protocols, an entity may be absent from one
partition. This is a metadata coverage issue, not a TSFM error.

This patch adds an **opt-in** `--coverage-aware` freeze method. It:

- Scans time blocks from latest to earliest and chooses the first with at least
  the requested number of entities represented in both partitions.
- Uses only the protocol's entity IDs, forecast origins, and target-end times
  (not raw observations, model outputs, predictions, or accuracy).
- Requires calibration targets to end **strictly before** the first evaluation
  origin, preserving pooled label maturity.
- Samples eligible entities deterministically using the supplied seed.
- Includes the selected IDs and time-block information in `cohort.json`.
- Leaves the previous default mode unchanged for reproducibility.

## Install

Extract ZIP in PersTSFM repository root (`unzip -n`). Run:

```
python scripts/maintenance/patch_b34_coverage.py
python scripts/maintenance/patch_b34_coverage.py --apply
python -m pytest -q tests/direction_a
python -m pytest -q
```

The installer checks the B3 source layout and creates a backup in the ignored
`archive/maintenance/` folder. If it cannot identify the expected B3.0 code,
it refuses to modify it. No existing cohorts or forecasting results change.

## Solar freeze

```
OUT=results/direction_a/b34_solar_8_coverage
COHORT="$OUT/cohort.json"
python -m src.budgeted_forecasting.capability_atlas freeze \
  --dataset solar --entities 8 --calibration-origins 3 \
  --evaluation-origins 3 --max-context-hours 96 \
  --seed 20261102 --coverage-aware --output-dir "$OUT"
```

The number of cases is a measured result, not necessarily 8 x 6. If no valid
block exists for eight entities, the selector reports the largest feasible
entity count. Reduce `--entities` (in a fresh output directory) or adjust
origin counts based on coverage, not model outcomes.

For a reproducible study, treat this as an exploratory development cohort.
Selection by availability can bias which entities are included; report its
criterion, eligible count, and excluded-entity coverage, then use independent
held-out datasets for confirmation.
