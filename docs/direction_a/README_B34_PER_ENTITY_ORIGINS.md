# PersTSFM B3.4 — Optional per-entity origin selection

## Why this exists

`freeze --dataset solar --coverage-aware --entities 8` can fail with
`maximum eligible together: 3`. The global-time selector requires every
selected series to have forecast origins on the same global calendar slots.
This is unnecessarily restrictive when the source protocol samples origins
asynchronously across entities.

## New opt-in mode

`--per-entity-origins` searches for the latest **global temporal cutoff** at
which at least N entities have:

- Exactly `--calibration-origins` fully label-mature forecast origins before or
  at the cutoff (maturity at `target_end + one time step`).
- Exactly `--evaluation-origins` forecast origins at or after the cutoff.
- No duplicate entity/time or entity/evaluation-index pairs.

For each eligible entity it takes the last K mature calibration origins and the
first M evaluation origins. The N entities are chosen by a deterministic seeded
hash of the entity ID, using **protocol availability metadata only**, never
model errors, prediction values or outcome values. All selected calibration
labels mature before the earliest selected evaluation origin. The frozen
cohort stores each exact entity/origin pair, source hashes, the cutoff and
individual entity time slots.

**Importantly:** Unlike `--coverage-aware`, forecast timestamps do not have to
be identical across entities. This may change the calendar distribution of
cases; the selection mechanism is recorded in `cohort.json` and remains a
non-confirmatory development protocol. Inspect
`selection_details.evaluation_span_days` and each entity's dates. If origins
span dramatically different periods, do not claim this is a common-time
comparison. The same frozen cohort is still used for every model within solar.

This patch does not modify previous electricity, traffic, or solar cohorts. The
old default and `--coverage-aware` modes remain available.

## Apply (dry-run by default)

```bash
cd ~/RDS/home/projects/PersTSFM
conda activate PERTSFM_ENV
unzip -n PersTSFM_B34_per_entity_origins_patch.zip
python scripts/maintenance/patch_b34_per_entity_origins.py
python scripts/maintenance/patch_b34_per_entity_origins.py --apply
python -m pytest -q tests/direction_a
python -m pytest -q
```

`--apply` archives a copy of `capability_atlas.py` under
`archive/maintenance/b34_entity_TIMESTAMP/`; it checks exact code anchors and
refuses to edit an unexpected local version. Existing `coverage.py` is
untouched.

## Retry freeze (no GPU)

```bash
OUT=results/direction_a/b34_solar_8_per_entity
COHORT="$OUT/cohort.json"
python -m src.budgeted_forecasting.capability_atlas freeze \
  --dataset solar --entities 8 --calibration-origins 3 \
  --evaluation-origins 3 --max-context-hours 96 \
  --seed 20261102 --per-entity-origins --output-dir "$OUT"
```

Expect 48 balanced cases if it passes: 24 calibration and 24 evaluation. A
smaller actual number is an error, not silently accepted. If the protocol does
not offer 8 valid entities even under the pooled-cutoff approach, the selector
fails and reports the maximum feasible entity count. Do not weaken temporal
checks to force 8 entities.

Audit selected timeline before GPU work:

```bash
python - <<'PY'
import json, os
from pathlib import Path
p = json.loads((Path(os.environ.get('COHORT','results/direction_a/b34_solar_8_per_entity/cohort.json'))).read_text())
d = p['selection_details']
print('Selection:', p['selection'])
print('Boundary:', d['boundary'])
print('Entities:', len(d['selected_entities']), 'Cases:', len(p['cases']))
print('Eval span days:', d['evaluation_span_days'])
print('Cal labels available by:', d['latest_calibration_label_available'])
print('First evaluation:', d['earliest_evaluation'])
for entity, origins in sorted(d['per_entity'].items()):
    print(entity, 'cal=', origins['calibration_origins'], 'eval=', origins['evaluation_origins'])
PY
```

Then reuse the existing B3.4 `run-action` and `report` commands, with `OUT`
pointing at `b34_solar_8_per_entity`.

## Verification

Developed with 7 new selector/regression tests, plus the existing Direction A
tests in a dependency-light isolated project. Real solar raw files are **not**
available in that environment: only the user's RTX6 freeze can confirm whether
this cohort exists in the actual source protocol.
