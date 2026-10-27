# PersTSFM Phase 3.3A — safe rolling-history fix

The Phase 2 preofficial exporter saves the **last `max(contexts_steps)` observations** for each historical origin, not complete prefixes. For mature entities, lag-2 and lag-1 saved histories are equal-length, shifted rolling windows. The initial Phase 3.3A chronology checker incorrectly required `len(lag1)=len(lag2)+horizon` and prefix identity.

This installer replaces that check with strict timestamps/values overlap verification:

- `lag1 history == suffix(lag2 history) + lag2 future period` (where lag2 future observations are causally visible to the later lag1 forecast);
- no lag2 training samples after lag2 forecast origin;
- exact entity matching, lag2 future timestamp equality, monotonic time, and lag1 future starting later remain required.

It also prevents pretending to have 6,144 training steps when the Phase 2 export saved only 3,072. The new default is 3,072; requests beyond saved history fail clearly.

## Apply in PersTSFM repo root

```bash
conda activate PERTSFM_ENV
unzip -n PersTSFM_phase3_3a_rolling_fix.zip
python PersTSFM_phase3_3a_fix_rolling_history.py
python -m pytest -q tests/phase3/test_phase3_3a_chronos_lora.py tests/phase3/test_phase3_3a_rolling_history.py
python PersTSFM_phase3_3a_chronos_lora.py --context 768 --budgets 20 100 500
```

Then try the 2-step GPU smoke:

```bash
mkdir -p results/phase3
set -o pipefail
python PersTSFM_phase3_3a_chronos_lora.py \
  --context 768 --max-entities 4 --budgets 2 \
  --train-history-steps 3072 --train-batch-size 4 --execute \
  2>&1 | tee results/phase3/phase3_3a_smoke.log
```

After smoke passes, try the full run (30 entities) with `--train-history-steps 3072`, **not** `6144`.

The installer creates `PersTSFM_phase3_3a_chronos_lora.before_rolling_fix.py` before modification, refuses unknown source versions, and is idempotent. The regression tests use synthetic sliding and prefix histories and reject altered overlapping observations or invalid timestamps. No official test labels are used.
