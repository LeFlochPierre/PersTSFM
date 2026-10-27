# PersTSFM Phase 3.3A — Chronos-2 LoRA feasibility

This is a controlled **historical pilot**, NOT FEV official test evaluation, NOT a learned allocation policy, and NOT incremental training. The model is independently restarted from the same pretrained weights at each adaptation step budget. It uses existing frozen preofficial exports in `results/phase2/preofficial_calibration/fev_bench_v0_10_0/fev_0_10_0_044`.

## Temporal split

- **LoRA train:** 30 M-DENSE entity histories up to the start of lag2, truncated to the last 6,144 observations per entity by default.
- **Validation/selection:** lag2 next-horizon targets. Choose step budget by macro case MAE.
- **Later historical evaluation:** lag1 next-horizon targets, in a separate `--score-later` command after validation selection was saved.
- **Official FEV test:** never opened. **No cross-family generalisation claims** from this single-source pilot.

The lag1 *forecast input* naturally includes observations up to the lag1 cutoff (and can contain earlier lag2 targets); this is causally available history. No lag1 future targets enter the training or budget-selection stage. We explicitly check historical prefix overlap and prediction/label timestamps.

## Install

From repo root, with `PERTSFM_ENV` and working PEFT:

```bash
unzip -n PersTSFM_phase3_3a_chronos_lora.zip
python -m pytest -q tests/phase3/test_phase3_3a_chronos_lora.py
```

## Dry-run (no GPU)

```bash
python PersTSFM_phase3_3a_chronos_lora.py --context 768 --budgets 20 100 500
```

## Tiny smoke training (4 entities, 2 LoRA steps)

```bash
mkdir -p results/phase3
set -o pipefail
python PersTSFM_phase3_3a_chronos_lora.py \
  --context 768 --max-entities 4 --budgets 2 --train-batch-size 4 \
  --execute 2>&1 | tee results/phase3/phase3_3a_smoke.log
```

Find printed output directory, then run, for example:

```bash
python PersTSFM_phase3_3a_chronos_lora.py \
  --score-later results/phase3/lora_feasibility/<ACTUAL_RUN_DIRECTORY>
```

## Full 30-entity exploration (only after smoke passes)

```bash
python PersTSFM_phase3_3a_chronos_lora.py \
  --context 768 --max-entities 0 --budgets 20 100 500 \
  --train-history-steps 6144 --train-batch-size 8 \
  --execute 2>&1 | tee results/phase3/phase3_3a_mdense_full.log
```

Evaluate later **only after the validation decision is locked**:

```bash
python PersTSFM_phase3_3a_chronos_lora.py \
  --score-later results/phase3/lora_feasibility/<ACTUAL_FULL_RUN_DIRECTORY>
```

## Outputs

Run directory: `results/phase3/lora_feasibility/<UTC>_chronos2_lora_fev_0_10_0_044_e<N>/`

- `run_manifest.json` — configuration, training provenance, versions.
- `validation_summary.csv` — loss versus independent LoRA step budget and training wall seconds.
- `validation_case_metrics.csv` — per-entity lag2 metrics.
- `predictions.jsonl` — lag2 and lag1 predictions, **no lag1 labels**.
- `selection.json` — validation-chosen budget, SHA-256 of predictions and validation results.
- `adapters/lora_<steps>/` — official Chronos final checkpoints and trainer outputs (can be large).
- `later_summary.csv`, `later_case_metrics.csv` — written only by `--score-later`.

## Caveats

- The pilot independently repeats 20/100/500-step runs. Training time is for each **entire restart**, not cumulative continuation; checkpoint-reuse economics must be studied separately.
- We use upstream `Chronos2Pipeline.fit` and `src.models.adapters.Chronos2Model` for matching zero-shot forecast preparation. The original pretrained pipeline must remain unmodified.
- No official FEV targets, native SQL or 9-quantile probabilistic metric are used. `mean_pinball_3` is diagnostic.
- Comparisons across step counts are exploratory; later outcomes must not be used to retune. For publication, expand to independent families, multiple seeds/backbones, cheap residual baselines, and true incremental checkpoint-based allocation.
- Memory measurements are torch allocated peaks including model copies; not total GPU requirements. Time includes upstream checkpoint-saving cost. The output can consume significant disk space.
- Pilot LoRA default lr=1e-5 follows upstream guidance; the values 20/100/500 are not claimed optimal.
