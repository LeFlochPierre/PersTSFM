# PersTSFM Phase 3.3D — LoRA training-history saturation

**Question.** With Chronos-2 input context fixed to 768 hourly steps, does increasing the *pool of observations available for adaptation* from 3,072 to 6,144 to 15,504 alter the relationship between LoRA training compute and future forecasting loss?

This phase reuses the **corrected Phase 3.3C** historical input snapshot (`..._ns_v2.npz`) and its inference adapter. No FEV imports are needed in the GPU environment. **Do not use the old unversioned `.npz` snapshot.** The snapshot is the full pre-official input history; no official FEV evaluation labels are read.

## Setup

Place the distribution ZIP in the PersTSFM repository root and extract:

```bash
cd ~/RDS/home/projects/PersTSFM
conda activate PERTSFM_ENV
unzip -n PersTSFM_phase3_3d_lora_history.zip
python -m pytest -q tests/phase3/test_lora_history_saturation.py
```

**Requires:** corrected `scripts/phase3/context_saturation.py` (Phase 3.3C timestamp fix); existing `src/models/adapters.py`; PEFT and functioning Chronos-2 `fit` in `PERTSFM_ENV`.

## Dry run: full matched cohort, zero GPU training

```bash
python scripts/phase3/lora_history_saturation.py \
  --snapshot results/phase3/context_saturation_inputs/mdense_preofficial_full_ns_v2.npz \
  --context 768 --histories 3072 6144 15504 --budgets 20 100 500
```

Expected: 30 entities; 15,504 available pre-lag-2 values per entity; PASS dry run.

## Full-history LoRA smoke test

```bash
mkdir -p results/phase3
set -o pipefail
python scripts/phase3/lora_history_saturation.py \
  --snapshot results/phase3/context_saturation_inputs/mdense_preofficial_full_ns_v2.npz \
  --context 768 --max-entities 4 --histories 15504 --budgets 2 \
  --train-batch-size 4 --batch-size 4 --execute \
  2>&1 | tee results/phase3/phase3_3d_smoke.log
```

This checks that Chronos-2 can train on the full-length historical pool; it is **not** a scientific result. The dry run and smoke run do not alter earlier frozen results.

## Full M-DENSE 3 × 3 training-history / steps study

```bash
python scripts/phase3/lora_history_saturation.py \
  --snapshot results/phase3/context_saturation_inputs/mdense_preofficial_full_ns_v2.npz \
  --context 768 --max-entities 0 \
  --histories 3072 6144 15504 --budgets 20 100 500 \
  --train-batch-size 8 --batch-size 8 --learning-rate 1e-5 --seed 42 \
  --expected-zero-shot-mae 53.89950 --zero-shot-atol 0.02 \
  --execute 2>&1 | tee results/phase3/phase3_3d_mdense_full.log
```

The script separately trains nine LoRA configurations *from the identical pretrained checkpoint*, plus a zero-shot baseline. These are **independent restarts, not a continuing-training cost curve**. An increase in history does not increase the number of training steps; it changes the windows sampled by the Chronos-2 training dataset. The training input is always the last `history_steps` observations **before lag-2's validation targets**; it never includes lag-2 or lag-1 target values. At inference the context is always 768, and the same 30 entities are evaluated for every configuration.

Outputs under `results/phase3/lora_history_saturation/<timestamp>...`:

- `manifest.json` — cohort hashes, GPU, versions, hyperparameters, and protocol.
- `validation.csv` — lag-2 MAE and three-quantile diagnostic pinball, training wall seconds, PyTorch peak allocation, raw quantile crossings.
- `forecasts.npz` — lag-2 and lag-1 forecasts (not scored on lag 1 yet).
- `selection.json` — selected model/history/step budget chosen from lag-2 MAE, with SHA256 checks.
- `adapters/` — Chronos-2 training checkpoints/logs for each LoRA configuration.

Later score is intentionally separate:

```bash
python scripts/phase3/lora_history_saturation.py \
  --score-later results/phase3/lora_history_saturation/<PRINTED_RUN_DIRECTORY>
```

Replace the argument with the actual printed directory, not a literal placeholder. Later scores are locked against `manifest.json`, `validation.csv`, `forecasts.npz` and the original snapshot.

## Interpretation and limitations

We are probing one model, one inference context, one learning rate, and one random seed. The study establishes whether *training history availability* affects the adaptation response on M-DENSE. It does **not** establish family generalisation or a NeurIPS-worthy learned selection policy. The diagnostic pinball score uses `.1/.5/.9` quantiles (sorted to repair crossings), not official FEV SQL. Training seconds reported are wall clock for the model fit on this workstation, not transferable GPU-hours.

A useful parity check is that the new zero-shot lag-2 baseline at context 768 should approximately equal **53.89950 MAE** (the existing 30-entity M-DENSE Phase 3.3A/3.3C result). If not, **stop and diagnose** before interpreting LoRA differences.
