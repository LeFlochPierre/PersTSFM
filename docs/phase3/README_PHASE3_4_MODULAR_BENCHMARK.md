# PersTSFM Phase 3.4 — Modular cross-dataset adaptation benchmark (v0.1)

**Purpose:** Establish matched zero-shot context curves and Chronos-2 LoRA adaptation-response curves on multiple independent datasets. Separate *evidence gathering* from downstream learned policies. Initial focus: Chronos-2; backend can be extended without changing experiment selection, metrics or storage. This code adds files; it does not replace existing Phase 2/3.3 experiment runners.

## Package layout

| Module | Responsibility |
|---|---|
| `src/experiments/adaptation_benchmark/history.py` | Canonical timestamps, portable full-history FEV snapshots, causal lag2/lag1 windows, pre-lag2 training pools, hashes |
| `design.py` | JSON config validation, matched entity feasibility, trial grid, seed-mean validation selection |
| `features.py` | Strictly pre-lag2 seasonality, persistence, entropy and drift fingerprints for future predictive analyses |
| `metrics.py` | Entity-macro MAE and three-quantile pinball with established quantile rearrangement policy |
| `models.py` | Lazy GPU imports and Chronos-2 inference/PEFT LoRA backend |
| `artifacts.py` | Per-trial predictions cached with SHA256; atomic no-overwrite writes |
| `experiment.py` | Matched chronological protocol, resume, frozen selection, explicit lag1 scoring, printed summaries |
| `scripts/phase3/cross_dataset_adaptation.py` | Single CLI: `export`, `plan`, `run`, `score-later`, `report` |

This is an initial working framework; **GPU code is not validated on the user's workstation here**. Synthetic CPU tests cover the protocol and fake-backend execution. First run smoke tests below; report any Chronos-2 runtime/API errors rather than silently changing checkpoints or fit semantics.

## Install

Extract the zip at repo root (`unzip -n PersTSFM_phase3_4_modular_benchmark.zip`). No existing files should be overwritten. Verify:

```bash
conda activate PERTSFM_ENV
python -m pytest -q tests/phase3/test_adaptation_benchmark.py
python -m pytest -q tests/phase2 tests/phase3
```

## Export FEV preofficial histories: DATA environment

```bash
cd ~/RDS/home/projects/PersTSFM
conda activate PERTSFM_DATA_ENV
python -m scripts.phase3.cross_dataset_adaptation export \
  --task LOOP_SEATTLE_1H \
  --out results/phase3/context_saturation_inputs/loop_preofficial_full_ns_v2.npz
```

The exporter reads `task.get_window(0).get_input_data()` only from the pinned FEV v0.10.0 registry. It **does not access official test labels**. It creates a portable v2 ns snapshot identical in schema to the corrected 3.3C exporter. Existing M-DENSE v2 snapshot is reusable.

## Dry-run and GPU smoke: MODEL environment

```bash
conda activate PERTSFM_ENV
python -m scripts.phase3.cross_dataset_adaptation plan \
  --config configs/phase3/adaptation_loop_smoke.json

set -o pipefail
python -m scripts.phase3.cross_dataset_adaptation run \
  --config configs/phase3/adaptation_loop_smoke.json \
  2>&1 | tee results/phase3/phase34_loop_smoke.log
```

Save the actual directory printed by the runner. Then:

```bash
python -m scripts.phase3.cross_dataset_adaptation score-later --run <ACTUAL_RUN_DIRECTORY>
python -m scripts.phase3.cross_dataset_adaptation report --run <ACTUAL_RUN_DIRECTORY>
```

`score-later` does not retrain or load the model; it uses frozen forecasts and validates checksums. It is idempotent: if `later.csv` already exists, it prints the existing results without overwriting.

## Full pilot

Only after the 4-entity smoke test is validated, launch:

```bash
python -m scripts.phase3.cross_dataset_adaptation plan --config configs/phase3/adaptation_loop_pilot.json
python -m scripts.phase3.cross_dataset_adaptation run --config configs/phase3/adaptation_loop_pilot.json \
  2>&1 | tee results/phase3/phase34_loop_full.log
```

**Note:** the full pilot plans all 323 LOOP entities if all meet the entire grid and has 9 LoRA fits (3 budgets × 3 seeds), plus 7 zero-shot contexts. The 4-entity smoke intentionally includes only 20 and 100 step fine-tunes. Review GPU resource capacity; adjust batch size if needed (not after inspecting lag1 outcomes). For broader automation, create a batch JSON with several `experiments`; each receives its own run directory.

## Determinism and interruption/recovery

- Snapshot and cohort identities are hashed; snapshot time unit is nanoseconds regardless of pandas source resolution.
- Matched cohort is fixed for the *entire grid*. Missing data are rejected by default; optional `invalid_policy="drop"` records exclusions without looking at lag1 labels. Do not silently impute.
- Independent LoRA restarts are requested from the same pretrained model, and all random seeds are logged. The Chronos/PEFT implementation itself must be audited if stronger state-independence guarantees are required.
- LoRA strategy validation scores are **averaged across seeds before choosing a strategy**, not maximised over favourable seeds. Lag1 prediction is generated but never scored in `run`.
- Each finished trial is stored in `<RUN>/trials/<KEY>.npz` with checksum metadata. In case of interruption, resume with **the exact same config and snapshot**:

```bash
python -m scripts.phase3.cross_dataset_adaptation run \
  --config configs/phase3/adaptation_loop_pilot.json \
  --resume results/phase3/cross_dataset_adaptation/<ACTUAL_PARTIAL_RUN_DIRECTORY>
```

- Final results are `validation.csv` (per trial), `validation_strategies.csv` (per strategy), `validation_entity.csv`, `forecasts.npz`, `selection.json` (checksummed lock), plus `environment.json` (GPU and framework versions), `entity_features.csv` and `dataset_features.csv` computed from only pre-lag2 data, and, once explicitly revealed, `later.csv`, `later_trials.csv`, `later_entity.csv`.
- Costs measure local GPU training seconds and forecast-call seconds, but not end-to-end energy, CI, model download or amortised cold-start cost. First-call timings may include warm-up.

## Scientific limitations / checks before general claims

1. Snapshot lag1 observations are preofficial historical data, not an untouched official test set. **Previously inspected families (e.g., M-DENSE, LOOP) are development families.** Later meta-learning demands family-disjoint frozen test families.
2. The internal Chronos-2.fit sampling scheme has **not** been audited; passing 15,504 points as a pool does not prove that all were sampled. Inspect sampler and log training-window start positions before interpreting history effects causally.
3. LoRA learning rate is fixed at 1e-5 here; 0/20/100/500 steps are not proof that adaptation is always detrimental.
4. The pilot repeats seeds for uncertainty but independence of *entities* is not the same as independence of source datasets. Analyses should use paired entity bootstrap and dataset-family summaries.
5. Only Chronos-2 is implemented as a trainable backend. Other TSFM model families require a reproducible *training* implementation; zero-shot adapters must not be treated as LoRA-compatible automatically.
6. The current data QC uses complete matched inputs; lag1 target finiteness is only checked on the separate score-later pass. For datasets with missing lag1 targets, a predeclared evaluation mask or imputation protocol must be added BEFORE the study is frozen; do not choose an ad-hoc mask after seeing outcomes.
7. For M-DENSE C768, check original zero-shot lag2 MAE ~53.89950. `configs/phase3/adaptation_mdense_regression.json` includes this parity guard, but rerunning LoRA on M-DENSE adds limited new science.

## Next engineering milestones

- Add verified output-calibration baseline trained only on pre-lag2 residuals, with multiple rolling origins.
- Audit/specialise time-series sampling inside Chronos.fit and quantify actual unique window coverage per history pool.
- Factor additional trainable TSFM backends behind `forecast/fit_lora/reset`, with architecture-specific fit semantics and full reproducibility metadata.
- Add multiple chronological origin pairs and independent family-disjoint train/validation/test splits for the future policy model.
- Add safe and well-specified missing data support for ERCOT. Until then, use strict complete cases, not automatic interpolation.
