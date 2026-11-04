# PersTSFM Phase 3.7A — Adaptation Opportunity Atlas (development v1)

## Research goal

Search *systematically* for forecasting data regimes where a shared dataset-level Chronos-2 LoRA adapter produces **material** improvements versus a context-optimised zero-shot baseline. Explore predictors of adaptation opportunity rather than continuing to tune the previously inspected LOOP Seattle fold. This is **development screening**, not independent confirmation, final FEV test evaluation, or personalised-per-entity LoRA.

This patch **only adds** the atlas module, CLI, tests, and config; it does **not** modify Phase 3.6 training code. Requires your already installed Phase 3.6 Accelerate forward-wrapper fix (GPU smoke: `reload=True`).

## Protocol

- Predeclared main-screen variants: LoRA at 100 and 500 steps, learning rates 1e-5 and 3e-5, batch32, seed42 (one-seed exploratory screen). Inference batch8. Do not retrospectively modify these choices after inspecting later-period results.
- Per dataset, choose a feasible context at or below a **predeclared** preferred context using only history lengths/validity. Compare against shorter, matching, and (when feasible) longer zero-shot contexts. Long contexts are retained only if at least 80% of matched-context entities remain feasible.
- Use a shared, hash-sampled entity cohort per dataset; default up to 128 entities, never the first 128 lexicographically sorted sensors. Every candidate within the dataset is evaluated on identical entities and identical lag2 validation targets.
- Train the adapter on the pre-lag2 historical prefix, *strictly before validation*. Use the existing Phase 3.6 post-training eval, Accelerate autocast cleanup, sampled-crop audit, and save/reload equivalence gate. The runner generates later forecasts but never accesses later ground truth to lock the selection.
- Also score predeclared last-value and calendar-seasonal-naive validation baselines, for scientific sanity. These reference methods are **reported**, not part of the old Phase 3.6 selection lock.
- Write per-entity pre-lag2 fingerprints, validation rankings, paired per-entity improvement distribution, and *exploratory* paired bootstrap CIs. The CIs are **not adjusted for hyperparameter selection**, nor do they measure independent across-family uncertainty.
- Flag a discovery candidate only if the best LoRA strategy beats both the best zero-shot context and the best simple naive reference by at least 5% with >=16 matched entities. The threshold is a triage rule, not proof of significance or generalisation.
- Preserve all datasets (including negative results) in the atlas summary. Do not pick a positive development dataset and then call it a blind test.

## Install

Place `PersTSFM_phase3_7A_adaptation_opportunity_atlas.zip` in the PersTSFM repository root.

```bash
cd ~/RDS/home/projects/PersTSFM
conda activate PERTSFM_ENV
unzip -n PersTSFM_phase3_7A_adaptation_opportunity_atlas.zip
python -m pytest -q tests/phase2 tests/phase3
```

## First: dataset preflight (no GPU)

```bash
python -m scripts.phase3.adaptation_atlas preflight \
  --config configs/phase3/atlas_development_v1.json \
  2>&1 | tee results/phase3/atlas_preflight.log
```

The LOOP Seattle and M-DENSE snapshot locations are taken from the existing Phase 3 configs. The additional `solar`, `pems_bay`, `vitaldb`, `physionet2019`, and `hupa_ucm` entries are **candidate sources**, not a claim that their preofficial snapshots already exist or that these lowercase identifiers are exact FEV v0.10.0 task identifiers. Missing or incompatible sources will be listed explicitly. No fake data or silent horizon adjustments are made.

For datasets available as exact FEV v0.10.0 `dataset_config` names:

```bash
conda activate PERTSFM_DATA_ENV
python -m scripts.phase3.adaptation_atlas list-fev --filter vital
python -m scripts.phase3.adaptation_atlas list-fev --filter solar
# Use export-fev only after verifying the config's task name exactly matches the registry.
python -m scripts.phase3.adaptation_atlas export-fev \
  --config configs/phase3/atlas_development_v1.json --dataset solar
conda activate PERTSFM_ENV
```

**`export-fev` only calls the existing `export_fev_snapshot` which reads `task.get_window(0).get_input_data()`**. It never calls official future/test data. If a candidate does not exist as a FEV task, do not guess an alias or export full raw files containing test periods. Construct and vet a historical-only snapshot from the user's existing Phase 2 development data or omit the candidate. The original Phase 2 pilot may use differently named data than FEV.

If you edit the atlas config to update task names, source paths, or preferred contexts, make the change **before running experiments**. A completed atlas run records the exact generated Phase 3.6 config and source hashes; use a fresh output root for a changed design.

## GPU integration smoke first (not a scientific result)

This checks the complete atlas/Phase 3.6 interface on just 8 hash-selected sensors and 2 LoRA optimizer steps. It uses its own output root and never enters the scientific atlas summary.

```bash
conda activate PERTSFM_ENV
set -o pipefail
python -m scripts.phase3.adaptation_atlas run \
  --config configs/phase3/atlas_smoke_v1.json \
  --dataset loop_seattle \
  --output-root results/phase37a/atlas_smoke_v1 \
  2>&1 | tee results/phase3/atlas_smoke_v1.log
```

Expect the underlying Phase 3.6 `ADAPTATION AUDIT ... reload=True`, a `LOCKED` validation winner, and an `OPPORTUNITY loop_seattle` line. No later scoring is performed. If that passes, run the full development screen below in **a different output root**.

## Run the available development screen

Use the existing verified RTX6 environment:

```bash
set -o pipefail
python -m scripts.phase3.adaptation_atlas run \
  --config configs/phase3/atlas_development_v1.json \
  --output-root results/phase37a/atlas_dev_v1 \
  2>&1 | tee results/phase3/atlas_dev_v1.log
```

This runs every **READY** dataset and prints MISSING/INELIGIBLE entries; run one dataset with `--dataset loop_seattle` or `--dataset mdense`. Each dataset's Phase 3.6 runner prints a validation ranking and `LOCKED`. The atlas then prints the global summary across all finished datasets.

Default screen: 4 LoRA configs + 2–3 zero-shot configs per dataset, one seed, shared per-dataset adapter (not entity-specific). **Do not use Phase 3.6's old LOOP Seattle results as a direct numerical control**, since this atlas uses a fixed hash-selected 128-sensor cohort from the full historical snapshot rather than the old 323-sensor rolling fold. Compare only within a matched atlas cohort.

### Resumption and file guards

The atlas resumes a single interrupted Phase 3.6 dataset run when its trial cache is valid. A failed adapter attempt whose checkpoint directory already exists can require manual investigation; do not delete failed checkpoints without inspecting the audit. A completed locked dataset is skipped on reruns only when its plan, source hash, and generated config match exactly.

## Inspect the validation-only results

```bash
python -m scripts.phase3.adaptation_atlas summarize \
  --output-root results/phase37a/atlas_dev_v1
```

Reports `best_zero_mae`, `best_lora_mae`, `best_naive_mae`, `lora_gain_best_zero_pct`, `lora_gain_best_available_pct`, `entity_win_fraction`, exploratory paired CI, training seconds, and candidate flag. Saved under:

- `results/phase37a/atlas_dev_v1/atlas_validation_summary.csv`
- `results/phase37a/atlas_dev_v1/<dataset>_fingerprint.csv` and `<dataset>_fingerprint_entities.csv`
- `results/phase37a/atlas_dev_v1/derived_snapshots/<dataset>.npz` + provenance JSON
- `results/phase37a/atlas_dev_v1/runs/<dataset>/<timestamp>_.../validation.csv`
- `.../validation_entity.csv`, `.../atlas_naive_validation.csv`, `.../selection.json`, adapter audits and crop traces.

The atlas **does not train a meta-learner or choose policies using later-period results**. Fingerprints are pre-lag2 and currently exploratory descriptors.

## Later-period development diagnostics: separate explicit action

Only after validation configurations and selection locks are frozen:

```bash
python -m scripts.phase3.adaptation_atlas score-later \
  --dataset loop_seattle \
  --output-root results/phase37a/atlas_dev_v1
```

Later labels are historical preofficial input data, not independent test labels. These sources and periods have already influenced our broader research; use for robustness checks, not strong test claims.

## Scientific interpretation / stopping criteria

1. Look for gains >5% over the best context-optimised zero-shot and simple naive references, with broad entity-level improvements, not one outlier sensor.
2. Replicate promising regimes with seeds 42/43/44, multiple prespecified rolling origins, and negative controls. Use family-grouped holdouts for a later policy; don't train/test on sibling dataset slices as independent families.
3. If no substantial LoRA gains emerge, *report that result*. Consider shared domain vs per-entity LoRA, more LoRA steps, or other adaptation methods only in a preregistered second development sweep, not retrofitting the blind evaluation.
4. Benchmark against meaningful trained forecasters (ridge, gradient-boosted trees, or seasonal statistical models) in the confirmation stage; last-value/seasonal naive are only minimum-sanity controls.
5. Compare additional training GPU time and subsequent inference costs, including longer-context zero-shot inference. One promising win is not a paper by itself.

## Research limitations

- Some proposed datasets may fail due to missingness, short history, nonuniform cadence, or because the FEV v0.10.0 task does not exist. The atlas refuses to quietly make incomparable cohorts or use official targets.
- Large adapted-vs-zero-shot gains are *screening hypotheses*. Within-dataset strategy selection induces optimism; entity bootstrap CIs are not selection-adjusted. Subsequent evaluation must use uninspected dataset families and predetermined rules.
- Phase3.7A tests *shared global LoRA on a matched cohort*, not personalisation per entity. Further individual/domain adaptation is future work.
