# PersTSFM Phase 3.0 — Value of cheap probes and personalization headroom

## Research question
Before developing a learned/active configuration policy, test whether a few
historical forecasts from a subset of entities are informative about later
model × context performance. Assess whether entity-level personalization
has prospective value beyond a dataset-level configuration.

This script **does not** claim real-world GPU cost improvements. It reports
**configuration–case forecast calls** as a first, inexpensive cost proxy.
Model initialization, batching, memory and longer contexts can change runtime.

## Installation
From the PersTSFM repository root, unpack the zip or copy the standalone Python
file there. Only `pandas` and `numpy` are needed.

```
conda activate PERTSFM_DATA_ENV
python -m pytest -q tests/phase3/test_phase3_probe_value.py
set -o pipefail
python PersTSFM_phase3_0_probe_value.py --repeats 200 --seed 42 \
  2>&1 | tee results/phase3/phase3_0_probe_value.log
```

**Note:** Make sure `results/phase3` exists before using `tee`.

## Catalog handling
Automatically searches `results/phase2/cross_task_matrix/*/case_catalog.csv`
and matching `score_catalog.csv`, selecting for each task the snapshot with the
most complete model/context matrix (tie-break: higher K, then later folder name).
It never mixes unmatched K cohorts. Existing ERCOT K=2 (~18 configs) and K=3
(~6 configs) will resolve to K=2; the other three complete tasks use K=3.
Use `--catalog-dir PATH` (repeated) to restrict catalogs manually.

The script validates matched entity/lag/case signatures across model/context
configurations. It does not open FEV official evaluation labels.

## Protocol
- Chronological selection: lag 2 first, and lag 3 when available.
- Evaluation: lag 1 only (never used for selecting a configuration).
- Common randomly sampled entities across configurations; budgets by number
  of entities (2, 4, 8, 16, ...), and 1 or 2 historical origins as available.
- Candidate pools: full grid, one context per model (longest/middle/shortest),
  and fixed zero-probe baselines (Chronos2-longest and Toto313M-longest).
- Metrics: diagnostic MAE and three-quantile pinball, **not official FEV SQL**.
- Per-dataset comparison uses later regret relative to hindsight best on lag1.
- Personalization: blend normalized dataset prior (alpha=0) and normalized
  per-entity older historical loss (alpha=1). The entity-level lag1 oracle is
  a **hindsight upper bound**, not deployable.

## Outputs
`results/phase3/probe_value/<timestamp>/`
- `probe_trials.csv`: each replay run and later error
- `probe_summary.csv`: mean/p90 regret, fraction within 1%/5%, and correlation
- `personalization.csv`: prospective entity-level relative improvements
- `manifest.json`: source files plus sha256 hashes, selection/evaluation role,
  parameters, and scientific limitations

## Future work
1. Warm/cold model-cost benchmark and GPU memory tracking.
2. Adaptive policy with controlled acquisition (not just subsampling all
   configurations) and comparisons to random/racing/multi-fidelity BO.
3. Cross-family meta training and a completely held-out family benchmark.
4. Native fev SQL quantile consistency before main probabilistic claims.
