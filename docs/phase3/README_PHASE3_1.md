# PersTSFM Phase 3.1 — Adaptive candidate shortlist (exploratory baseline)

This script uses the *existing scored historical case catalogs*. It never runs a
foundation model, accesses the official FEV benchmark targets, or opens lag 1
when making a configuration-selection decision.

It requires `PersTSFM_phase3_0_probe_value.py` in the same repository root.

## Run

```bash
cd ~/RDS/home/projects/PersTSFM
conda activate PERTSFM_DATA_ENV
python -m pytest -q tests/phase3/test_phase3_adaptive_racing.py
mkdir -p results/phase3
set -o pipefail
python PersTSFM_phase3_1_adaptive_racing.py --repeats 200 --seed 42 \
    --budgets 4 8 16 32 2>&1 | tee results/phase3/phase3_1_adaptive_racing.log
```

## Hypothesis

Start with a very small sample (2 entities) on a medium context for each model.
Shortlist one or two model families; only evaluate additional context lengths for
those shortlisted models on 4, 8, 16, or 32 historical entities. Choose the
configuration with the best observed lag-2 loss and evaluate it on lag 1.

Compare against:
- A zero-validation-probe Chronos 2 or Toto 313M longest-context fixed default.
- Longest-only, middle-only, or full-grid searches with the *same number of entities*.
- Longest-only and full-grid searches with **no more configuration-case forecast calls**
  than the adaptive shortlist.

The budget counts unique configuration × entity × historical-origin forecasts.
**It is not GPU time**: model loading and long contexts can dominate.

The script reports mean and p90 later regret over repeated entity subsets.
The repeated subsets are correlated Monte Carlo trials on the same data, NOT
independent dataset replicates. There are only four dataset/source families.
No cross-family machine-learning generalization claims should be made.

## Outputs

`results/phase3/adaptive_racing/<UTC>/`:
- `trials.csv`: repeat-specific choices, later outcomes, and costs.
- `summary.csv`: grouped mean/p90 regret, call counts, and within-1%/5% success.
- `manifest.json`: reproducibility information and input catalog hashes.
