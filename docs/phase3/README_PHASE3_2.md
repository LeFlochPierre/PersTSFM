# PersTSFM Phase 3.2 — Controlled GPU Cost Pilot

This is an additive measurement utility. It **does not** run accuracy benchmarks, open the official FEV evaluation targets or write into previous Phase 2 forecasts. It uses frozen pre-official **input** histories, builds each model in its existing Conda environment, warms up inference, and measures repeated synchronized GPU inference time. Every job starts in a fresh Conda process to measure start-up overhead separately. Driver **dry-runs** by default.

## Install

Download ZIP to your repository root, then:

```bash
cd ~/RDS/home/projects/PersTSFM
conda activate PERTSFM_DATA_ENV
unzip -n PersTSFM_phase3_2_gpu_cost.zip
python -m pytest -q tests/phase3/test_phase3_gpu_cost.py
```

## Cost pilot

```bash
# Review the exact 6 jobs without executing.
python PersTSFM_phase3_2_gpu_cost.py \
  --tasks M_DENSE_1H \
  --models chronos2 moirai2_small \
  --contexts 96 768 3072 \
  --entities 16 --batch-size 8

# Execute two jobs as a smoke check.
mkdir -p results/phase3
set -o pipefail
python PersTSFM_phase3_2_gpu_cost.py \
  --tasks M_DENSE_1H \
  --models chronos2 moirai2_small \
  --contexts 96 768 3072 \
  --entities 16 --batch-size 8 \
  --warmup 2 --repeats 5 \
  --execute --max-jobs 2 \
  2>&1 | tee results/phase3/phase3_2_cost_pilot.log
```

Once pilot passes, run a measured cross-model grid (beware model loads and GPU use):

```bash
python PersTSFM_phase3_2_gpu_cost.py \
  --tasks M_DENSE_1H LOOP_SEATTLE_1H hospital_admissions_1D \
  --models chronos2 timesfm3 tirex2 toto2_22m toto2_313m moirai2_small \
  --contexts 96 384 768 1536 3072 \
  --entities 16 --batch-size 8 \
  --warmup 2 --repeats 5
```

Review the feasibility plan before adding `--execute`; many jobs load models multiple times. The two small-feasibility datasets have at most 8 and 30 entities, respectively. Excluded contexts are labelled.

## Outputs and methodological limitations

Each execution writes under `results/phase3/gpu_cost/<UTC timestamp>/`, including `plan.json`, `logs/`, per-job JSON and `measurements.csv`. Columns include end-to-end cold-process wall time (Conda startup + history loading + model setup + warmup/repeats + writing), model import-and-load time, median warm inference time for a fixed number of cases, throughput, and PyTorch peak allocated/reserved memory.

**Do not interpret** forecast-call reductions as GPU-time reductions. PyTorch memory metrics do not include all non-PyTorch allocations. The cold-process timing is an entire *benchmark job*, not a single cold forecast. No energy or real deployment latency is measured. Same cases used repeatedly to reduce timing noise; ensure the GPU is otherwise idle. Bench is strictly K=1 historical origins for cost only, not a prospective model-selection evaluation.

In future, measure different batch sizes, workloads, repeated deployments, native FEV scoring and model loading/deployment amortisation before making cost-effectiveness claims.
