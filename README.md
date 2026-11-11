# PersTSFM

PersTSFM is a feasibility-stage project on **horizon-aware reliable personalisation for cold-start time-series forecasting**. For a previously unseen entity with little history, we ask when its deviation from a population forecast will persist long enough to improve future forecasts. Methodological novelty is not established.

The first study will hold out entire electricity customers, reveal their history chronologically, and compare a frozen population forecaster with residual correction, empirical Bayes, hierarchical/Kalman correction, and a matched horizon-aware gate. If established methods explain the phenomenon, the hypothesis should be revised.

See the [research question](docs/research_proposal/research_question.md), [experiment plan](docs/research_proposal/experiment_plan.md), and [persistence workspace](research/persistence/README.md). The [previous project goal](docs/history/Project_Goal_budget_aware_2026-11-04.md) is historical documentation.

## Repository map

- `src/data/`, `src/models/`, `src/evaluation/`: existing loaders, forecast interfaces, metrics, and aggregation. Reuse requires protocol checks.
- `src/selfconfig/`, `src/budgeted_forecasting/`, `src/experiments/adaptation_benchmark/`: previous research implementations retained with tests for reproducibility.
- `scripts/phase2/`, `scripts/phase3/`, `scripts/direction_a/`: historical experiment entry points.
- `docs/research_proposal/`: current scientific design. Other phase documents record previous work.
- `data/`, `results/`, `archive/`: data, outputs, and archived material excluded from routine Git commits.

No persistence forecasting method has been implemented or validated yet.
