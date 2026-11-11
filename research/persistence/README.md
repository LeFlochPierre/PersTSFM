# Persistence research workspace

This directory is reserved for the **feasibility study** in the [research question](../../docs/research_proposal/research_question.md) and [experiment plan](../../docs/research_proposal/experiment_plan.md). No persistence model or experiment is present yet.

The first pilot will freeze a population forecaster, hold out complete UCI electricity clients, reveal short chronological prefixes, and compare residual decay, empirical Bayes, Kalman/state-space correction, a generic horizon-aware gate, and a persistence-aware candidate on later outcomes. London Smart Meters is proposed as independent energy validation, subject to a data audit.

Potential reuse: dataset metadata/loading in `src/data/registry.py`, forecast interfaces in `src/models/adapters.py`, metrics in `src/evaluation/metrics.py`, and chronology/provenance patterns in `src/experiments/adaptation_benchmark/rolling.py`. The latter is historical Chronos-2 LoRA infrastructure and should not be modified merely to host this pilot. Generated outputs belong under ignored `results/` with source and configuration hashes and entity-level scores.
