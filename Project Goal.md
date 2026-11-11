# PersTSFM — Current Research Goal

**Stage:** feasibility and falsification, 11 November 2026. The earlier budget-aware TSFM goal is preserved in [docs/history/Project_Goal_budget_aware_2026-11-04.md](docs/history/Project_Goal_budget_aware_2026-11-04.md).

For a new persistent entity with little chronological history, determine whether its deviation from a population forecast will remain predictive at a requested horizon. Historical entities may help estimate persistence, but their dynamics may not transfer. Test whether learning the **future reliability of individual evidence** improves forecasts and limits harmful personalisation beyond well-fitted empirical Bayes, hierarchical state-space models, and generic horizon-aware mixtures.

First test this on held-out electricity clients using a frozen global predictor, multiple history lengths, and later outcomes. Treat a standard hierarchical model that matches the candidate as a successful falsification of the novelty hypothesis. Use controlled synthetic processes to distinguish persistent offsets from temporary shocks. Validate on an independent dataset only if the pilot passes.

The mathematical question, candidate methods, protocol, and go/no-go criteria are in [docs/research_proposal/](docs/research_proposal/). This records a research question, not an established result.
