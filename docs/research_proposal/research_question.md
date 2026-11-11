# Research question: horizon-aware evidence persistence

**Status (11 November 2026):** approved research direction for feasibility, not a novelty or efficacy claim. H1 is primary; H2 supplies limits and falsification. No model has been implemented.

## Setting and estimand

Historical entities (i=1,ldots,N) provide chronological trajectories. A population predictor (f_G) is trained without test entities. At time (t), a new entity has only history (H_{i,t}). For horizon (h), an adaptation action (a) produces (f^a_{i,t,h}). Define prospective benefit

\[
G_{i,t,h}(a)=\mathbb E[\ell(Y_{i,t+h},f_G)-\ell(Y_{i,t+h},f^a_{i,t,h})\mid H_{i,t}],
\]

with the expectation under an explicitly stated population of new entities and future dynamics. Positive values favour adaptation. A realised future loss difference estimates this only noisily. Outcome availability, forecast timing, exogenous inputs and loss must be fixed before analysis.

**H1:** early residual evidence differs in future persistence across entities and horizons; historical entities can help estimate its future value and improve held-out-entity forecasts beyond matched-capacity generic gates and well-tuned hierarchical dynamic models.

**H2:** when a persistent offset and a temporary state induce nearly identical length-(n) histories, the sign of (G) at horizon (h) is not reliably identifiable without source-task assumptions. A useful result would quantify the dependence on (n,N,h), reversion rate, noise and source mismatch. Under unrestricted shift, no unconditional individual no-harm claim is possible.

## Established and potentially new

Empirical-Bayes short-panel forecasting [Liu et al. 2020](https://doi.org/10.3982/ECTA14952), optimal pooled/individual combinations [Pesaran et al. 2026](https://doi.org/10.3982/QE2589), online personal ensembles [Malenica et al. 2023](https://doi.org/10.1002/sim.9655), drift decomposition [Chen et al. 2025](https://doi.org/10.24963/ijcai.2025/542), and few-shot identifiable dynamics [Ye et al. 2026](https://proceedings.mlr.press/v306/ye26p.html) already cover much of this setting. The proposed distinction is *learning the future horizon-specific reliability of individual evidence*, with an explicit test of whether it improves on those mechanisms. Merely writing (f_G+\alpha_h b_i+\beta_h s_{it}) is classical state-space correction and is not a contribution.
