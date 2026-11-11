# Research gaps and initial novelty verdict

Date: 11 November 2026. Status: **initial verdict; no algorithm selected**.

## Already occupied

Global-versus-personal choice, cross-period online/offline mixtures, bounded residual adapters, hierarchical partial pooling, online calibration, and anytime model comparison each have direct prior work. Applying one with another backbone, threshold or dataset is not a NeurIPS/ICML/ICLR methodological contribution.

## Candidate gaps and falsifiers

1. **Action-specific future benefit at cold start** (plausible unresolved gap, substantial overlap with Qin et al.). Formal target: for new entity `i` and action `a`, predict the conditional future paired loss difference `E[L(pop)-L(a) | early history, source cohort]`, rather than current model score. Closest threat is Qin et al.'s gain calculation and switching. Falsifier: its method, hierarchical posterior risk, or recent-error rule already predicts later paired gain as well as a learned cross-entity estimator on held-out people.
2. **Cross-entity transfer under drift with conservative adaptation** (potentially distinct mechanism requiring validation). Learn a prior over entity-specific benefit trajectories and update it prequentially; abstain when estimated gain is unreliable. Closest threats: MOOE and UAI 2026 certificate-driven residual calibration. Falsifier: standard hierarchical shrinkage plus online expert aggregation matches the entire accuracy–harm–cost frontier. Any theory would require explicit boundedness/martingale or restricted-shift assumptions; arbitrary future drift prevents unconditional individual no-harm claims.
3. **Limits of predictability of personalisation benefit** (scientific discovery, not necessarily algorithm). Quantify how much early observations can predict later paired gain after separating patient difficulty from treatment benefit. Closest threat is existing global/local comparison and Qin et al. Falsifier: signs are almost uniform, or variance is largely measurement noise and no transferable signal survives independent cohorts.
4. **Sequential value of acquiring individual evidence** (substantial overlap with online learning). Decide when to spend labels/compute on an individual rather than immediately adapt. Closest threats: online experts and sequential model confidence sets. Falsifier: standard e-process/aggregation methods plus a fixed schedule give equal results at matched cost.

## Reviewer test

A paper must show a result that cannot be replicated by applying Qin et al., a hierarchical model, local forecast pools, MOOE, or certificate-driven residual calibration with a tuned threshold. It must use prospective patient/entity holdouts and at least one independent source family. Millions of windows within a few people do not establish cross-person learning. A generic gate and an unsupported safety guarantee would be rejected.

## Initial verdict

The **broad proposal is not novel**. Gap 1 is worth a low-cost falsification study, but it is not yet a defensible paper claim. Gap 2 might support a method only if cross-entity information provides an empirically and theoretically distinct benefit beyond strong shrinkage/online baselines. No direction currently clears the novelty bar without full-text comparison and prospective evidence.
