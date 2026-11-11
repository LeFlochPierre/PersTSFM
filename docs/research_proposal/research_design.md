# Three competing H1 designs

**Design status:** alternatives for feasibility testing, not an approved implementation specification. Shared input: frozen population forecast, residuals revealed before origin (t), horizon (h), and a training set of disjoint historical entities. Shared output: correction to the frozen forecast or a decision to retain it.

## A. Hierarchical persistence mixture (reference design)

Model residual (r_{it}=b_i+s_{it}+\epsilon_{it}), with (b_i\sim P_B) and (s_{i,t+1}=\rho_i s_{it}+\eta_{it}); learn a population distribution over (\rho_i). The posterior correction is (E[b_i\mid H_{it}]+E[\rho_i^h s_{it}\mid H_{it}]). Transparent and cheap, but this is a hierarchical state-space model, not a novel method. Its primary role is a **strong baseline** and controlled synthetic reference. An extension would need robust learning under misspecified, heterogeneous persistence, not just more flexible neural emissions.

## B. Direct horizon-specific evidence-value learner

Construct matched chronological training tasks across source entities. For each early-history summary (z_{it}), learn the prospective paired-risk difference (g_\theta(z_{it},h,a)\approx E[\ell_G-\ell_a\mid z_{it},h]), including uncertainty. Use only pre-origin information to choose adaptation strength. This directly targets future benefit, is backbone-agnostic and may outperform state estimation when latent states are poorly identified. Objection: Qin-style gain prediction or POSL with contextual weights may already implement it. Must compare to equal-capacity generic horizon gate and recent-error regression. Avoid using the same future target in feature construction and its benefit label.

## C. Robust persistence-set adaptation

Learn a source-informed set (\mathcal P_{it}) of plausible reversion rates/offsets. Choose correction (c) by minimising worst-case forecast risk, or retain the population forecast when plausible models disagree on benefit: (c^*=\arg\min_c\sup_{P\in\mathcal P_{it}}R_h(c;P)). This addresses source-to-target mismatch and can expose H2's ambiguity. Objection: robust Bayes and confidence-set decision rules are established; a distinctive guarantee requires a new tight persistence uncertainty set or rate. More theoretical and difficult to tune than B.

## Recommendation

**Test A and B first.** A is the hardest simple explanation; B is the clearest possible new objective. C should enter only if source mismatch causes a repeatable failure that ordinary EB and B cannot handle. If B does not beat A and a matched generic gate, pivot to a scientific characterisation of the limits rather than add complexity.

## H2 theoretical target

In a restricted Gaussian foil, compare (M_1:r_t=b+\epsilon_t) and (M_2:r_t=s_t+\epsilon_t,\ s_{t+1}=\rho s_t+\eta_t), chosen to be close in distribution for the first (n) observations but different at horizon (h). Seek a lower bound on correct action selection from total variation or KL between prefix laws, and an upper bound using (N) source trajectories under an explicit shared hyperdistribution. A standard two-point bound is **not automatically a new theorem**; the contribution would require a sharp, problem-specific dependence on (n,N,h) and mismatch that existing EB/meta-dynamics theory does not already give.
