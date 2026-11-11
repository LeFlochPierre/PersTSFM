# PersTSFM — Research Recap

**Working research question**

Can we predict **when and how to adapt time-series foundation models**, using dataset characteristics and limited historical observations, while minimising forecasting error and computational expenditure?

We want to determine:

- **Which model?** Select an appropriate pretrained time-series foundation model.
- **How much context?** Identify the amount of historical input needed for competitive forecasting.
- **Should we fine-tune?** Determine whether zero-shot forecasting is sufficient or adaptation is worthwhile.
- **How much adaptation?** Choose the amount of training history and the fine-tuning budget.
- **When should we stop?** Avoid spending compute when additional evaluations or adaptation are unlikely to help.

## Two research contributions

**Paper A — Empirical study**

*When Should Time-Series Foundation Models Adapt?*

Conduct a systematic study across independent datasets and different model architectures, investigating context saturation, fine-tuning gains, training-history availability, adaptation budgets, and computational cost.

The goal is to discover reproducible relationships between dataset properties and the usefulness of adaptation, including meaningful negative results.

**Paper B — Methodological contribution**

*Learning to Adapt Time-Series Foundation Models Under Compute Budgets*

Develop a transferable, uncertainty-aware algorithm that predicts the value of additional context, model evaluation and fine-tuning, then allocates a computational budget intelligently.

The algorithm must outperform strong simple baselines under matched GPU budgets on previously unseen dataset families.

## What we currently know

Our M-DENSE experiments demonstrate that:

- Longer input contexts are not always better.
- The context selected on an earlier period may not transfer optimally to a later period.
- Zero-shot Chronos-2 outperformed our tested LoRA configurations.
- Increasing LoRA training history from 3,072 to 15,504 observations barely changed validation performance.

These findings motivate the research question but do not yet prove that dataset characteristics can predict the appropriate adaptation decisions.

## Next objective

Build a reproducible cross-dataset, cross-model collection of context and adaptation-response curves.

Evaluate whether dataset characteristics and inexpensive early experiments reliably predict which computational investments are worthwhile.

**Scientific success criterion:** Demonstrate that an algorithm can make better future forecasting decisions at matched computational cost, rather than simply identifying the best configuration retrospectively.