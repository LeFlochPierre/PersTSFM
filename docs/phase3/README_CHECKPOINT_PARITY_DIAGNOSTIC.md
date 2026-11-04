# Phase 3.6 — Trained vs unmerged-reloaded Chronos-2 forensic audit

This is a *diagnostic only*. It does not alter the Phase 3.6 research pipeline or relax its tolerance. It runs one fresh 2-step LoRA smoke fine-tune with the existing four-entity config, then compares the trained and reloaded models **simultaneously in memory**.

From the PersTSFM root in `PERTSFM_ENV`:

```bash
python -m scripts.maintenance.diagnose_chronos2_reload \
  --config configs/phase3/capability_loop_smoke.json \
  2>&1 | tee results/phase3/checkpoint_parity_diagnostic.log
```

The audit prints: named-parameter and buffer differences; parameter dtypes; direct-forward and forecast differences; PEFT LoRA scaling and active-adapter flags; Chronos core configs; GPU autocast/TF32 flags. Results are saved in a new output folder. **It does not access lag1 targets or official FEV test labels.**

Interpretation:
- Parameter values equal, dtype different: checkpoint reloaded with a different compute dtype; reproduce with an explicitly matched dtype.
- Parameter dtypes equal but buffers/config differ: investigate the first buffer/config mismatch. `named_parameters()` alone is not sufficient.
- Everything equal but direct forward differs: examine execution flags and module operations (TF32, SDPA backend, GPU kernel nondeterminism).
- Direct forward matches but `predict_df()` differs: examine pipeline preprocessing, quantile unrolling, frequency and context handling.

The diagnostic never claims the 0.0017556 difference is an MAE change. It is a scaled maximum forecast difference.
