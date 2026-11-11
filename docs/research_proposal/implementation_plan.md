# Conditional prototype plan — no implementation authorised

This document maps a future minimal prototype so the research design is reviewable. It is **not** an instruction to edit code now. Detailed task-level planning follows only after the design and novelty gates are accepted.

Reuse `src/data/registry.py` for existing electricity metadata, `src/experiments/adaptation_benchmark/rolling.py` for chronology/provenance concepts, `src/experiments/adaptation_benchmark/features.py` for pre-cutoff summaries, and `src/evaluation/metrics.py` for forecast diagnostics. Existing rolling adaptation is shared Chronos-2 LoRA, not a patient-only persistence method. Do not modify it to force this experiment into its interface.

If approved, keep the pilot self-contained under `src/experiments/persistence/` with modules for entity split/origin protocol, frozen global forecast interface, residual baselines, candidate B, evaluation and artifact provenance; configurations under `configs/persistence/`; narrow chronology/leakage tests under `tests/persistence/`. Use Python/PyTorch only where the candidate requires learning; EB/Kalman baselines can use standard numerical routines. Preserve existing experiments and Git changes.

Prototype order: (1) eligibility and split audit with no training, (2) synthetic process plus chronology checks, (3) frozen population predictions, (4) simple baselines including EB/Kalman, (5) candidate B, (6) locked prospective comparison. Tests must assert entity disjointness, target availability, no post-origin features, matched origins and baseline equivalence when correction strength is zero. Record seeds, source hashes, configuration, predictions, per-entity losses and compute. The first computational run should be a small smoke/pilot, never a full campaign.
