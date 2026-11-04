# PersTSFM Phase 3.6 — strict unmerged LoRA checkpoint parity

## Diagnosis

Official Chronos-2 `Chronos2Pipeline.from_pretrained` checks for a PEFT adapter, calls `AutoPeftModel.from_pretrained`, and then **`merge_and_unload()`**. The prior capability audit compared the trained **unmerged** `PeftModel` against the official **merged** model. Merge-time numerical changes can cause prediction differences even if the adapter was saved correctly. The observed change after the previous eval-mode fix (scaled max difference 0.0017556) does not yet prove merging is the cause.

## Fix and falsifiable checks

1. Reload the checkpoint with official upstream `AutoPeftModel.from_pretrained`, retaining the **unmerged** PEFT structure.
2. Verify no missing/extra named parameter tensors, at least one shared LoRA tensor, and parameter max abs difference <= 1e-6.
3. Verify forecasts from the trained and unmerged-reloaded models agree to the original strict scaled maximum <= 1e-3. Check run-to-run jitter by forecasting twice on the same original model.
4. Load using the official **merged** Chronos-2 pipeline and record its scaled prediction drift for diagnosis, but **never** use that representation to satisfy the unmerged check.
5. After a verified reload, use the **unmerged, reloaded pipeline** for the lag-2/lag-1 forecasts of the capability experiment. Neither scores nor selection use future labels while deciding.
6. If weights or forecasts differ beyond the thresholds, write `capability_audit_failed.json` and reject that trial.

## Install

From the PersTSFM repository root, with the updated ZIP in that root:

```bash
unzip -n PersTSFM_phase3_6_unmerged_reload_fix.zip
python scripts/maintenance/install_phase36_unmerged_reload.py
python scripts/maintenance/install_phase36_unmerged_reload.py --apply
python -m pytest -q tests/phase2 tests/phase3
```

The patcher refuses to overwrite changes that do not match the previous Phase 3.6 eval-mode patch, stores source backups under `archive/patch_backups/`, and can be rerun without damage.

## GPU smoke, fresh destination

```bash
set -o pipefail
python -m scripts.phase3.adaptation_capability run \
  --config configs/phase3/capability_loop_smoke.json \
  --output-root results/phase3/capability_unmerged_smoke_v3 \
  2>&1 | tee results/phase3/capability_unmerged_smoke_v3.log
```

Look for `ADAPTATION AUDIT: lora ... weight_change=True reload=True`. Inspect `capability_audit.json`: `checkpoint_reload_representation`, `checkpoint_parameter_comparison`, `checkpoint_reload_relative_max_delta`, `repeated_original_scaled_delta`, and `official_merged_reload_scaled_delta`.

If it fails again, share the **new** `capability_audit_failed.json` (not the old merged-vs-unmerged diagnostic) and do not run the full pilot. Do not raise tolerances or reuse an incomplete failed run.

## Scientific caution

This modifies the Phase 3.6 capability backend only. The old Phase 3.3–3.5 LoRA results have not been re-evaluated with this audited pipeline, so avoid publication claims comparing them until we establish their inference equivalence. The full pilot is historical development evidence, not the official FEV test.
