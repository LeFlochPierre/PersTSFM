# Phase 3.6 — Chronos-2 numerical parity diagnostic (diagnostic only)

The trained and unmerged-reloaded Chronos-2 LoRA models have identical parameter and buffer values, while their raw forward outputs disagree. PyTorch's GPU setting `matmul_precision='high'` can permit reduced internal precision, so this experiment compares **both models in the same process with both `high` and `highest` precision**, after an independent two-step training run. The script additionally compares saved vs loaded PEFT config fields and CUDA tensor devices/strides/contiguity.

This adds a standalone diagnostic; it **does not modify the model backend, replace the earlier result directories, score lag-1 targets, or access FEV official test labels**.

```bash
cd ~/RDS/home/projects/PersTSFM
conda activate PERTSFM_ENV
unzip -n PersTSFM_phase3_6_precision_diagnostic.zip
python -m pytest -q tests/maintenance/test_chronos2_precision_diag.py
set -o pipefail
python -m scripts.maintenance.diagnose_chronos2_precision \
  --config configs/phase3/capability_loop_smoke.json \
  2>&1 | tee results/phase3/precision_diagnostic.log
```

Check `GPU PRECISION PARITY`, `PEFT METADATA DIFFERENCES` and `TENSOR LAYOUT DIFFERENCES`. High-vs-highest parity is a diagnostic test, **not grounds to automatically accept a trained checkpoint**. If `highest` eliminates the mismatch, we can implement inference precision consistency and benchmark runtime costs. If it does not, we can localise the first discrepant layer or module. Do not loosen the existing threshold or continue the larger GPU sweep until resolved.

The diagnostic prints max-absolute output deviations and scaled maxima. These are **not changes in forecasting MAE**.
