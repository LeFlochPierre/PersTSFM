# Phase 3.6 — Chronos-2 attention implementation checkpoint parity diagnostic

**Purpose:** Distinguish different attention backends from PEFT or parameter serialization errors. The prior diagnostic established bit-identical parameters/buffers and reproducible models, but different raw forward outputs; switching PyTorch high/highest matmul precision did not eliminate the mismatch.

Official Chronos-2 source `chronos/chronos2/layers.py` chooses SDPA vs eager via `MHA.config._attn_implementation`. `Chronos2Model` deep-copies its config for encoder construction. Therefore comparing `model.config.to_dict()` alone does not prove the runtime attention implementations are equal.

The script trains two LoRA steps using the four-entity smoke config, loads the unmerged PEFT adapter, reports each attention module's backend and forecast parity, then forces **both models** to `eager`, then `sdpa` and repeats the same direct and end-to-end forecast comparisons. The script does not change existing files or official FEV test data; diagnostic in-memory model configurations are not saved or used for benchmark scores.

Run:

```bash
cd ~/RDS/home/projects/PersTSFM
conda activate PERTSFM_ENV
unzip -n PersTSFM_phase3_6_attention_diagnostic.zip
python -m pytest -q tests/maintenance/test_chronos2_attention_diag.py
python -m scripts.maintenance.diagnose_chronos2_attention --config configs/phase3/capability_loop_smoke.json 2>&1 | tee results/phase3/attention_diagnostic.log
```

If parity is restored under either common attention backend, update the research backend later to enforce the same attention implementation for base load, fine-tuning, checkpoint reload, and all evaluation. Only then repeat the strict capability smoke test. If not, capture the report and investigate hooks at the first divergent layer. Do not loosen tolerances.
