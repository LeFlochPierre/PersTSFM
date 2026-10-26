# PersTSFM self-configuration Phase 1 core

Copy these files into the repository root, then run:

```bash
cp -r src/selfconfig ~/RDS/home/projects/PersTSFM/src/
cp -r tests/selfconfig ~/RDS/home/projects/PersTSFM/tests/
cp configs/selfconfig/phase1.yaml ~/RDS/home/projects/PersTSFM/configs/selfconfig/phase1.yaml
cd ~/RDS/home/projects/PersTSFM
pytest -q tests/selfconfig/test_core.py
```

The core currently implements:
- dataset fingerprinting from training data only;
- normalized context search space H,2H,4H,8H,16H;
- model-specific maximum context constraints (TiRex-2: 2048 steps);
- deterministic nnU-Net-style rule-based candidate pruning;
- plan serialization/printing;
- benchmark-table, selection, and regret/search-reduction utilities.

The next step is to wire each existing dataset loader into `build_plan()` and then connect the existing TSFM adapters/context-curve evaluator to the candidate configurations.
