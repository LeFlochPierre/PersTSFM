from pathlib import Path
import json
import pandas as pd
import pytest
from scripts.phase2.summarise_development_pilot import build_analysis, read_outputs


def test_rank_reversal_and_best_context():
    rows = []
    for model, losses in {"a": (1.0, 0.8), "b": (1.1, 0.7)}.items():
        for c, loss in zip((24, 96), losses):
            rows.append({"dataset": "toy", "model": model, "context_steps": c,
                         "mean_entity_mae": loss, "n_cases": 8, "n_entities": 2})
    df = pd.DataFrame(rows)
    winners, gains, flips = build_analysis(df)
    assert winners.iloc[0]["model"] == "b"
    assert len(gains) == 2
    assert bool(flips.iloc[0]["rank_reversal"])


def test_case_mismatch_detected(tmp_path: Path):
    root = tmp_path / "dev"
    for model, entity in (("a", "X"), ("b", "Y")):
        d = root / "toy" / "context" / model
        d.mkdir(parents=True)
        pd.DataFrame([{"entity": entity, "eval_index": 0, "eval_origin": "2026-01-01"}]).to_csv(
            d / "case_selection.csv", index=False
        )
        pd.DataFrame([{"context_hours": 24, "context_steps": 24,
                       "mean_entity_mae": 1.0, "n_cases": 1, "n_entities": 1}]).to_csv(
            d / "summary.csv", index=False
        )
        (d / "phase2_run.json").write_text(json.dumps({
            "exit_code": 0, "subprocess_wall_seconds": 1.0,
        }))
    with pytest.raises(RuntimeError, match="different case selection"):
        read_outputs(root)
