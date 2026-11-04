"""Synthetic smoke-test cache only; these are NOT scientific research results."""
from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .cache import FIELDS


def write_demo(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for dataset in ("synthetic_regular", "synthetic_variable"):
            for series in ("s0", "s1", "s2"):
                for idx in range(8):
                    origin = start + timedelta(days=idx)
                    horizon = 4
                    available = origin + timedelta(hours=horizon)
                    split = "calibration" if idx < 4 else "evaluation"
                    case_id = f"{dataset}/{series}/{idx:02d}"
                    for action, latency in (
                        ("cheap_short", 1.0), ("medium_short", 8.0), ("large_long", 24.0),
                    ):
                        for step in range(1, horizon + 1):
                            truth = 50 + step * 0.4 + int(series[-1]) * 2 + 2 * math.sin(idx / 2)
                            phase = -1 if (idx + int(series[-1])) % 2 else 1
                            if dataset == "synthetic_regular":
                                error = {
                                    "cheap_short": 0.5,
                                    "medium_short": 1.2,
                                    "large_long": 0.15 if phase > 0 else 1.9,
                                }[action]
                            else:
                                error = {
                                    "cheap_short": 2.9,
                                    "medium_short": 0.3 if phase > 0 else 1.4,
                                    "large_long": 0.55 if phase < 0 else 1.8,
                                }[action]
                            writer.writerow({
                                "dataset_id": dataset,
                                "series_id": series,
                                "case_id": case_id,
                                "split": split,
                                "origin_time": origin.isoformat(),
                                "label_available_at": available.isoformat(),
                                "action_id": action,
                                "step": step,
                                "truth": f"{truth:.8f}",
                                "prediction": f"{truth + phase * error:.8f}",
                                "latency_ms": f"{latency:.4f}",
                            })
