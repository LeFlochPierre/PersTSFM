from pathlib import Path

import numpy as np
import pandas as pd

from src.data.registry import context_label


OUT_DIR = Path(
    "results/context_synthesis"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


RUNS = [
    {
        "dataset": "Electricity",
        "model": "Chronos-2",
        "path": Path(
            "results/electricity/context/"
            "chronos2/summary.csv"
        ),
        "reference_hours": 96,
    },
    {
        "dataset": "Electricity",
        "model": "TimesFM 3",
        "path": Path(
            "results/electricity/context/"
            "timesfm3/summary.csv"
        ),
        "reference_hours": 96,
    },
    {
        "dataset": "PEMS-BAY",
        "model": "Chronos-2",
        "path": Path(
            "results/pems_bay/context/"
            "chronos2/summary.csv"
        ),
        "reference_hours": 24,
    },
    {
        "dataset": "PEMS-BAY",
        "model": "TimesFM 3",
        "path": Path(
            "results/pems_bay/context/"
            "timesfm3/summary.csv"
        ),
        "reference_hours": 24,
    },
    {
        "dataset": "HUPA-UCM",
        "model": "Chronos-2",
        "path": Path(
            "results/hupa_ucm/context/"
            "chronos2/summary.csv"
        ),
        "reference_hours": 24,
    },
    {
        "dataset": "HUPA-UCM",
        "model": "TimesFM 3",
        "path": Path(
            "results/hupa_ucm/context/"
            "timesfm3/summary.csv"
        ),
        "reference_hours": 24,
    },
]


def earliest_within(
    df,
    best_mae,
    fraction,
):
    threshold = (
        best_mae
        * (1.0 + fraction)
    )

    eligible = df[
        df["mean_entity_mae"]
        <= threshold
    ]

    if len(eligible) == 0:
        return np.nan

    return float(
        eligible[
            "context_hours"
        ].min()
    )


summary_rows = []
curve_frames = []


for run in RUNS:

    if not run["path"].exists():
        raise FileNotFoundError(
            run["path"]
        )

    df = pd.read_csv(
        run["path"]
    )

    required = {
        "context_hours",
        "context_label",
        "mean_entity_mae",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{run['path']} missing "
            f"columns: {sorted(missing)}"
        )

    df = (
        df
        .sort_values(
            "context_hours"
        )
        .reset_index(
            drop=True
        )
    )

    ref_mask = np.isclose(
        df["context_hours"],
        run["reference_hours"],
    )

    if ref_mask.sum() != 1:
        raise RuntimeError(
            f"{run['dataset']} / "
            f"{run['model']}: expected "
            "exactly one reference row."
        )

    reference_mae = float(
        df.loc[
            ref_mask,
            "mean_entity_mae",
        ].iloc[0]
    )

    best_index = (
        df[
            "mean_entity_mae"
        ].idxmin()
    )

    best = df.loc[
        best_index
    ]

    best_hours = float(
        best["context_hours"]
    )

    best_mae = float(
        best["mean_entity_mae"]
    )

    max_hours = float(
        df[
            "context_hours"
        ].max()
    )

    l1_hours = earliest_within(
        df,
        best_mae,
        0.01,
    )

    l2_hours = earliest_within(
        df,
        best_mae,
        0.02,
    )

    improvement_pct = (
        100.0
        * (
            reference_mae
            - best_mae
        )
        / reference_mae
    )

    right_censored = bool(
        np.isclose(
            best_hours,
            max_hours,
        )
    )

    summary_rows.append(
        {
            "dataset": (
                run["dataset"]
            ),
            "model": (
                run["model"]
            ),
            "reference_context_hours": (
                run[
                    "reference_hours"
                ]
            ),
            "reference_context_label": (
                context_label(
                    run[
                        "reference_hours"
                    ]
                )
            ),
            "reference_mae": (
                reference_mae
            ),
            "best_context_hours": (
                best_hours
            ),
            "best_context_label": (
                context_label(
                    best_hours
                )
            ),
            "best_mae": (
                best_mae
            ),
            "improvement_pct": (
                improvement_pct
            ),
            "l1_context_hours": (
                l1_hours
            ),
            "l1_context_label": (
                context_label(
                    l1_hours
                )
            ),
            "l2_context_hours": (
                l2_hours
            ),
            "l2_context_label": (
                context_label(
                    l2_hours
                )
            ),
            "max_context_hours": (
                max_hours
            ),
            "max_context_label": (
                context_label(
                    max_hours
                )
            ),
            "right_censored": (
                right_censored
            ),
        }
    )

    curve = df[
        [
            "context_hours",
            "context_label",
            "mean_entity_mae",
        ]
    ].copy()

    curve.insert(
        0,
        "model",
        run["model"],
    )

    curve.insert(
        0,
        "dataset",
        run["dataset"],
    )

    curve[
        "context_days"
    ] = (
        curve[
            "context_hours"
        ]
        / 24.0
    )

    curve[
        "reference_mae"
    ] = reference_mae

    curve[
        "mae_relative_to_reference"
    ] = (
        curve[
            "mean_entity_mae"
        ]
        / reference_mae
    )

    curve[
        "improvement_vs_reference_pct"
    ] = (
        100.0
        * (
            reference_mae
            - curve[
                "mean_entity_mae"
            ]
        )
        / reference_mae
    )

    curve_frames.append(
        curve
    )


summary = pd.DataFrame(
    summary_rows
)

curves = pd.concat(
    curve_frames,
    ignore_index=True,
)


summary_path = (
    OUT_DIR
    / "context_regime_summary.csv"
)

curves_path = (
    OUT_DIR
    / "normalized_context_curves.csv"
)


summary.to_csv(
    summary_path,
    index=False,
)

curves.to_csv(
    curves_path,
    index=False,
)


display = summary.copy()

for column in [
    "reference_mae",
    "best_mae",
    "improvement_pct",
]:
    display[column] = (
        display[column]
        .round(4)
    )


print("=" * 100)
print(
    "PersTSFM context-regime synthesis"
)
print("=" * 100)
print()

print(
    display[
        [
            "dataset",
            "model",
            "reference_context_label",
            "reference_mae",
            "best_context_label",
            "best_mae",
            "improvement_pct",
            "l1_context_label",
            "l2_context_label",
            "max_context_label",
            "right_censored",
        ]
    ].to_string(
        index=False
    )
)

print()
print("Saved:")
print(f"  {summary_path}")
print(f"  {curves_path}")
