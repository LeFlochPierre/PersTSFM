from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

DATASETS = [
    {
        "dataset": "Electricity",
        "model": "Chronos-2",
        "path": Path(
            "results/context_threshold/"
            "chronos2_context_summary.csv"
        ),
        "context_col": "context_days",
        "label_col": "context_label",
        "metric_col": "mean_mae",
        "reference_context": 4,
        "context_unit": "days",
    },
    {
        "dataset": "Electricity",
        "model": "TimesFM 3",
        "path": Path(
            "results/context_threshold/"
            "timesfm3_context_summary.csv"
        ),
        "context_col": "context_days",
        "label_col": "context_label",
        "metric_col": "mean_mae",
        "reference_context": 4,
        "context_unit": "days",
    },
    {
        "dataset": "PEMS-BAY",
        "model": "Chronos-2",
        "path": Path(
            "results/pems_bay/context/"
            "chronos2_pems_bay_context_summary.csv"
        ),
        "context_col": "context_hours",
        "label_col": "context_label",
        "metric_col": "mean_mae",
        "reference_context": 24,
        "context_unit": "hours",
    },
    {
        "dataset": "PEMS-BAY",
        "model": "TimesFM 3",
        "path": Path(
            "results/pems_bay/context/"
            "timesfm3_pems_bay_context_summary.csv"
        ),
        "context_col": "context_hours",
        "label_col": "context_label",
        "metric_col": "mean_mae",
        "reference_context": 24,
        "context_unit": "hours",
    },
    {
        "dataset": "HUPA-UCM",
        "model": "Chronos-2",
        "path": Path(
            "results/hupa_ucm/context/"
            "chronos2_hupa_context_summary.csv"
        ),
        "context_col": "context_hours",
        "label_col": "context_label",
        "metric_col": "mean_participant_mae",
        "reference_context": 24,
        "context_unit": "hours",
    },
    {
        "dataset": "HUPA-UCM",
        "model": "TimesFM 3",
        "path": Path(
            "results/hupa_ucm/context/"
            "timesfm3_hupa_context_summary.csv"
        ),
        "context_col": "context_hours",
        "label_col": "context_label",
        "metric_col": "mean_participant_mae",
        "reference_context": 24,
        "context_unit": "hours",
    },
]


OUT_DIR = Path(
    "results/context_synthesis"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_PATH = (
    OUT_DIR
    / "context_regime_summary.csv"
)

CURVES_PATH = (
    OUT_DIR
    / "normalized_context_curves.csv"
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def to_days(
    value,
    unit,
):

    if unit == "days":
        return float(value)

    if unit == "hours":
        return float(value) / 24.0

    raise ValueError(
        f"Unknown context unit: {unit}"
    )


# ---------------------------------------------------------------------
# Analyse curves
# ---------------------------------------------------------------------

summary_rows = []
curve_rows = []


for spec in DATASETS:

    path = spec["path"]

    if not path.exists():
        print(
            f"WARNING: missing {path}"
        )
        continue

    df = pd.read_csv(
        path
    )

    context_col = spec[
        "context_col"
    ]

    label_col = spec[
        "label_col"
    ]

    metric_col = spec[
        "metric_col"
    ]


    # -----------------------------------------------------------------
    # Some older result files, especially Electricity, do not contain
    # an explicit context_label column. Reconstruct it consistently.
    # -----------------------------------------------------------------

    if label_col not in df.columns:

        def make_context_label(value):

            value = float(value)

            if spec["context_unit"] == "days":

                if value.is_integer():
                    return f"{int(value)}d"

                return f"{value:g}d"

            if spec["context_unit"] == "hours":

                if (
                    value >= 24
                    and value % 24 == 0
                ):

                    days = int(
                        value // 24
                    )

                    return f"{days}d"

                if value.is_integer():
                    return f"{int(value)}h"

                return f"{value:g}h"

            raise ValueError(
                "Unknown context unit: "
                f"{spec['context_unit']}"
            )


        df[label_col] = (
            df[
                context_col
            ]
            .apply(
                make_context_label
            )
        )


    df = (
        df
        .sort_values(
            context_col
        )
        .reset_index(drop=True)
    )

    contexts = df[
        context_col
    ].to_numpy(
        dtype=float
    )

    maes = df[
        metric_col
    ].to_numpy(
        dtype=float
    )

    # -------------------------------------------------------------
    # Reference
    # -------------------------------------------------------------

    reference_mask = (
        df[
            context_col
        ]
        == spec[
            "reference_context"
        ]
    )

    if (
        reference_mask.sum()
        != 1
    ):
        raise RuntimeError(
            f"{spec['dataset']} / "
            f"{spec['model']}: "
            "reference context not unique."
        )

    reference_row = df.loc[
        reference_mask
    ].iloc[0]

    reference_mae = float(
        reference_row[
            metric_col
        ]
    )

    # -------------------------------------------------------------
    # Best observed
    # -------------------------------------------------------------

    best_idx = int(
        np.argmin(
            maes
        )
    )

    best_row = df.iloc[
        best_idx
    ]

    best_context = float(
        best_row[
            context_col
        ]
    )

    best_mae = float(
        best_row[
            metric_col
        ]
    )

    maximum_context = float(
        contexts.max()
    )

    right_censored = bool(
        best_context
        == maximum_context
    )

    # -------------------------------------------------------------
    # Earliest context within 1% and 2% of best
    # -------------------------------------------------------------

    one_percent_threshold = (
        best_mae * 1.01
    )

    two_percent_threshold = (
        best_mae * 1.02
    )

    within_1 = df[
        df[
            metric_col
        ]
        <= one_percent_threshold
    ]

    within_2 = df[
        df[
            metric_col
        ]
        <= two_percent_threshold
    ]

    earliest_1 = (
        within_1.iloc[0]
    )

    earliest_2 = (
        within_2.iloc[0]
    )

    earliest_1_context = float(
        earliest_1[
            context_col
        ]
    )

    earliest_2_context = float(
        earliest_2[
            context_col
        ]
    )

    # -------------------------------------------------------------
    # Improvement
    # -------------------------------------------------------------

    absolute_gain = (
        reference_mae
        - best_mae
    )

    relative_gain = (
        100.0
        * absolute_gain
        / reference_mae
    )

    # -------------------------------------------------------------
    # Normalized gain curve
    #
    # 0 = reference performance
    # 1 = best observed performance
    # -------------------------------------------------------------

    if absolute_gain > 0:

        gain_fraction = (
            reference_mae
            - maes
        ) / absolute_gain

    else:

        gain_fraction = np.full(
            len(df),
            np.nan,
        )

    for i, row in df.iterrows():

        context_value = float(
            row[
                context_col
            ]
        )

        curve_rows.append(
            {
                "dataset": (
                    spec["dataset"]
                ),
                "model": (
                    spec["model"]
                ),
                "context": (
                    context_value
                ),
                "context_unit": (
                    spec[
                        "context_unit"
                    ]
                ),
                "context_days": (
                    to_days(
                        context_value,
                        spec[
                            "context_unit"
                        ],
                    )
                ),
                "context_label": (
                    row[
                        label_col
                    ]
                ),
                "mae": float(
                    row[
                        metric_col
                    ]
                ),
                "reference_mae": (
                    reference_mae
                ),
                "best_mae": (
                    best_mae
                ),
                "gain_fraction": float(
                    gain_fraction[i]
                ),
            }
        )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    summary_rows.append(
        {
            "dataset": (
                spec["dataset"]
            ),

            "model": (
                spec["model"]
            ),

            "reference_context": (
                spec[
                    "reference_context"
                ]
            ),

            "reference_context_days": (
                to_days(
                    spec[
                        "reference_context"
                    ],
                    spec[
                        "context_unit"
                    ],
                )
            ),

            "reference_mae": (
                reference_mae
            ),

            "best_context": (
                best_context
            ),

            "best_context_days": (
                to_days(
                    best_context,
                    spec[
                        "context_unit"
                    ],
                )
            ),

            "best_context_label": (
                best_row[
                    label_col
                ]
            ),

            "best_mae": (
                best_mae
            ),

            "relative_improvement_percent": (
                relative_gain
            ),

            "earliest_within_1pct_context": (
                earliest_1_context
            ),

            "earliest_within_1pct_days": (
                to_days(
                    earliest_1_context,
                    spec[
                        "context_unit"
                    ],
                )
            ),

            "earliest_within_1pct_label": (
                earliest_1[
                    label_col
                ]
            ),

            "earliest_within_2pct_context": (
                earliest_2_context
            ),

            "earliest_within_2pct_days": (
                to_days(
                    earliest_2_context,
                    spec[
                        "context_unit"
                    ],
                )
            ),

            "earliest_within_2pct_label": (
                earliest_2[
                    label_col
                ]
            ),

            "maximum_tested_context_days": (
                to_days(
                    maximum_context,
                    spec[
                        "context_unit"
                    ],
                )
            ),

            "best_at_maximum_tested_context": (
                right_censored
            ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)

curves = pd.DataFrame(
    curve_rows
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

curves.to_csv(
    CURVES_PATH,
    index=False,
)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

print()
print("=" * 100)
print("PersTSFM cross-dataset context regime synthesis")
print("=" * 100)

display = summary[
    [
        "dataset",
        "model",
        "reference_mae",
        "best_context_label",
        "best_mae",
        "relative_improvement_percent",
        "earliest_within_1pct_label",
        "earliest_within_2pct_label",
        "maximum_tested_context_days",
        "best_at_maximum_tested_context",
    ]
].copy()

for col in [
    "reference_mae",
    "best_mae",
    "relative_improvement_percent",
    "maximum_tested_context_days",
]:
    display[col] = (
        display[col]
        .round(3)
    )

print()
print(
    display.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("Effective-context scale in days")
print("=" * 100)

effective = summary[
    [
        "dataset",
        "model",
        "best_context_days",
        "earliest_within_1pct_days",
        "earliest_within_2pct_days",
        "best_at_maximum_tested_context",
    ]
].copy()

for col in [
    "best_context_days",
    "earliest_within_1pct_days",
    "earliest_within_2pct_days",
]:
    effective[col] = (
        effective[col]
        .round(2)
    )

print()
print(
    effective.to_string(
        index=False
    )
)


print()
print("Saved:")
print(f"  {SUMMARY_PATH}")
print(f"  {CURVES_PATH}")
