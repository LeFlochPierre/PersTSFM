from pathlib import Path

import pandas as pd


ROOT = Path(
    "results/electricity/interventions"
)

OUT_DIR = Path(
    "results/electricity/intervention_synthesis"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS = [
    "chronos2",
    "timesfm3",
]

LONG_HOURS = [
    120,
    144,
    168,
    192,
    240,
    288,
    336,
    504,
    744,
]

RECENT_HOURS = 96

rows = []


for model in MODELS:

    for long_hours in LONG_HOURS:

        path = (
            ROOT
            / model
            / (
                f"recent_{RECENT_HOURS}h"
                f"_long_{long_hours}h"
            )
            / "summary.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        df = pd.read_csv(
            path
        )

        values = {
            row["intervention"]:
            float(
                row[
                    "mean_entity_mae"
                ]
            )
            for _, row in df.iterrows()
        }

        short = values[
            "short"
        ]

        same = values[
            "same_entity"
        ]

        other = values[
            "matched_other_affine"
        ]

        shuffled = values[
            "same_entity_day_shuffle"
        ]

        total_gain = (
            short
            - same
        )

        generic_gain = (
            short
            - other
        )

        identity_gap = (
            other
            - same
        )

        order_gap = (
            shuffled
            - same
        )

        if total_gain > 0:
            generic_fraction = (
                generic_gain
                / total_gain
            )

            identity_fraction = (
                identity_gap
                / total_gain
            )

            order_fraction = (
                order_gap
                / total_gain
            )

        else:
            generic_fraction = float(
                "nan"
            )

            identity_fraction = float(
                "nan"
            )

            order_fraction = float(
                "nan"
            )

        rows.append(
            {
                "model": model,
                "context_hours": (
                    long_hours
                ),
                "context_days": (
                    long_hours
                    / 24.0
                ),
                "short_mae": short,
                "same_entity_mae": (
                    same
                ),
                "matched_other_mae": (
                    other
                ),
                "shuffled_same_mae": (
                    shuffled
                ),
                "total_gain": (
                    total_gain
                ),
                "generic_gain": (
                    generic_gain
                ),
                "identity_gap": (
                    identity_gap
                ),
                "order_gap": (
                    order_gap
                ),
                "generic_fraction": (
                    generic_fraction
                ),
                "identity_fraction": (
                    identity_fraction
                ),
                "order_fraction": (
                    order_fraction
                ),
            }
        )


result = pd.DataFrame(
    rows
)

output = (
    OUT_DIR
    / "mechanism_curves.csv"
)

result.to_csv(
    output,
    index=False,
)


print("=" * 120)
print(
    "Electricity context mechanism curves"
)
print("=" * 120)
print()

display = result[
    [
        "model",
        "context_days",
        "short_mae",
        "same_entity_mae",
        "matched_other_mae",
        "shuffled_same_mae",
        "total_gain",
        "generic_gain",
        "identity_gap",
        "order_gap",
        "generic_fraction",
        "identity_fraction",
    ]
].copy()

for column in display.columns:

    if column not in [
        "model",
        "context_days",
    ]:
        display[column] = (
            display[column]
            .round(4)
        )

print(
    display.to_string(
        index=False
    )
)

print()
print(
    "Saved:",
    output,
)
