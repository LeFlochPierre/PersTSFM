from pathlib import Path

import pandas as pd


ROOT = Path(
    "results/pems_bay/interventions"
)

OUT_DIR = Path(
    "results/pems_bay/intervention_synthesis"
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
    48,
    72,
    120,
    168,
    240,
    336,
    504,
    672,
]


rows = []


for model in MODELS:

    for long_hours in LONG_HOURS:

        path = (
            ROOT
            / model
            / (
                "recent_24h"
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
            for _, row in (
                df.iterrows()
            )
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
            identity_fraction = (
                identity_gap
                / total_gain
            )

            order_fraction = (
                order_gap
                / total_gain
            )

            generic_fraction = (
                generic_gain
                / total_gain
            )

        else:
            identity_fraction = float(
                "nan"
            )

            order_fraction = float(
                "nan"
            )

            generic_fraction = float(
                "nan"
            )

        rows.append(
            {
                "model": model,
                "context_hours": (
                    long_hours
                ),
                "context_days": (
                    long_hours / 24
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

result.to_csv(
    OUT_DIR
    / "mechanism_curves.csv",
    index=False,
)


print("=" * 110)
print(
    "PEMS-BAY context mechanism curves"
)
print("=" * 110)

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
        "identity_fraction",
        "order_fraction",
    ]
].copy()

for col in display.columns:
    if col not in [
        "model",
        "context_days",
    ]:
        display[col] = (
            display[col]
            .round(4)
        )

print()
print(
    display.to_string(
        index=False
    )
)

print()
print(
    "Saved:",
    OUT_DIR
    / "mechanism_curves.csv",
)
