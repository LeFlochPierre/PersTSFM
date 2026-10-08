from pathlib import Path
import json

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SPLIT_PATH = Path(
    "data/processed/electricity/electricity_entity_splits.csv"
)

OUT_DIR = Path(
    "data/processed/electricity"
)

FREQUENCY_MINUTES = 15

CONTEXT_DAYS = 4
FORECAST_DAYS = 1

PERSONALIZATION_DAYS = [
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
]

N_EVAL_ORIGINS = 12
EVAL_SPACING_DAYS = 7

EVALUATION_SPLITS = [
    "validation",
    "heldout_test",
]


# ---------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------

freq = pd.Timedelta(
    minutes=FREQUENCY_MINUTES
)

context_steps = int(
    CONTEXT_DAYS
    * 24
    * 60
    / FREQUENCY_MINUTES
)

forecast_steps = int(
    FORECAST_DAYS
    * 24
    * 60
    / FREQUENCY_MINUTES
)

max_personalization_days = max(
    PERSONALIZATION_DAYS
)

context_span = (
    context_steps - 1
) * freq

forecast_span = (
    forecast_steps
) * freq


# ---------------------------------------------------------------------
# Load split table
# ---------------------------------------------------------------------

df = pd.read_csv(
    SPLIT_PATH
)

df["first_nonzero_timestamp"] = pd.to_datetime(
    df["first_nonzero_timestamp"]
)

df["last_nonzero_timestamp"] = pd.to_datetime(
    df["last_nonzero_timestamp"]
)

eval_df = df[
    df["split"].isin(
        EVALUATION_SPLITS
    )
].copy()


# ---------------------------------------------------------------------
# Build evaluation origins
# ---------------------------------------------------------------------

rows = []

for _, row in eval_df.iterrows():

    entity = row["entity"]
    split = row["split"]

    first_active = row[
        "first_nonzero_timestamp"
    ]

    last_active = row[
        "last_nonzero_timestamp"
    ]

    # Latest possible origin such that the complete
    # 1-day target is still within the active interval.
    latest_origin = (
        last_active
        - pd.Timedelta(
            days=FORECAST_DAYS
        )
    )

    entity_origins = []

    for k in range(
        N_EVAL_ORIGINS
    ):

        origin = (
            latest_origin
            - pd.Timedelta(
                days=(
                    k
                    * EVAL_SPACING_DAYS
                )
            )
        )

        entity_origins.append(
            origin
        )

    # Store chronologically.
    entity_origins = sorted(
        entity_origins
    )

    for eval_index, origin in enumerate(
        entity_origins,
        start=1,
    ):

        context_end = origin

        context_start = (
            context_end
            - context_span
        )

        target_start = (
            origin
            + freq
        )

        target_end = (
            origin
            + forecast_span
        )

        # For n = 64 personalization days,
        # the earliest previous daily forecast
        # origin is 64 days before this origin.
        earliest_personal_origin = (
            origin
            - pd.Timedelta(
                days=max_personalization_days
            )
        )

        earliest_personal_context_start = (
            earliest_personal_origin
            - context_span
        )

        # Safety conditions.
        if (
            earliest_personal_context_start
            < first_active
        ):
            raise RuntimeError(
                f"{entity}: insufficient history "
                f"for evaluation origin {origin}. "
                f"Need data from "
                f"{earliest_personal_context_start}, "
                f"but active history begins "
                f"{first_active}."
            )

        if target_end > last_active:
            raise RuntimeError(
                f"{entity}: forecast target "
                f"extends beyond active history."
            )

        available_history_days = (
            origin
            - first_active
        ).total_seconds() / 86400.0

        rows.append(
            {
                "entity": entity,
                "split": split,
                "eval_index": eval_index,
                "eval_origin": origin,
                "context_start": context_start,
                "context_end": context_end,
                "target_start": target_start,
                "target_end": target_end,
                "first_active_timestamp": (
                    first_active
                ),
                "last_active_timestamp": (
                    last_active
                ),
                "earliest_personal_origin": (
                    earliest_personal_origin
                ),
                (
                    "earliest_personal_"
                    "context_start"
                ): (
                    earliest_personal_context_start
                ),
                "available_history_days": (
                    available_history_days
                ),
                "active_span_days": (
                    row["active_span_days"]
                ),
            }
        )


manifest = pd.DataFrame(
    rows
)


# ---------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------

expected_entities = len(
    eval_df
)

expected_rows = (
    expected_entities
    * N_EVAL_ORIGINS
)

assert len(manifest) == expected_rows

assert not manifest.duplicated(
    [
        "entity",
        "eval_index",
    ]
).any()

counts = (
    manifest
    .groupby("entity")
    .size()
)

assert (
    counts
    == N_EVAL_ORIGINS
).all()

assert (
    manifest[
        "target_end"
    ]
    <= manifest[
        "last_active_timestamp"
    ]
).all()

assert (
    manifest[
        "earliest_personal_context_start"
    ]
    >= manifest[
        "first_active_timestamp"
    ]
).all()


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

manifest_path = (
    OUT_DIR
    / "electricity_evaluation_origins.csv"
)

manifest.to_csv(
    manifest_path,
    index=False,
)


metadata = {
    "frequency_minutes": (
        FREQUENCY_MINUTES
    ),
    "context_days": (
        CONTEXT_DAYS
    ),
    "context_steps": (
        context_steps
    ),
    "forecast_days": (
        FORECAST_DAYS
    ),
    "forecast_steps": (
        forecast_steps
    ),
    "personalization_days": (
        PERSONALIZATION_DAYS
    ),
    "personalization_unit": (
        "completed previous 24-hour "
        "forecast windows from the "
        "same entity"
    ),
    "n_evaluation_origins_per_entity": (
        N_EVAL_ORIGINS
    ),
    "evaluation_spacing_days": (
        EVAL_SPACING_DAYS
    ),
    "evaluation_splits": (
        EVALUATION_SPLITS
    ),
    "design_note": (
        "For a given evaluation origin, "
        "all personalization levels predict "
        "the exact same target window. "
        "Only the amount of previous "
        "same-entity feedback changes."
    ),
}

metadata_path = (
    OUT_DIR
    / "electricity_protocol_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

print("=" * 72)
print(
    "PersTSFM Electricity evaluation protocol"
)
print("=" * 72)

print()
print(
    f"Resolution:            "
    f"{FREQUENCY_MINUTES} minutes"
)

print(
    f"Context:               "
    f"{CONTEXT_DAYS} days "
    f"({context_steps} steps)"
)

print(
    f"Forecast horizon:      "
    f"{FORECAST_DAYS} day "
    f"({forecast_steps} steps)"
)

print(
    "Personalization days: "
    + ", ".join(
        str(x)
        for x in PERSONALIZATION_DAYS
    )
)

print(
    f"Origins/entity:        "
    f"{N_EVAL_ORIGINS}"
)

print(
    f"Origin spacing:        "
    f"{EVAL_SPACING_DAYS} days"
)

print()

print(
    "Evaluation cases by split:"
)

print(
    manifest[
        "split"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print()

print(
    "Entities by split:"
)

print(
    manifest[
        [
            "entity",
            "split",
        ]
    ]
    .drop_duplicates()[
        "split"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print()

print(
    "Available history at evaluation origins:"
)

print(
    manifest.groupby(
        "split"
    )[
        "available_history_days"
    ]
    .agg(
        [
            "min",
            "median",
            "mean",
            "max",
        ]
    )
    .round(1)
    .to_string()
)

print()

print(
    "First five protocol rows:"
)

print(
    manifest.head()[
        [
            "entity",
            "split",
            "eval_index",
            "eval_origin",
            "context_start",
            "target_start",
            "target_end",
        ]
    ].to_string(
        index=False
    )
)

print()
print("Saved:")
print(f"  {manifest_path}")
print(f"  {metadata_path}")
