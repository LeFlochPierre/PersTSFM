from pathlib import Path
import json

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RAW_PATH = Path(
    "data/raw/pems_bay/PEMS-BAY.csv"
)

SPLIT_PATH = Path(
    "data/processed/pems_bay/"
    "pems_bay_sensor_splits.csv"
)

OUT_DIR = Path(
    "data/processed/pems_bay"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROTOCOL_PATH = (
    OUT_DIR
    / "pems_bay_evaluation_origins.csv"
)

METADATA_PATH = (
    OUT_DIR
    / "pems_bay_protocol_metadata.json"
)


RESOLUTION_MINUTES = 5

FORECAST_HOURS = 1

FORECAST_STEPS = (
    FORECAST_HOURS
    * 60
    // RESOLUTION_MINUTES
)

CONTEXT_HOURS = [
    3,
    6,
    12,
    24,
    48,
    72,
    120,
    168,
    240,
    336,
    504,
    744,
]

CONTEXT_STEPS = [
    hours
    * 60
    // RESOLUTION_MINUTES
    for hours in CONTEXT_HOURS
]

MAX_CONTEXT_HOURS = max(
    CONTEXT_HOURS
)

MAX_CONTEXT_STEPS = max(
    CONTEXT_STEPS
)


# ---------------------------------------------------------------------
# Fixed evaluation origins
#
# Four clock times:
#   00:00
#   06:00
#   12:00
#   18:00
#
# Dates are spaced four days apart so weekdays rotate.
#
# Earliest origin is sufficiently late that the complete
# 31-day context lies after the March 12 DST discontinuity.
# ---------------------------------------------------------------------

EVALUATION_ORIGINS = [
    "2017-04-17 06:00:00",
    "2017-04-21 12:00:00",
    "2017-04-25 18:00:00",
    "2017-04-29 00:00:00",
    "2017-05-03 06:00:00",
    "2017-05-07 12:00:00",
    "2017-05-11 18:00:00",
    "2017-05-15 00:00:00",
    "2017-05-19 06:00:00",
    "2017-05-23 12:00:00",
    "2017-05-27 18:00:00",
    "2017-05-31 00:00:00",
    "2017-06-04 06:00:00",
    "2017-06-08 12:00:00",
    "2017-06-12 18:00:00",
    "2017-06-16 00:00:00",
]

EVALUATION_ORIGINS = [
    pd.Timestamp(x)
    for x in EVALUATION_ORIGINS
]


# ---------------------------------------------------------------------
# Load raw timestamps only
# ---------------------------------------------------------------------

header = pd.read_csv(
    RAW_PATH,
    nrows=0,
)

timestamp_col = header.columns[0]

timestamps = pd.read_csv(
    RAW_PATH,
    usecols=[
        timestamp_col
    ],
)

timestamps[timestamp_col] = pd.to_datetime(
    timestamps[timestamp_col]
)

index = pd.DatetimeIndex(
    timestamps[timestamp_col]
).sort_values()


# ---------------------------------------------------------------------
# Load entity split
# ---------------------------------------------------------------------

splits = pd.read_csv(
    SPLIT_PATH
)

splits["sensor"] = (
    splits["sensor"]
    .astype(str)
)

evaluation_sensors = splits[
    splits["split"].isin(
        [
            "validation",
            "heldout_test",
        ]
    )
].copy()


# ---------------------------------------------------------------------
# Window safety checks
# ---------------------------------------------------------------------

print("=" * 78)
print("PersTSFM PEMS-BAY evaluation protocol")
print("=" * 78)

print()
print(
    f"Resolution:          "
    f"{RESOLUTION_MINUTES} min"
)

print(
    f"Forecast horizon:    "
    f"{FORECAST_HOURS} hour "
    f"({FORECAST_STEPS} steps)"
)

print(
    f"Max context:         "
    f"{MAX_CONTEXT_HOURS / 24:.0f} days "
    f"({MAX_CONTEXT_STEPS} steps)"
)

print(
    f"Evaluation origins:  "
    f"{len(EVALUATION_ORIGINS)} "
    f"per sensor"
)

print()
print(
    "Context grid (hours):"
)

print(
    ", ".join(
        str(x)
        for x in CONTEXT_HOURS
    )
)


window_rows = []

for eval_index, origin in enumerate(
    EVALUATION_ORIGINS,
    start=1,
):

    max_context_start = (
        origin
        - pd.Timedelta(
            minutes=(
                MAX_CONTEXT_STEPS - 1
            )
            * RESOLUTION_MINUTES
        )
    )

    target_start = (
        origin
        + pd.Timedelta(
            minutes=RESOLUTION_MINUTES
        )
    )

    target_end = (
        origin
        + pd.Timedelta(
            minutes=(
                FORECAST_STEPS
                * RESOLUTION_MINUTES
            )
        )
    )

    context_index = index[
        (
            index
            >= max_context_start
        )
        &
        (
            index
            <= origin
        )
    ]

    target_index = index[
        (
            index
            >= target_start
        )
        &
        (
            index
            <= target_end
        )
    ]

    if len(context_index) != MAX_CONTEXT_STEPS:
        raise RuntimeError(
            f"Origin {origin}: "
            f"expected {MAX_CONTEXT_STEPS} "
            f"points in maximum context, "
            f"found {len(context_index)}."
        )

    if len(target_index) != FORECAST_STEPS:
        raise RuntimeError(
            f"Origin {origin}: "
            f"expected {FORECAST_STEPS} "
            f"target points, "
            f"found {len(target_index)}."
        )

    if context_index[-1] != origin:
        raise RuntimeError(
            f"Origin {origin}: "
            "context does not end "
            "at forecast origin."
        )

    if target_index[0] != target_start:
        raise RuntimeError(
            f"Origin {origin}: "
            "target does not start "
            "one step after origin."
        )

    window_rows.append(
        {
            "eval_index": eval_index,
            "eval_origin": origin,
            "weekday": (
                origin.day_name()
            ),
            "hour": int(
                origin.hour
            ),
            "max_context_start": (
                max_context_start
            ),
            "target_start": (
                target_start
            ),
            "target_end": (
                target_end
            ),
        }
    )


window_df = pd.DataFrame(
    window_rows
)


print()
print(
    "Maximum-context safety checks: "
    "PASSED"
)


# ---------------------------------------------------------------------
# Expand across validation and held-out sensors
# ---------------------------------------------------------------------

rows = []

for _, sensor_row in (
    evaluation_sensors.iterrows()
):

    sensor = sensor_row[
        "sensor"
    ]

    split = sensor_row[
        "split"
    ]

    for _, window in (
        window_df.iterrows()
    ):

        rows.append(
            {
                "sensor": sensor,
                "split": split,
                "eval_index": int(
                    window[
                        "eval_index"
                    ]
                ),
                "eval_origin": (
                    window[
                        "eval_origin"
                    ]
                ),
                "weekday": (
                    window[
                        "weekday"
                    ]
                ),
                "hour": int(
                    window[
                        "hour"
                    ]
                ),
                "max_context_start": (
                    window[
                        "max_context_start"
                    ]
                ),
                "target_start": (
                    window[
                        "target_start"
                    ]
                ),
                "target_end": (
                    window[
                        "target_end"
                    ]
                ),
                "forecast_steps": (
                    FORECAST_STEPS
                ),
                "resolution_minutes": (
                    RESOLUTION_MINUTES
                ),
                "max_context_steps": (
                    MAX_CONTEXT_STEPS
                ),
            }
        )


protocol = pd.DataFrame(
    rows
)

protocol = protocol.sort_values(
    [
        "split",
        "sensor",
        "eval_index",
    ]
).reset_index(
    drop=True
)


# ---------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------

expected_per_split = (
    49
    * len(EVALUATION_ORIGINS)
)

counts = (
    protocol[
        "split"
    ]
    .value_counts()
)

if (
    counts[
        "validation"
    ]
    != expected_per_split
):
    raise RuntimeError(
        "Unexpected validation "
        "case count."
    )

if (
    counts[
        "heldout_test"
    ]
    != expected_per_split
):
    raise RuntimeError(
        "Unexpected held-out test "
        "case count."
    )


# ---------------------------------------------------------------------
# Day-of-week and hour diagnostics
# ---------------------------------------------------------------------

origin_summary = (
    window_df[
        [
            "eval_index",
            "eval_origin",
            "weekday",
            "hour",
        ]
    ]
    .copy()
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

protocol.to_csv(
    PROTOCOL_PATH,
    index=False,
)

metadata = {
    "resolution_minutes": (
        RESOLUTION_MINUTES
    ),
    "forecast_hours": (
        FORECAST_HOURS
    ),
    "forecast_steps": (
        FORECAST_STEPS
    ),
    "context_hours": (
        CONTEXT_HOURS
    ),
    "context_steps": (
        CONTEXT_STEPS
    ),
    "max_context_hours": (
        MAX_CONTEXT_HOURS
    ),
    "max_context_steps": (
        MAX_CONTEXT_STEPS
    ),
    "evaluation_origins_per_sensor": (
        len(
            EVALUATION_ORIGINS
        )
    ),
    "validation_cases": int(
        counts[
            "validation"
        ]
    ),
    "heldout_test_cases": int(
        counts[
            "heldout_test"
        ]
    ),
}

with open(
    METADATA_PATH,
    "w",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("Protocol summary")
print("=" * 78)

print()
print(
    f"Validation sensors:       "
    f"{protocol.loc[protocol['split'] == 'validation', 'sensor'].nunique()}"
)

print(
    f"Held-out test sensors:    "
    f"{protocol.loc[protocol['split'] == 'heldout_test', 'sensor'].nunique()}"
)

print(
    f"Validation cases:         "
    f"{counts['validation']:,}"
)

print(
    f"Held-out test cases:      "
    f"{counts['heldout_test']:,}"
)

print()
print(
    "Evaluation origins:"
)

print()
print(
    origin_summary.to_string(
        index=False
    )
)


print()
print(
    "Origins by weekday:"
)

print(
    window_df[
        "weekday"
    ]
    .value_counts()
    .to_string()
)


print()
print(
    "Origins by hour:"
)

print(
    window_df[
        "hour"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print()
print("Saved:")
print(f"  {PROTOCOL_PATH}")
print(f"  {METADATA_PATH}")
