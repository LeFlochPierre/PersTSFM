from pathlib import Path
import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RAW_DIR = Path(
    "data/raw/hupa_ucm/"
    "HUPA-UCM-extracted-glucose-files"
)

SPLIT_PATH = Path(
    "data/processed/hupa_ucm/"
    "hupa_ucm_participant_splits.csv"
)

OUT_DIR = Path(
    "data/processed/hupa_ucm"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROTOCOL_PATH = (
    OUT_DIR
    / "hupa_ucm_evaluation_origins.csv"
)

METADATA_PATH = (
    OUT_DIR
    / "hupa_ucm_protocol_metadata.json"
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
    18,
    24,
    36,
    48,
    72,
    96,
    120,
    168,
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

N_ORIGINS_PER_PERSON = 5


# ---------------------------------------------------------------------
# Load split
# ---------------------------------------------------------------------

splits = pd.read_csv(
    SPLIT_PATH
)

splits["person_id"] = (
    splits["person_id"]
    .astype(str)
)

splits = splits[
    splits["split"].isin(
        [
            "development",
            "heldout_test",
        ]
    )
].copy()


print("=" * 78)
print("PersTSFM HUPA-UCM evaluation protocol")
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
    f"{MAX_CONTEXT_HOURS} hours "
    f"({MAX_CONTEXT_HOURS / 24:.1f} days, "
    f"{MAX_CONTEXT_STEPS} steps)"
)

print(
    f"Origins/person:      "
    f"{N_ORIGINS_PER_PERSON}"
)

print()
print(
    "Context grid (hours): "
    + ", ".join(
        str(x)
        for x in CONTEXT_HOURS
    )
)


# ---------------------------------------------------------------------
# Build participant-specific origins
# ---------------------------------------------------------------------

rows = []

origin_diagnostics = []


for _, split_row in splits.iterrows():

    person_id = split_row[
        "person_id"
    ]

    split = split_row[
        "split"
    ]

    path = (
        RAW_DIR
        / f"{person_id}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    df = pd.read_csv(
        path,
        usecols=[
            "timestamp",
            "glucose_value_mg_dl",
        ],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="raise",
    )

    df[
        "glucose_value_mg_dl"
    ] = pd.to_numeric(
        df[
            "glucose_value_mg_dl"
        ],
        errors="raise",
    )

    df = (
        df
        .sort_values(
            "timestamp"
        )
        .drop_duplicates(
            subset="timestamp",
            keep="first",
        )
        .reset_index(drop=True)
    )

    timestamps = pd.DatetimeIndex(
        df["timestamp"]
    )

    # -------------------------------------------------------------
    # Verify completely regular 5-minute sampling
    # -------------------------------------------------------------

    diffs = (
        pd.Series(timestamps)
        .diff()
        .dropna()
    )

    if not (
        diffs
        == pd.Timedelta(
            minutes=RESOLUTION_MINUTES
        )
    ).all():

        raise RuntimeError(
            f"{person_id}: non-regular "
            "sampling detected."
        )

    n = len(df)

    # Earliest usable origin must have the full
    # seven-day context available.
    first_origin_idx = (
        MAX_CONTEXT_STEPS - 1
    )

    # Latest usable origin must leave the full
    # one-hour target after the origin.
    last_origin_idx = (
        n
        - FORECAST_STEPS
        - 1
    )

    if (
        last_origin_idx
        < first_origin_idx
    ):
        raise RuntimeError(
            f"{person_id}: insufficient data "
            "for maximum context + target."
        )

    # -------------------------------------------------------------
    # Exactly five equally spaced origins across each person's
    # usable forecasting period.
    #
    # This prevents the 574-day participant from contributing
    # more cases than an 8-day participant.
    # -------------------------------------------------------------

    origin_indices = np.linspace(
        first_origin_idx,
        last_origin_idx,
        num=N_ORIGINS_PER_PERSON,
    )

    origin_indices = np.rint(
        origin_indices
    ).astype(int)

    if (
        len(
            np.unique(
                origin_indices
            )
        )
        != N_ORIGINS_PER_PERSON
    ):
        raise RuntimeError(
            f"{person_id}: could not create "
            f"{N_ORIGINS_PER_PERSON} unique origins."
        )

    person_origins = []

    for eval_index, origin_idx in enumerate(
        origin_indices,
        start=1,
    ):

        origin = timestamps[
            origin_idx
        ]

        max_context_start_idx = (
            origin_idx
            - MAX_CONTEXT_STEPS
            + 1
        )

        target_start_idx = (
            origin_idx + 1
        )

        target_end_idx = (
            origin_idx
            + FORECAST_STEPS
        )

        max_context_start = timestamps[
            max_context_start_idx
        ]

        target_start = timestamps[
            target_start_idx
        ]

        target_end = timestamps[
            target_end_idx
        ]

        # ---------------------------------------------------------
        # Positional safety checks
        # ---------------------------------------------------------

        context_length = (
            origin_idx
            - max_context_start_idx
            + 1
        )

        target_length = (
            target_end_idx
            - target_start_idx
            + 1
        )

        if (
            context_length
            != MAX_CONTEXT_STEPS
        ):
            raise RuntimeError(
                f"{person_id} eval {eval_index}: "
                "incorrect context length."
            )

        if (
            target_length
            != FORECAST_STEPS
        ):
            raise RuntimeError(
                f"{person_id} eval {eval_index}: "
                "incorrect target length."
            )

        expected_context_span = pd.Timedelta(
            minutes=(
                MAX_CONTEXT_STEPS - 1
            )
            * RESOLUTION_MINUTES
        )

        if (
            origin
            - max_context_start
            != expected_context_span
        ):
            raise RuntimeError(
                f"{person_id} eval {eval_index}: "
                "context timestamp continuity "
                "check failed."
            )

        if (
            target_start
            - origin
            != pd.Timedelta(
                minutes=RESOLUTION_MINUTES
            )
        ):
            raise RuntimeError(
                f"{person_id} eval {eval_index}: "
                "target does not begin one "
                "step after origin."
            )

        rows.append(
            {
                "person_id": person_id,
                "split": split,
                "eval_index": eval_index,
                "eval_origin": origin,
                "weekday": (
                    origin.day_name()
                ),
                "hour": int(
                    origin.hour
                ),
                "minute": int(
                    origin.minute
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
                "resolution_minutes": (
                    RESOLUTION_MINUTES
                ),
                "forecast_steps": (
                    FORECAST_STEPS
                ),
                "max_context_steps": (
                    MAX_CONTEXT_STEPS
                ),
                "n_total_person_points": (
                    n
                ),
            }
        )

        person_origins.append(
            origin
        )

    # -------------------------------------------------------------
    # Participant origin diagnostics
    # -------------------------------------------------------------

    separations = (
        pd.Series(
            person_origins
        )
        .diff()
        .dropna()
        .dt.total_seconds()
        / 86400.0
    )

    origin_diagnostics.append(
        {
            "person_id": person_id,
            "split": split,
            "n_points": n,
            "first_timestamp": (
                timestamps[0]
            ),
            "last_timestamp": (
                timestamps[-1]
            ),
            "first_eval_origin": (
                min(
                    person_origins
                )
            ),
            "last_eval_origin": (
                max(
                    person_origins
                )
            ),
            "min_origin_spacing_days": (
                float(
                    separations.min()
                )
            ),
            "median_origin_spacing_days": (
                float(
                    separations.median()
                )
            ),
        }
    )


protocol = pd.DataFrame(
    rows
)

diagnostics = pd.DataFrame(
    origin_diagnostics
)

protocol = (
    protocol
    .sort_values(
        [
            "split",
            "person_id",
            "eval_index",
        ]
    )
    .reset_index(drop=True)
)


# ---------------------------------------------------------------------
# Global sanity checks
# ---------------------------------------------------------------------

development = protocol[
    protocol["split"]
    == "development"
]

heldout = protocol[
    protocol["split"]
    == "heldout_test"
]

expected_dev = (
    20
    * N_ORIGINS_PER_PERSON
)

expected_test = (
    5
    * N_ORIGINS_PER_PERSON
)

if len(development) != expected_dev:
    raise RuntimeError(
        f"Expected {expected_dev} "
        f"development cases, "
        f"found {len(development)}."
    )

if len(heldout) != expected_test:
    raise RuntimeError(
        f"Expected {expected_test} "
        f"held-out cases, "
        f"found {len(heldout)}."
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
    "origins_per_person": (
        N_ORIGINS_PER_PERSON
    ),
    "development_participants": int(
        development[
            "person_id"
        ].nunique()
    ),
    "heldout_test_participants": int(
        heldout[
            "person_id"
        ].nunique()
    ),
    "development_cases": int(
        len(development)
    ),
    "heldout_test_cases": int(
        len(heldout)
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
    f"Development participants: "
    f"{development['person_id'].nunique()}"
)

print(
    f"Held-out participants:    "
    f"{heldout['person_id'].nunique()}"
)

print(
    f"Development cases:        "
    f"{len(development)}"
)

print(
    f"Held-out cases:           "
    f"{len(heldout)}"
)


print()
print("=" * 78)
print("Origin-spacing diagnostics")
print("=" * 78)

display = diagnostics.copy()

for col in [
    "min_origin_spacing_days",
    "median_origin_spacing_days",
]:
    display[col] = (
        display[col]
        .round(2)
    )

print()
print(
    display[
        [
            "person_id",
            "split",
            "n_points",
            "first_eval_origin",
            "last_eval_origin",
            "min_origin_spacing_days",
            "median_origin_spacing_days",
        ]
    ]
    .sort_values(
        [
            "split",
            "person_id",
        ]
    )
    .to_string(
        index=False
    )
)


print()
print("=" * 78)
print("Origin clock-time distribution")
print("=" * 78)

print()

clock_summary = (
    development
    .groupby(
        [
            "hour",
            "minute",
        ]
    )
    .size()
    .sort_values(
        ascending=False
    )
)

print(
    clock_summary
    .head(20)
    .to_string()
)


print()
print("Saved:")
print(f"  {PROTOCOL_PATH}")
print(f"  {METADATA_PATH}")
