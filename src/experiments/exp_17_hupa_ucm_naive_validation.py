from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RAW_DIR = Path(
    "data/raw/hupa_ucm/"
    "HUPA-UCM-extracted-glucose-files"
)

PROTOCOL_PATH = Path(
    "data/processed/hupa_ucm/"
    "hupa_ucm_evaluation_origins.csv"
)

OUT_DIR = Path(
    "results/hupa_ucm/baselines"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SPLIT = "development"

FORECAST_STEPS = 12


# ---------------------------------------------------------------------
# Load protocol
# ---------------------------------------------------------------------

protocol = pd.read_csv(
    PROTOCOL_PATH,
    parse_dates=[
        "eval_origin",
        "target_start",
        "target_end",
    ],
)

protocol = (
    protocol[
        protocol["split"] == SPLIT
    ]
    .copy()
    .sort_values(
        [
            "person_id",
            "eval_index",
        ]
    )
    .reset_index(drop=True)
)

people = sorted(
    protocol["person_id"].unique()
)


print("=" * 78)
print("PersTSFM HUPA-UCM naive baselines")
print("=" * 78)

print()
print(
    f"Development participants: "
    f"{len(people)}"
)

print(
    f"Evaluation cases:         "
    f"{len(protocol)}"
)

print(
    f"Forecast steps:           "
    f"{FORECAST_STEPS}"
)


# ---------------------------------------------------------------------
# Load each person's series
# ---------------------------------------------------------------------

series_by_person = {}

for person_id in people:

    path = (
        RAW_DIR
        / f"{person_id}.csv"
    )

    df = pd.read_csv(
        path,
        usecols=[
            "timestamp",
            "glucose_value_mg_dl",
        ],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df[
        "glucose_value_mg_dl"
    ] = pd.to_numeric(
        df[
            "glucose_value_mg_dl"
        ]
    )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset="timestamp",
            keep="first",
        )
        .set_index("timestamp")
    )

    series_by_person[
        person_id
    ] = df[
        "glucose_value_mg_dl"
    ].astype(float)


# ---------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------

rows = []

for _, case in protocol.iterrows():

    person_id = case[
        "person_id"
    ]

    eval_index = int(
        case["eval_index"]
    )

    origin = case[
        "eval_origin"
    ]

    series = series_by_person[
        person_id
    ]

    target_index = pd.date_range(
        start=case["target_start"],
        end=case["target_end"],
        freq="5min",
    )

    if len(target_index) != FORECAST_STEPS:
        raise RuntimeError(
            f"{person_id} eval {eval_index}: "
            f"expected {FORECAST_STEPS} "
            f"target timestamps, "
            f"found {len(target_index)}."
        )

    truth = series.loc[
        target_index
    ].to_numpy(
        dtype=float
    )

    # -------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------

    last_value = float(
        series.loc[
            origin
        ]
    )

    persistence = np.full(
        FORECAST_STEPS,
        last_value,
        dtype=float,
    )

    # -------------------------------------------------------------
    # Same trajectory 24 hours earlier
    # -------------------------------------------------------------

    daily_index = (
        target_index
        - pd.Timedelta(days=1)
    )

    daily = series.loc[
        daily_index
    ].to_numpy(
        dtype=float
    )

    # -------------------------------------------------------------
    # Same trajectory 48 hours earlier
    # -------------------------------------------------------------

    two_day_index = (
        target_index
        - pd.Timedelta(days=2)
    )

    two_day = series.loc[
        two_day_index
    ].to_numpy(
        dtype=float
    )

    # -------------------------------------------------------------
    # Same trajectory 7 days earlier
    # -------------------------------------------------------------

    weekly_index = (
        target_index
        - pd.Timedelta(days=7)
    )

    weekly = series.loc[
        weekly_index
    ].to_numpy(
        dtype=float
    )

    methods = {
        "persistence": persistence,
        "daily_seasonal": daily,
        "two_day_seasonal": two_day,
        "weekly_seasonal": weekly,
    }

    for method, prediction in (
        methods.items()
    ):

        error = (
            prediction
            - truth
        )

        mae = float(
            np.mean(
                np.abs(error)
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        )

        rows.append(
            {
                "person_id": person_id,
                "eval_index": eval_index,
                "eval_origin": origin,
                "hour": int(
                    origin.hour
                ),
                "method": method,
                "mae": mae,
                "rmse": rmse,
            }
        )


case_metrics = pd.DataFrame(
    rows
)


# ---------------------------------------------------------------------
# Participant-level metrics
# ---------------------------------------------------------------------

participant_metrics = (
    case_metrics
    .groupby(
        [
            "person_id",
            "method",
        ],
        as_index=False,
    )
    .agg(
        mean_mae=(
            "mae",
            "mean",
        ),
        median_mae=(
            "mae",
            "median",
        ),
        mean_rmse=(
            "rmse",
            "mean",
        ),
    )
)


# ---------------------------------------------------------------------
# Overall summary
#
# Participant first, then cohort.
# ---------------------------------------------------------------------

summary = (
    participant_metrics
    .groupby(
        "method",
        as_index=False,
    )
    .agg(
        mean_participant_mae=(
            "mean_mae",
            "mean",
        ),
        median_participant_mae=(
            "mean_mae",
            "median",
        ),
        mean_participant_rmse=(
            "mean_rmse",
            "mean",
        ),
    )
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

case_path = (
    OUT_DIR
    / "hupa_ucm_naive_case_metrics.csv"
)

participant_path = (
    OUT_DIR
    / "hupa_ucm_naive_participant_metrics.csv"
)

summary_path = (
    OUT_DIR
    / "hupa_ucm_naive_summary.csv"
)

case_metrics.to_csv(
    case_path,
    index=False,
)

participant_metrics.to_csv(
    participant_path,
    index=False,
)

summary.to_csv(
    summary_path,
    index=False,
)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("Naive baseline summary")
print("=" * 78)

display = summary.copy()

for col in [
    "mean_participant_mae",
    "median_participant_mae",
    "mean_participant_rmse",
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
print("Saved:")
print(f"  {case_path}")
print(f"  {participant_path}")
print(f"  {summary_path}")
