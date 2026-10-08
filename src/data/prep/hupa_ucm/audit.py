from pathlib import Path
import json

import numpy as np 
import pandas as pd

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RAW_DIR = Path(
    "data/raw/hupa_ucm/"
    "HUPA-UCM-extracted-glucose-files"
)

METADATA_INPUT = Path(
    "data/raw/hupa_ucm/"
    "HUPA-UCM-metadata.csv"
)

OUT_DIR = Path(
    "data/processed/hupa_ucm"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

AUDIT_PATH = (
    OUT_DIR
    / "hupa_ucm_participant_audit.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "hupa_ucm_audit_metadata.json"
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def longest_regular_run(
    timestamps,
    expected_gap,
):
    """
    Number of timestamps in the longest run where every adjacent
    timestamp is separated by exactly expected_gap.
    """

    timestamps = pd.DatetimeIndex(
        timestamps
    ).sort_values()

    if len(timestamps) == 0:
        return 0

    if len(timestamps) == 1:
        return 1

    diffs = (
        pd.Series(timestamps)
        .diff()
        .iloc[1:]
    )

    best = 1
    current = 1

    for gap in diffs:

        if gap == expected_gap:
            current += 1

        else:
            best = max(
                best,
                current,
            )

            current = 1

    best = max(
        best,
        current,
    )

    return int(best)


# ---------------------------------------------------------------------
# Discover participant files
# ---------------------------------------------------------------------

files = sorted(
    RAW_DIR.glob("*.csv")
)

if len(files) == 0:
    raise RuntimeError(
        f"No participant CSV files found in "
        f"{RAW_DIR}"
    )


print("=" * 78)
print("PersTSFM HUPA-UCM CGM audit")
print("=" * 78)

print()
print(
    f"Participant files: "
    f"{len(files)}"
)

print(
    f"Raw directory:     "
    f"{RAW_DIR}"
)


# ---------------------------------------------------------------------
# Load optional metadata
# ---------------------------------------------------------------------

metadata = None

if METADATA_INPUT.exists():

    metadata = pd.read_csv(
        METADATA_INPUT
    )

    print()
    print(
        f"Metadata rows:      "
        f"{len(metadata)}"
    )

    print(
        "Metadata columns:   "
        + ", ".join(
            metadata.columns
            .astype(str)
            .tolist()
        )
    )


# ---------------------------------------------------------------------
# Audit each participant
# ---------------------------------------------------------------------

rows = []


for path in files:

    person_id = path.stem

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    required = {
        "timestamp",
        "glucose_value_mg_dl",
    }

    missing_columns = (
        required
        - set(df.columns)
    )

    if missing_columns:
        raise RuntimeError(
            f"{person_id}: missing required "
            f"columns {sorted(missing_columns)}"
        )

    # -------------------------------------------------------------
    # Parse timestamps and glucose
    # -------------------------------------------------------------

    timestamp = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    glucose = pd.to_numeric(
        df["glucose_value_mg_dl"],
        errors="coerce",
    )

    n_rows = len(df)

    bad_timestamp_count = int(
        timestamp.isna().sum()
    )

    missing_glucose_count = int(
        glucose.isna().sum()
    )

    valid_mask = (
        timestamp.notna()
        &
        glucose.notna()
    )

    valid = pd.DataFrame(
        {
            "timestamp": (
                timestamp[
                    valid_mask
                ]
            ),
            "glucose": (
                glucose[
                    valid_mask
                ]
            ),
        }
    )

    valid = (
        valid
        .sort_values(
            "timestamp"
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------
    # Duplicate timestamps
    # -------------------------------------------------------------

    duplicate_timestamp_count = int(
        valid[
            "timestamp"
        ]
        .duplicated(
            keep=False
        )
        .sum()
    )

    unique = (
        valid
        .drop_duplicates(
            subset="timestamp",
            keep="first",
        )
        .sort_values(
            "timestamp"
        )
        .reset_index(drop=True)
    )

    n_unique_timestamps = len(
        unique
    )

    if n_unique_timestamps < 2:

        rows.append(
            {
                "person_id": person_id,
                "n_rows": n_rows,
                "n_valid": len(valid),
                "n_unique_timestamps": (
                    n_unique_timestamps
                ),
                "bad_timestamp_count": (
                    bad_timestamp_count
                ),
                "missing_glucose_count": (
                    missing_glucose_count
                ),
                "duplicate_timestamp_count": (
                    duplicate_timestamp_count
                ),
                "first_timestamp": pd.NaT,
                "last_timestamp": pd.NaT,
                "span_days": np.nan,
                "unique_calendar_days": np.nan,
                "median_interval_minutes": np.nan,
                "modal_interval_minutes": np.nan,
                "modal_interval_fraction": np.nan,
                "coverage_fraction_full_span": np.nan,
                "n_nonmodal_gaps": np.nan,
                "n_gaps_gt_30min": np.nan,
                "n_gaps_gt_1h": np.nan,
                "n_gaps_gt_6h": np.nan,
                "max_gap_hours": np.nan,
                "longest_regular_run_steps": (
                    n_unique_timestamps
                ),
                "longest_regular_run_hours": np.nan,
                "longest_regular_run_days": np.nan,
                "mean_glucose": (
                    unique[
                        "glucose"
                    ].mean()
                    if n_unique_timestamps > 0
                    else np.nan
                ),
                "median_glucose": (
                    unique[
                        "glucose"
                    ].median()
                    if n_unique_timestamps > 0
                    else np.nan
                ),
                "sd_glucose": (
                    unique[
                        "glucose"
                    ].std()
                    if n_unique_timestamps > 0
                    else np.nan
                ),
                "min_glucose": (
                    unique[
                        "glucose"
                    ].min()
                    if n_unique_timestamps > 0
                    else np.nan
                ),
                "max_glucose": (
                    unique[
                        "glucose"
                    ].max()
                    if n_unique_timestamps > 0
                    else np.nan
                ),
                "n_below_40": int(
                    (
                        unique[
                            "glucose"
                        ] < 40
                    ).sum()
                ),
                "n_above_400": int(
                    (
                        unique[
                            "glucose"
                        ] > 400
                    ).sum()
                ),
            }
        )

        continue

    # -------------------------------------------------------------
    # Timing structure
    # -------------------------------------------------------------

    ts = pd.DatetimeIndex(
        unique[
            "timestamp"
        ]
    )

    diffs = (
        pd.Series(ts)
        .diff()
        .dropna()
    )

    median_gap = (
        diffs.median()
    )

    modal_gap = (
        diffs
        .value_counts()
        .idxmax()
    )

    median_interval_minutes = float(
        median_gap.total_seconds()
        / 60.0
    )

    modal_interval_minutes = float(
        modal_gap.total_seconds()
        / 60.0
    )

    modal_interval_fraction = float(
        (
            diffs
            == modal_gap
        ).mean()
    )

    nonmodal_gaps = int(
        (
            diffs
            != modal_gap
        ).sum()
    )

    n_gaps_gt_30min = int(
        (
            diffs
            > pd.Timedelta(
                minutes=30
            )
        ).sum()
    )

    n_gaps_gt_1h = int(
        (
            diffs
            > pd.Timedelta(
                hours=1
            )
        ).sum()
    )

    n_gaps_gt_6h = int(
        (
            diffs
            > pd.Timedelta(
                hours=6
            )
        ).sum()
    )

    max_gap_hours = float(
        diffs.max()
        .total_seconds()
        / 3600.0
    )

    first_timestamp = ts.min()
    last_timestamp = ts.max()

    span_seconds = (
        last_timestamp
        - first_timestamp
    ).total_seconds()

    span_days = float(
        span_seconds
        / 86400.0
    )

    unique_calendar_days = int(
        pd.Series(ts.date)
        .nunique()
    )

    # -------------------------------------------------------------
    # Coverage relative to expected regular grid
    # -------------------------------------------------------------

    modal_seconds = (
        modal_gap.total_seconds()
    )

    expected_count = int(
        np.floor(
            span_seconds
            / modal_seconds
        )
        + 1
    )

    coverage_fraction = float(
        n_unique_timestamps
        / expected_count
    )

    # -------------------------------------------------------------
    # Longest uninterrupted regular segment
    # -------------------------------------------------------------

    longest_steps = (
        longest_regular_run(
            ts,
            modal_gap,
        )
    )

    longest_hours = float(
        longest_steps
        * modal_interval_minutes
        / 60.0
    )

    longest_days = float(
        longest_hours
        / 24.0
    )

    # -------------------------------------------------------------
    # Glucose diagnostics
    # -------------------------------------------------------------

    g = unique[
        "glucose"
    ].to_numpy(
        dtype=float
    )

    rows.append(
        {
            "person_id": person_id,

            "n_rows": n_rows,

            "n_valid": len(
                valid
            ),

            "n_unique_timestamps": (
                n_unique_timestamps
            ),

            "bad_timestamp_count": (
                bad_timestamp_count
            ),

            "missing_glucose_count": (
                missing_glucose_count
            ),

            "duplicate_timestamp_count": (
                duplicate_timestamp_count
            ),

            "first_timestamp": (
                first_timestamp
            ),

            "last_timestamp": (
                last_timestamp
            ),

            "span_days": (
                span_days
            ),

            "unique_calendar_days": (
                unique_calendar_days
            ),

            "median_interval_minutes": (
                median_interval_minutes
            ),

            "modal_interval_minutes": (
                modal_interval_minutes
            ),

            "modal_interval_fraction": (
                modal_interval_fraction
            ),

            "coverage_fraction_full_span": (
                coverage_fraction
            ),

            "n_nonmodal_gaps": (
                nonmodal_gaps
            ),

            "n_gaps_gt_30min": (
                n_gaps_gt_30min
            ),

            "n_gaps_gt_1h": (
                n_gaps_gt_1h
            ),

            "n_gaps_gt_6h": (
                n_gaps_gt_6h
            ),

            "max_gap_hours": (
                max_gap_hours
            ),

            "longest_regular_run_steps": (
                longest_steps
            ),

            "longest_regular_run_hours": (
                longest_hours
            ),

            "longest_regular_run_days": (
                longest_days
            ),

            "mean_glucose": float(
                np.mean(g)
            ),

            "median_glucose": float(
                np.median(g)
            ),

            "sd_glucose": float(
                np.std(
                    g,
                    ddof=1,
                )
            ),

            "min_glucose": float(
                np.min(g)
            ),

            "max_glucose": float(
                np.max(g)
            ),

            "n_below_40": int(
                np.sum(
                    g < 40
                )
            ),

            "n_above_400": int(
                np.sum(
                    g > 400
                )
            ),
        }
    )


audit = pd.DataFrame(
    rows
)


# ---------------------------------------------------------------------
# Optional metadata comparison
# ---------------------------------------------------------------------

if (
    metadata is not None
    and "person_id"
    in metadata.columns
):

    metadata[
        "person_id"
    ] = (
        metadata[
            "person_id"
        ]
        .astype(str)
    )

    audit[
        "person_id"
    ] = (
        audit[
            "person_id"
        ]
        .astype(str)
    )

    keep_columns = [
        x
        for x in [
            "person_id",
            "diabetes_type",
            "CGM_type",
            "glucose_level_record_count",
            "count_days_with_CGM_data",
        ]
        if x in metadata.columns
    ]

    audit = audit.merge(
        metadata[
            keep_columns
        ],
        on="person_id",
        how="left",
        validate="one_to_one",
    )


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

audit.to_csv(
    AUDIT_PATH,
    index=False,
)


# ---------------------------------------------------------------------
# Overall summary
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("Timing summary")
print("=" * 78)

print()
print(
    f"Participants:              "
    f"{len(audit)}"
)

print(
    f"Total rows:                "
    f"{int(audit['n_rows'].sum()):,}"
)

print(
    f"Bad timestamps:            "
    f"{int(audit['bad_timestamp_count'].sum()):,}"
)

print(
    f"Missing glucose values:    "
    f"{int(audit['missing_glucose_count'].sum()):,}"
)

print(
    f"Duplicate timestamp rows:  "
    f"{int(audit['duplicate_timestamp_count'].sum()):,}"
)


print()
print(
    "Modal sampling interval counts:"
)

print(
    audit[
        "modal_interval_minutes"
    ]
    .value_counts(
        dropna=False
    )
    .sort_index()
    .to_string()
)


print()
print(
    "Full-span duration quantiles (days):"
)

print(
    audit[
        "span_days"
    ]
    .quantile(
        [
            0.0,
            0.25,
            0.50,
            0.75,
            1.0,
        ]
    )
    .round(2)
    .to_string()
)


print()
print(
    "Full-span coverage quantiles:"
)

print(
    audit[
        "coverage_fraction_full_span"
    ]
    .quantile(
        [
            0.0,
            0.25,
            0.50,
            0.75,
            1.0,
        ]
    )
    .round(4)
    .to_string()
)


print()
print("=" * 78)
print("Longest uninterrupted regular CGM segments")
print("=" * 78)

print()

run_display = (
    audit[
        [
            "person_id",
            "modal_interval_minutes",
            "span_days",
            "coverage_fraction_full_span",
            "longest_regular_run_days",
            "max_gap_hours",
        ]
    ]
    .sort_values(
        "longest_regular_run_days",
        ascending=False,
    )
    .copy()
)

for col in [
    "modal_interval_minutes",
    "span_days",
    "coverage_fraction_full_span",
    "longest_regular_run_days",
    "max_gap_hours",
]:
    run_display[col] = (
        run_display[col]
        .round(3)
    )

print(
    run_display.to_string(
        index=False
    )
)


# ---------------------------------------------------------------------
# Eligibility counts at candidate context durations
# ---------------------------------------------------------------------

threshold_days = [
    1,
    2,
    3,
    5,
    7,
    10,
    14,
]


print()
print("=" * 78)
print("Participants with uninterrupted CGM history")
print("=" * 78)

print()

for days in threshold_days:

    n = int(
        (
            audit[
                "longest_regular_run_days"
            ]
            >= days
        ).sum()
    )

    print(
        f">= {days:2d} days: "
        f"{n:2d} / {len(audit)}"
    )


# ---------------------------------------------------------------------
# Glucose diagnostics
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("Glucose-value diagnostics")
print("=" * 78)

print()

print(
    f"Overall minimum glucose: "
    f"{audit['min_glucose'].min():.2f}"
)

print(
    f"Overall maximum glucose: "
    f"{audit['max_glucose'].max():.2f}"
)

print(
    f"Readings < 40 mg/dL:     "
    f"{int(audit['n_below_40'].sum()):,}"
)

print(
    f"Readings > 400 mg/dL:    "
    f"{int(audit['n_above_400'].sum()):,}"
)


print()
print(
    "Participant mean-glucose quantiles:"
)

print(
    audit[
        "mean_glucose"
    ]
    .quantile(
        [
            0.0,
            0.25,
            0.50,
            0.75,
            1.0,
        ]
    )
    .round(2)
    .to_string()
)


# ---------------------------------------------------------------------
# Metadata output
# ---------------------------------------------------------------------

summary = {
    "n_participants": int(
        len(audit)
    ),
    "total_rows": int(
        audit[
            "n_rows"
        ].sum()
    ),
    "total_bad_timestamps": int(
        audit[
            "bad_timestamp_count"
        ].sum()
    ),
    "total_missing_glucose": int(
        audit[
            "missing_glucose_count"
        ].sum()
    ),
    "total_duplicate_timestamp_rows": int(
        audit[
            "duplicate_timestamp_count"
        ].sum()
    ),
    "median_span_days": float(
        audit[
            "span_days"
        ].median()
    ),
    "median_coverage_fraction": float(
        audit[
            "coverage_fraction_full_span"
        ].median()
    ),
    "median_longest_regular_run_days": float(
        audit[
            "longest_regular_run_days"
        ].median()
    ),
}

with open(
    SUMMARY_PATH,
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print()
print("=" * 78)
print("Audit complete")
print("=" * 78)

print()
print("Saved:")
print(f"  {AUDIT_PATH}")
print(f"  {SUMMARY_PATH}")