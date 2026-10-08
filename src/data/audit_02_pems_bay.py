from pathlib import Path
import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RAW_PATH = Path(
    "data/raw/pems_bay/PEMS-BAY.csv"
)

OUT_DIR = Path(
    "data/processed/pems_bay"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SENSOR_AUDIT_PATH = (
    OUT_DIR
    / "pems_bay_sensor_audit.csv"
)

METADATA_PATH = (
    OUT_DIR
    / "pems_bay_audit_metadata.json"
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def longest_true_run(values):
    values = np.asarray(
        values,
        dtype=bool,
    )

    if len(values) == 0:
        return 0

    if not values.any():
        return 0

    padded = np.concatenate(
        [
            np.array(
                [False]
            ),
            values,
            np.array(
                [False]
            ),
        ]
    )

    changes = np.diff(
        padded.astype(int)
    )

    starts = np.where(
        changes == 1
    )[0]

    ends = np.where(
        changes == -1
    )[0]

    return int(
        np.max(
            ends - starts
        )
    )


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

print("=" * 78)
print("PersTSFM PEMS-BAY audit")
print("=" * 78)

print()
print(f"Reading: {RAW_PATH}")

df = pd.read_csv(
    RAW_PATH,
    low_memory=False,
)

print()
print(
    f"Raw shape: "
    f"{df.shape[0]:,} rows x "
    f"{df.shape[1]:,} columns"
)

print()
print(
    "First columns:"
)

print(
    list(
        df.columns[:10]
    )
)


# ---------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------

timestamp_col = df.columns[0]

timestamps = pd.to_datetime(
    df[timestamp_col],
    errors="coerce",
)

bad_timestamps = int(
    timestamps.isna().sum()
)

if bad_timestamps > 0:
    raise RuntimeError(
        f"Could not parse "
        f"{bad_timestamps:,} timestamps "
        f"from column '{timestamp_col}'."
    )

df = df.drop(
    columns=[
        timestamp_col
    ]
)

sensor_cols = list(
    df.columns
)

df.index = timestamps
df.index.name = "timestamp"

df = df.sort_index()


# ---------------------------------------------------------------------
# Convert sensors to numeric
# ---------------------------------------------------------------------

numeric = df.apply(
    pd.to_numeric,
    errors="coerce",
)


# ---------------------------------------------------------------------
# Timestamp audit
# ---------------------------------------------------------------------

duplicate_timestamps = int(
    numeric.index.duplicated().sum()
)

time_diffs = (
    numeric.index
    .to_series()
    .diff()
    .dropna()
)

if len(time_diffs) == 0:
    raise RuntimeError(
        "Not enough timestamps "
        "to determine frequency."
    )

median_interval = (
    time_diffs.median()
)

median_interval_minutes = float(
    median_interval.total_seconds()
    / 60.0
)

mode_interval = (
    time_diffs.value_counts()
    .idxmax()
)

mode_interval_minutes = float(
    mode_interval.total_seconds()
    / 60.0
)

non_modal_gaps = int(
    (
        time_diffs
        != mode_interval
    ).sum()
)

non_positive_gaps = int(
    (
        time_diffs
        <= pd.Timedelta(0)
    ).sum()
)


# ---------------------------------------------------------------------
# Overall missingness
# ---------------------------------------------------------------------

total_values = (
    numeric.shape[0]
    * numeric.shape[1]
)

total_missing = int(
    numeric.isna().sum().sum()
)

overall_missing_fraction = (
    total_missing
    / total_values
)


rows_with_any_missing = int(
    numeric.isna()
    .any(axis=1)
    .sum()
)

sensors_with_any_missing = int(
    numeric.isna()
    .any(axis=0)
    .sum()
)


# ---------------------------------------------------------------------
# Sensor-level audit
# ---------------------------------------------------------------------

sensor_rows = []

for sensor in sensor_cols:

    values = numeric[sensor]

    valid = values.dropna()

    n_total = len(values)

    n_missing = int(
        values.isna().sum()
    )

    n_valid = int(
        values.notna().sum()
    )

    missing_fraction = (
        n_missing
        / n_total
    )

    if n_valid == 0:

        sensor_rows.append(
            {
                "sensor": sensor,
                "n_total": n_total,
                "n_valid": 0,
                "n_missing": n_missing,
                "missing_fraction": (
                    missing_fraction
                ),
                "n_zero": 0,
                "zero_fraction_valid": np.nan,
                "n_negative": 0,
                "negative_fraction_valid": np.nan,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "n_unique": 0,
                "first_valid": pd.NaT,
                "last_valid": pd.NaT,
                "span_days": np.nan,
                "longest_missing_run_steps": (
                    longest_true_run(
                        values.isna()
                        .to_numpy()
                    )
                ),
                "longest_zero_run_steps": np.nan,
                "all_zero": False,
                "constant": False,
            }
        )

        continue

    n_zero = int(
        (
            valid == 0
        ).sum()
    )

    zero_fraction = (
        n_zero
        / n_valid
    )

    n_negative = int(
        (
            valid < 0
        ).sum()
    )

    negative_fraction = (
        n_negative
        / n_valid
    )

    first_valid = (
        valid.index[0]
    )

    last_valid = (
        valid.index[-1]
    )

    span_days = float(
        (
            last_valid
            - first_valid
        ).total_seconds()
        / 86400.0
    )

    valid_values = valid.to_numpy(
        dtype=float
    )

    all_zero = bool(
        np.all(
            valid_values == 0
        )
    )

    n_unique = int(
        valid.nunique(
            dropna=True
        )
    )

    constant = bool(
        n_unique <= 1
    )

    zero_mask = (
        values.fillna(
            np.inf
        )
        .to_numpy()
        == 0
    )

    missing_mask = (
        values.isna()
        .to_numpy()
    )

    sensor_rows.append(
        {
            "sensor": sensor,
            "n_total": n_total,
            "n_valid": n_valid,
            "n_missing": n_missing,
            "missing_fraction": (
                missing_fraction
            ),
            "n_zero": n_zero,
            "zero_fraction_valid": (
                zero_fraction
            ),
            "n_negative": n_negative,
            "negative_fraction_valid": (
                negative_fraction
            ),
            "mean": float(
                valid.mean()
            ),
            "std": float(
                valid.std()
            ),
            "min": float(
                valid.min()
            ),
            "max": float(
                valid.max()
            ),
            "n_unique": n_unique,
            "first_valid": first_valid,
            "last_valid": last_valid,
            "span_days": span_days,
            "longest_missing_run_steps": (
                longest_true_run(
                    missing_mask
                )
            ),
            "longest_zero_run_steps": (
                longest_true_run(
                    zero_mask
                )
            ),
            "all_zero": all_zero,
            "constant": constant,
        }
    )


sensor_audit = pd.DataFrame(
    sensor_rows
)


# ---------------------------------------------------------------------
# Derived counts
# ---------------------------------------------------------------------

all_missing_sensors = int(
    (
        sensor_audit[
            "n_valid"
        ] == 0
    ).sum()
)

all_zero_sensors = int(
    sensor_audit[
        "all_zero"
    ].sum()
)

constant_sensors = int(
    sensor_audit[
        "constant"
    ].sum()
)

sensors_with_zeros = int(
    (
        sensor_audit[
            "n_zero"
        ] > 0
    ).sum()
)

sensors_with_negatives = int(
    (
        sensor_audit[
            "n_negative"
        ] > 0
    ).sum()
)


# ---------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("Timestamp structure")
print("=" * 78)

print()
print(
    f"Rows/timestamps:          "
    f"{len(numeric):,}"
)

print(
    f"Sensors/entities:         "
    f"{len(sensor_cols):,}"
)

print(
    f"First timestamp:          "
    f"{numeric.index.min()}"
)

print(
    f"Last timestamp:           "
    f"{numeric.index.max()}"
)

print(
    f"Median interval:          "
    f"{median_interval_minutes:.2f} min"
)

print(
    f"Modal interval:           "
    f"{mode_interval_minutes:.2f} min"
)

print(
    f"Non-modal time gaps:      "
    f"{non_modal_gaps:,}"
)

print(
    f"Non-positive time gaps:   "
    f"{non_positive_gaps:,}"
)

print(
    f"Duplicate timestamps:     "
    f"{duplicate_timestamps:,}"
)


print()
print("=" * 78)
print("Missingness")
print("=" * 78)

print()
print(
    f"Total values:             "
    f"{total_values:,}"
)

print(
    f"Missing values:           "
    f"{total_missing:,}"
)

print(
    f"Overall missing fraction: "
    f"{overall_missing_fraction:.6f}"
)

print(
    f"Rows with any missing:    "
    f"{rows_with_any_missing:,}"
)

print(
    f"Sensors with missing:     "
    f"{sensors_with_any_missing:,}"
)

print(
    f"All-missing sensors:      "
    f"{all_missing_sensors:,}"
)


print()
print("=" * 78)
print("Value diagnostics")
print("=" * 78)

print()
print(
    f"Sensors with zeros:       "
    f"{sensors_with_zeros:,}"
)

print(
    f"All-zero sensors:         "
    f"{all_zero_sensors:,}"
)

print(
    f"Constant sensors:         "
    f"{constant_sensors:,}"
)

print(
    f"Sensors with negatives:   "
    f"{sensors_with_negatives:,}"
)


print()
print("Missing-fraction quantiles:")

print(
    sensor_audit[
        "missing_fraction"
    ]
    .quantile(
        [
            0.0,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            1.0,
        ]
    )
    .round(6)
    .to_string()
)


print()
print("Zero-fraction quantiles:")

print(
    sensor_audit[
        "zero_fraction_valid"
    ]
    .quantile(
        [
            0.0,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            1.0,
        ]
    )
    .round(6)
    .to_string()
)


print()
print("Sensor mean-speed quantiles:")

print(
    sensor_audit[
        "mean"
    ]
    .quantile(
        [
            0.0,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            1.0,
        ]
    )
    .round(3)
    .to_string()
)


print()
print("Sensor span-days quantiles:")

print(
    sensor_audit[
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
    .round(3)
    .to_string()
)


print()
print("Longest missing-run steps:")

print(
    sensor_audit[
        "longest_missing_run_steps"
    ]
    .describe()
    .round(2)
    .to_string()
)


print()
print("Longest zero-run steps:")

print(
    sensor_audit[
        "longest_zero_run_steps"
    ]
    .describe()
    .round(2)
    .to_string()
)


# ---------------------------------------------------------------------
# Most problematic sensors
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("Sensors with highest missingness")
print("=" * 78)

problem_missing = (
    sensor_audit
    .sort_values(
        [
            "missing_fraction",
            "longest_missing_run_steps",
        ],
        ascending=False,
    )
    .head(15)
)

print()
print(
    problem_missing[
        [
            "sensor",
            "missing_fraction",
            "longest_missing_run_steps",
            "n_zero",
            "mean",
            "std",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 78)
print("Sensors with highest zero fraction")
print("=" * 78)

problem_zero = (
    sensor_audit
    .sort_values(
        [
            "zero_fraction_valid",
            "longest_zero_run_steps",
        ],
        ascending=False,
    )
    .head(15)
)

print()
print(
    problem_zero[
        [
            "sensor",
            "zero_fraction_valid",
            "longest_zero_run_steps",
            "missing_fraction",
            "mean",
            "std",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

sensor_audit.to_csv(
    SENSOR_AUDIT_PATH,
    index=False,
)

metadata = {
    "raw_path": str(
        RAW_PATH
    ),
    "n_timestamps": int(
        len(numeric)
    ),
    "n_sensors": int(
        len(sensor_cols)
    ),
    "first_timestamp": str(
        numeric.index.min()
    ),
    "last_timestamp": str(
        numeric.index.max()
    ),
    "median_interval_minutes": (
        median_interval_minutes
    ),
    "mode_interval_minutes": (
        mode_interval_minutes
    ),
    "non_modal_time_gaps": (
        non_modal_gaps
    ),
    "non_positive_time_gaps": (
        non_positive_gaps
    ),
    "duplicate_timestamps": (
        duplicate_timestamps
    ),
    "total_values": int(
        total_values
    ),
    "total_missing": (
        total_missing
    ),
    "overall_missing_fraction": (
        overall_missing_fraction
    ),
    "rows_with_any_missing": (
        rows_with_any_missing
    ),
    "sensors_with_any_missing": (
        sensors_with_any_missing
    ),
    "all_missing_sensors": (
        all_missing_sensors
    ),
    "sensors_with_zeros": (
        sensors_with_zeros
    ),
    "all_zero_sensors": (
        all_zero_sensors
    ),
    "constant_sensors": (
        constant_sensors
    ),
    "sensors_with_negatives": (
        sensors_with_negatives
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


print()
print("=" * 78)
print("Audit complete")
print("=" * 78)

print()
print("Saved:")
print(f"  {SENSOR_AUDIT_PATH}")
print(f"  {METADATA_PATH}")
