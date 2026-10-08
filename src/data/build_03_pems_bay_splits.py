from pathlib import Path
import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

AUDIT_PATH = Path(
    "data/processed/pems_bay/"
    "pems_bay_sensor_audit.csv"
)

OUT_DIR = Path(
    "data/processed/pems_bay"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 42


# ---------------------------------------------------------------------
# Load sensor audit
# ---------------------------------------------------------------------

audit = pd.read_csv(
    AUDIT_PATH
)

sensors = audit[
    "sensor"
].astype(str).to_numpy()

n_sensors = len(sensors)

if n_sensors != 325:
    raise RuntimeError(
        f"Expected 325 sensors, "
        f"found {n_sensors}."
    )


# ---------------------------------------------------------------------
# Deterministic random entity split
#
# 325 -> 227 / 49 / 49
# ---------------------------------------------------------------------

rng = np.random.default_rng(
    SEED
)

shuffled = sensors.copy()

rng.shuffle(
    shuffled
)

train = shuffled[:227]

validation = shuffled[
    227:276
]

heldout_test = shuffled[
    276:
]


assert len(train) == 227
assert len(validation) == 49
assert len(heldout_test) == 49

assert len(
    set(train)
    & set(validation)
) == 0

assert len(
    set(train)
    & set(heldout_test)
) == 0

assert len(
    set(validation)
    & set(heldout_test)
) == 0

assert (
    len(
        set(train)
        | set(validation)
        | set(heldout_test)
    )
    == 325
)


# ---------------------------------------------------------------------
# Build split table
# ---------------------------------------------------------------------

split_map = {}

for sensor in train:
    split_map[sensor] = (
        "population_train"
    )

for sensor in validation:
    split_map[sensor] = (
        "validation"
    )

for sensor in heldout_test:
    split_map[sensor] = (
        "heldout_test"
    )


split_df = audit.copy()

split_df["sensor"] = (
    split_df["sensor"]
    .astype(str)
)

split_df["split"] = (
    split_df["sensor"]
    .map(split_map)
)


if split_df["split"].isna().any():
    raise RuntimeError(
        "Some sensors were not assigned "
        "to a split."
    )


# ---------------------------------------------------------------------
# Save lists
# ---------------------------------------------------------------------

def save_list(values, path):
    with open(
        path,
        "w",
    ) as f:
        for value in sorted(values):
            f.write(
                str(value)
                + "\n"
            )


train_path = (
    OUT_DIR
    / "population_train_sensors.txt"
)

validation_path = (
    OUT_DIR
    / "validation_sensors.txt"
)

test_path = (
    OUT_DIR
    / "heldout_test_sensors.txt"
)

split_path = (
    OUT_DIR
    / "pems_bay_sensor_splits.csv"
)

metadata_path = (
    OUT_DIR
    / "pems_bay_split_metadata.json"
)


save_list(
    train,
    train_path,
)

save_list(
    validation,
    validation_path,
)

save_list(
    heldout_test,
    test_path,
)

split_df.to_csv(
    split_path,
    index=False,
)


# ---------------------------------------------------------------------
# Distribution diagnostics
# ---------------------------------------------------------------------

summary = (
    split_df
    .groupby("split")
    .agg(
        n_sensors=(
            "sensor",
            "count",
        ),
        mean_speed=(
            "mean",
            "mean",
        ),
        median_speed=(
            "mean",
            "median",
        ),
        mean_sensor_sd=(
            "std",
            "mean",
        ),
        median_sensor_sd=(
            "std",
            "median",
        ),
    )
    .reset_index()
)


metadata = {
    "seed": SEED,
    "n_total": int(
        n_sensors
    ),
    "n_population_train": int(
        len(train)
    ),
    "n_validation": int(
        len(validation)
    ),
    "n_heldout_test": int(
        len(heldout_test)
    ),
}

with open(
    metadata_path,
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

print("=" * 78)
print("PersTSFM PEMS-BAY entity split")
print("=" * 78)

print()
print(
    f"Total sensors:       "
    f"{n_sensors}"
)

print(
    f"Population train:    "
    f"{len(train)}"
)

print(
    f"Validation:          "
    f"{len(validation)}"
)

print(
    f"Held-out test:       "
    f"{len(heldout_test)}"
)

print()
print(
    "Sensor distribution diagnostics:"
)

display = summary.copy()

for col in [
    "mean_speed",
    "median_speed",
    "mean_sensor_sd",
    "median_sensor_sd",
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
print("Saved:")
print(f"  {train_path}")
print(f"  {validation_path}")
print(f"  {test_path}")
print(f"  {split_path}")
print(f"  {metadata_path}")
