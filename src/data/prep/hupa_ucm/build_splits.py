from pathlib import Path
import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

AUDIT_PATH = Path(
    "data/processed/hupa_ucm/"
    "hupa_ucm_participant_audit.csv"
)

OUT_DIR = Path(
    "data/processed/hupa_ucm"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 42

N_DEVELOPMENT = 20
N_HELDOUT_TEST = 5


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

audit = pd.read_csv(
    AUDIT_PATH
)

audit["person_id"] = (
    audit["person_id"]
    .astype(str)
)

if len(audit) != 25:
    raise RuntimeError(
        f"Expected 25 participants, "
        f"found {len(audit)}."
    )


# ---------------------------------------------------------------------
# Eligibility
#
# Require at least 7.75 continuous days.
#
# All audited HUPA participants should satisfy this.
# The small margin above 7 days gives us room for forecast origins.
# ---------------------------------------------------------------------

MIN_CONTINUOUS_DAYS = 7.75

eligible = audit[
    audit[
        "longest_regular_run_days"
    ]
    >= MIN_CONTINUOUS_DAYS
].copy()

excluded = audit[
    audit[
        "longest_regular_run_days"
    ]
    < MIN_CONTINUOUS_DAYS
].copy()


if len(eligible) != 25:
    raise RuntimeError(
        f"Expected all 25 HUPA participants "
        f"to be eligible, found "
        f"{len(eligible)}."
    )


# ---------------------------------------------------------------------
# Deterministic participant split
# ---------------------------------------------------------------------

people = (
    eligible[
        "person_id"
    ]
    .to_numpy()
)

rng = np.random.default_rng(
    SEED
)

shuffled = people.copy()

rng.shuffle(
    shuffled
)

development = shuffled[
    :N_DEVELOPMENT
]

heldout_test = shuffled[
    N_DEVELOPMENT:
]


if len(development) != N_DEVELOPMENT:
    raise RuntimeError(
        "Incorrect development count."
    )

if len(heldout_test) != N_HELDOUT_TEST:
    raise RuntimeError(
        "Incorrect held-out count."
    )

if (
    len(
        set(development)
        & set(heldout_test)
    )
    != 0
):
    raise RuntimeError(
        "Split overlap detected."
    )


# ---------------------------------------------------------------------
# Assign split
# ---------------------------------------------------------------------

split_map = {}

for person in development:
    split_map[person] = "development"

for person in heldout_test:
    split_map[person] = "heldout_test"


split_df = eligible.copy()

split_df["split"] = (
    split_df[
        "person_id"
    ]
    .map(split_map)
)


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

summary = (
    split_df
    .groupby(
        "split"
    )
    .agg(
        n_participants=(
            "person_id",
            "count",
        ),
        mean_span_days=(
            "span_days",
            "mean",
        ),
        median_span_days=(
            "span_days",
            "median",
        ),
        mean_glucose=(
            "mean_glucose",
            "mean",
        ),
        median_glucose=(
            "mean_glucose",
            "median",
        ),
        mean_glucose_sd=(
            "sd_glucose",
            "mean",
        ),
        median_glucose_sd=(
            "sd_glucose",
            "median",
        ),
    )
    .reset_index()
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

split_path = (
    OUT_DIR
    / "hupa_ucm_participant_splits.csv"
)

development_path = (
    OUT_DIR
    / "development_participants.txt"
)

test_path = (
    OUT_DIR
    / "heldout_test_participants.txt"
)

metadata_path = (
    OUT_DIR
    / "hupa_ucm_split_metadata.json"
)


split_df.to_csv(
    split_path,
    index=False,
)


with open(
    development_path,
    "w",
) as f:

    for person in sorted(
        development
    ):
        f.write(
            person + "\n"
        )


with open(
    test_path,
    "w",
) as f:

    for person in sorted(
        heldout_test
    ):
        f.write(
            person + "\n"
        )


metadata = {
    "seed": SEED,
    "min_continuous_days": (
        MIN_CONTINUOUS_DAYS
    ),
    "n_total": 25,
    "n_eligible": int(
        len(eligible)
    ),
    "n_excluded": int(
        len(excluded)
    ),
    "n_development": int(
        len(development)
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
print("PersTSFM HUPA-UCM participant split")
print("=" * 78)

print()
print(
    f"Total participants:    "
    f"{len(audit)}"
)

print(
    f"Eligible:              "
    f"{len(eligible)}"
)

print(
    f"Development:           "
    f"{len(development)}"
)

print(
    f"Held-out test:         "
    f"{len(heldout_test)}"
)


print()
print(
    "Distribution diagnostics:"
)

display = summary.copy()

for col in [
    "mean_span_days",
    "median_span_days",
    "mean_glucose",
    "median_glucose",
    "mean_glucose_sd",
    "median_glucose_sd",
]:
    display[col] = (
        display[col]
        .round(2)
    )

print()
print(
    display.to_string(
        index=False
    )
)


print()
print(
    "Development participants:"
)

print(
    ", ".join(
        sorted(
            development
        )
    )
)


print()
print(
    "Held-out test participants:"
)

print(
    ", ".join(
        sorted(
            heldout_test
        )
    )
)


print()
print("Saved:")
print(f"  {split_path}")
print(f"  {development_path}")
print(f"  {test_path}")
print(f"  {metadata_path}")
