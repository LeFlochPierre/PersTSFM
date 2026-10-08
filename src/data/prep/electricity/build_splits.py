from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42
MIN_ACTIVE_DAYS = 365.0

SUMMARY_PATH = Path(
    "results/tables/electricity_entity_summary.csv"
)

OUT_DIR = Path(
    "data/processed/electricity"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Load entity audit
# ---------------------------------------------------------------------

df = pd.read_csv(
    SUMMARY_PATH,
)

df["first_nonzero_timestamp"] = pd.to_datetime(
    df["first_nonzero_timestamp"]
)

df["last_nonzero_timestamp"] = pd.to_datetime(
    df["last_nonzero_timestamp"]
)

df["start_year"] = (
    df["first_nonzero_timestamp"].dt.year
)


# ---------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------

df["eligible"] = (
    df["active_span_days"]
    >= MIN_ACTIVE_DAYS
)

eligible = (
    df[df["eligible"]]
    .copy()
    .reset_index(drop=True)
)

excluded = (
    df[~df["eligible"]]
    .copy()
    .reset_index(drop=True)
)

print("=" * 70)
print("PersTSFM Electricity entity split")
print("=" * 70)

print()
print(f"Total entities:    {len(df)}")
print(f"Eligible entities: {len(eligible)}")
print(f"Excluded entities: {len(excluded)}")

print()
print("Eligible entities by start year:")
print(
    eligible["start_year"]
    .value_counts()
    .sort_index()
    .to_string()
)

print()
print("Excluded entities by start year:")
print(
    excluded["start_year"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ---------------------------------------------------------------------
# Entity-level split
#
# 70% population training
# 15% validation
# 15% completely held-out test
#
# Stratify by start year so calendar-entry cohorts are represented
# similarly across splits.
# ---------------------------------------------------------------------

train_df, temporary_df = train_test_split(
    eligible,
    test_size=0.30,
    random_state=SEED,
    shuffle=True,
    stratify=eligible["start_year"],
)

val_df, test_df = train_test_split(
    temporary_df,
    test_size=0.50,
    random_state=SEED,
    shuffle=True,
    stratify=temporary_df["start_year"],
)


# ---------------------------------------------------------------------
# Add split labels
# ---------------------------------------------------------------------

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "population_train"
val_df["split"] = "validation"
test_df["split"] = "heldout_test"

split_df = pd.concat(
    [
        train_df,
        val_df,
        test_df,
    ],
    ignore_index=True,
)

split_df = split_df.sort_values(
    "entity"
).reset_index(drop=True)


# ---------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------

train_entities = set(
    train_df["entity"]
)

val_entities = set(
    val_df["entity"]
)

test_entities = set(
    test_df["entity"]
)

assert train_entities.isdisjoint(
    val_entities
)

assert train_entities.isdisjoint(
    test_entities
)

assert val_entities.isdisjoint(
    test_entities
)

assert (
    len(train_entities)
    + len(val_entities)
    + len(test_entities)
    == len(eligible)
)

assert split_df["entity"].is_unique

assert (
    split_df["active_span_days"]
    >= MIN_ACTIVE_DAYS
).all()


# ---------------------------------------------------------------------
# Save entity lists
# ---------------------------------------------------------------------

def save_entity_list(
    entities,
    path,
):
    ordered = sorted(entities)

    path.write_text(
        "\n".join(ordered) + "\n",
        encoding="utf-8",
    )


save_entity_list(
    train_entities,
    OUT_DIR / "population_train_entities.txt",
)

save_entity_list(
    val_entities,
    OUT_DIR / "validation_entities.txt",
)

save_entity_list(
    test_entities,
    OUT_DIR / "heldout_test_entities.txt",
)

save_entity_list(
    set(eligible["entity"]),
    OUT_DIR / "eligible_entities.txt",
)

save_entity_list(
    set(excluded["entity"]),
    OUT_DIR / "excluded_entities.txt",
)


# ---------------------------------------------------------------------
# Save full metadata table
# ---------------------------------------------------------------------

split_df.to_csv(
    OUT_DIR / "electricity_entity_splits.csv",
    index=False,
)

excluded.to_csv(
    OUT_DIR / "excluded_entities.csv",
    index=False,
)


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------

metadata = {
    "seed": SEED,
    "eligibility": {
        "minimum_active_days": MIN_ACTIVE_DAYS,
        "definition": (
            "Time from first non-zero observation "
            "to last non-zero observation."
        ),
    },
    "n_total_entities": int(len(df)),
    "n_eligible_entities": int(len(eligible)),
    "n_excluded_entities": int(len(excluded)),
    "n_population_train": int(len(train_df)),
    "n_validation": int(len(val_df)),
    "n_heldout_test": int(len(test_df)),
    "split_fractions_actual": {
        "population_train": (
            len(train_df) / len(eligible)
        ),
        "validation": (
            len(val_df) / len(eligible)
        ),
        "heldout_test": (
            len(test_df) / len(eligible)
        ),
    },
    "stratification": "first_nonzero_start_year",
}

with open(
    OUT_DIR / "split_metadata.json",
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

print()
print("=" * 70)
print("Final split sizes")
print("=" * 70)

print(
    f"Population train: {len(train_df)} "
    f"({100 * len(train_df) / len(eligible):.1f}%)"
)

print(
    f"Validation:       {len(val_df)} "
    f"({100 * len(val_df) / len(eligible):.1f}%)"
)

print(
    f"Held-out test:    {len(test_df)} "
    f"({100 * len(test_df) / len(eligible):.1f}%)"
)

print()
print("Start-year composition:")

composition = pd.crosstab(
    split_df["split"],
    split_df["start_year"],
)

print(
    composition.to_string()
)

print()
print("Active-span summary by split:")

span_summary = (
    split_df
    .groupby("split")["active_span_days"]
    .agg(
        [
            "count",
            "min",
            "median",
            "mean",
            "max",
        ]
    )
)

print(
    span_summary.round(1).to_string()
)

print()
print("Excluded entities:")

print(
    excluded[
        [
            "entity",
            "first_nonzero_timestamp",
            "last_nonzero_timestamp",
            "active_span_days",
        ]
    ]
    .sort_values(
        "first_nonzero_timestamp"
    )
    .to_string(
        index=False
    )
)

print()
print("Saved split files to:")
print(f"  {OUT_DIR}")