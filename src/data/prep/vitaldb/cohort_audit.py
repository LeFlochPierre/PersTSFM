from pathlib import Path

import pandas as pd


AUDIT_DIR = Path(
    "results/vitaldb/audit"
)

clinical = pd.read_csv(
    AUDIT_DIR / "clinical_metadata.csv"
)

tracks = pd.read_csv(
    AUDIT_DIR / "available_tracks.csv"
)


# ------------------------------------------------------------------
# Basic durations
# ------------------------------------------------------------------

clinical["case_duration_h"] = (
    clinical["caseend"]
    - clinical["casestart"]
) / 3600.0

clinical["operation_duration_h"] = (
    clinical["opend"]
    - clinical["opstart"]
) / 3600.0

clinical["anaesthesia_duration_h"] = (
    clinical["aneend"]
    - clinical["anestart"]
) / 3600.0


# ------------------------------------------------------------------
# Track availability per case
# ------------------------------------------------------------------

availability = (
    tracks
    .assign(
        present=1
    )
    .pivot_table(
        index="caseid",
        columns="signal",
        values="present",
        aggfunc="max",
        fill_value=0,
    )
    .reset_index()
)


for col in [
    "heart_rate",
    "art_mbp",
    "nibp_mbp",
]:
    if col not in availability.columns:
        availability[col] = 0


availability["any_map"] = (
    (
        availability["art_mbp"]
        == 1
    )
    |
    (
        availability["nibp_mbp"]
        == 1
    )
).astype(int)


availability["all_three"] = (
    (
        availability["heart_rate"]
        == 1
    )
    & (
        availability["art_mbp"]
        == 1
    )
    & (
        availability["nibp_mbp"]
        == 1
    )
).astype(int)


df = clinical.merge(
    availability,
    on="caseid",
    how="left",
)

for col in [
    "heart_rate",
    "art_mbp",
    "nibp_mbp",
    "any_map",
    "all_three",
]:
    df[col] = (
        df[col]
        .fillna(0)
        .astype(int)
    )


# ------------------------------------------------------------------
# Basic cohort information
# ------------------------------------------------------------------

print("=" * 100)
print("VITALDB COHORT AUDIT")
print("=" * 100)

print()
print("Cases:", len(df))
print(
    "Unique subjects:",
    df["subjectid"].nunique(),
)

subject_counts = (
    df.groupby("subjectid")
    .size()
)

print(
    "Subjects with >1 case:",
    int(
        (subject_counts > 1).sum()
    ),
)

print(
    "Maximum cases for one subject:",
    int(
        subject_counts.max()
    ),
)

print()
print("Case duration (hours):")
print(
    df["case_duration_h"]
    .describe()
    .round(3)
    .to_string()
)

print()
print("Operation duration (hours):")
print(
    df["operation_duration_h"]
    .describe()
    .round(3)
    .to_string()
)


# ------------------------------------------------------------------
# Duration x track availability
# ------------------------------------------------------------------

thresholds = [
    2,
    3,
    4,
    5,
    6,
    8,
    10,
]

signals = [
    "heart_rate",
    "art_mbp",
    "nibp_mbp",
    "any_map",
    "all_three",
]


rows = []

for hours in thresholds:

    long_enough = (
        df["case_duration_h"]
        >= hours
    )

    for signal in signals:

        mask = (
            long_enough
            & (
                df[signal]
                == 1
            )
        )

        subset = df.loc[
            mask
        ]

        rows.append(
            {
                "minimum_case_hours": hours,
                "signal": signal,
                "n_cases": len(subset),
                "n_subjects": (
                    subset[
                        "subjectid"
                    ]
                    .nunique()
                ),
            }
        )


summary = pd.DataFrame(
    rows
)


print()
print("=" * 100)
print("CASES BY MINIMUM DURATION")
print("=" * 100)

pivot = summary.pivot(
    index="minimum_case_hours",
    columns="signal",
    values="n_cases",
)

print()
print(
    pivot
    .to_string()
)


# ------------------------------------------------------------------
# Save candidate cohorts
# ------------------------------------------------------------------

summary.to_csv(
    AUDIT_DIR
    / "cohort_duration_summary.csv",
    index=False,
)


for hours in [
    4,
    6,
    8,
]:

    subset = df[
        (
            df["art_mbp"]
            == 1
        )
        & (
            df["case_duration_h"]
            >= hours
        )
    ].copy()

    subset.to_csv(
        AUDIT_DIR
        / (
            f"art_mbp_cases_ge_{hours}h.csv"
        ),
        index=False,
    )


df.to_csv(
    AUDIT_DIR
    / "case_metadata_with_tracks.csv",
    index=False,
)


print()
print("=" * 100)
print("SAVED")
print("=" * 100)

print(
    AUDIT_DIR
    / "cohort_duration_summary.csv"
)

print(
    AUDIT_DIR
    / "case_metadata_with_tracks.csv"
)

for hours in [
    4,
    6,
    8,
]:
    print(
        AUDIT_DIR
        / f"art_mbp_cases_ge_{hours}h.csv"
    )
