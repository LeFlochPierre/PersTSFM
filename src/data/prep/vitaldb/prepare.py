from pathlib import Path

import numpy as np
import pandas as pd


AUDIT_DIR = Path(
    "results/vitaldb/audit"
)

OUT_DIR = Path(
    "data/processed/vitaldb"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TRACK_INDEX_URL = (
    "https://api.vitaldb.net/trks"
)

TRACK_DATA_BASE = (
    "https://api.vitaldb.net"
)

TRACK_NAME = (
    "Solar8000/ART_MBP"
)


# ---------------------------------------------------------------------
# Frozen protocol
# ---------------------------------------------------------------------

SEED = 42

RESOLUTION_MIN = 1

WINDOW_MIN = 300

CONTEXT_GRID_MIN = [
    30,
    60,
    120,
    180,
    240,
]

HORIZON_MIN = 15

ORIGIN_INDICES = [
    240,
    255,
    270,
    285,
]

# ART_MBP is sampled about every 2 seconds.
# Require at least 20 finite samples within a minute.
MIN_RAW_VALUES_PER_MINUTE = 20

# Deliberately broad artifact bounds.
# These are NOT clinical thresholds.
MAP_MIN = 20.0
MAP_MAX = 250.0


def longest_run(values):
    values = np.sort(
        np.unique(values)
    )

    if len(values) == 0:
        return None

    best_start = values[0]
    best_end = values[0]

    current_start = values[0]
    previous = values[0]

    for value in values[1:]:

        if value == previous + 1:
            pass
        else:
            if (
                previous - current_start
                >
                best_end - best_start
            ):
                best_start = (
                    current_start
                )
                best_end = previous

            current_start = value

        previous = value

    if (
        previous - current_start
        >
        best_end - best_start
    ):
        best_start = current_start
        best_end = previous

    return (
        int(best_start),
        int(best_end),
    )


print("=" * 100)
print("VITALDB PREPARATION")
print("=" * 100)


# ---------------------------------------------------------------------
# Candidate cases
# ---------------------------------------------------------------------

audit = pd.read_csv(
    AUDIT_DIR
    / "art_mbp_trace_audit_ge_6h.csv"
)

metadata = pd.read_csv(
    AUDIT_DIR
    / "case_metadata_with_tracks.csv"
)


candidate = audit[
    (
        audit["status"]
        == "ok"
    )
    & (
        audit[
            "longest_complete_run_h"
        ]
        >= 5.0
    )
].copy()


candidate = candidate[
    [
        "caseid",
        "subjectid",
    ]
].merge(
    metadata,
    on=[
        "caseid",
        "subjectid",
    ],
    how="left",
    validate="one_to_one",
)


print()
print(
    "Candidate cases from audit:",
    len(candidate),
)

print(
    "Candidate subjects:",
    candidate[
        "subjectid"
    ].nunique(),
)


# ---------------------------------------------------------------------
# Track IDs
# ---------------------------------------------------------------------

tracks = pd.read_csv(
    TRACK_INDEX_URL
)

tracks = tracks[
    tracks["tname"]
    == TRACK_NAME
][
    [
        "caseid",
        "tid",
    ]
].copy()


candidate = candidate.merge(
    tracks,
    on="caseid",
    how="inner",
    validate="one_to_one",
)


print(
    "Cases with ART_MBP track:",
    len(candidate),
)


# ---------------------------------------------------------------------
# Build one standardized 5-hour window per case
# ---------------------------------------------------------------------

series_rows = []
cohort_rows = []


for i, row in candidate.reset_index(
    drop=True
).iterrows():

    caseid = int(
        row["caseid"]
    )

    subjectid = int(
        row["subjectid"]
    )

    tid = row["tid"]

    if (
        i == 0
        or (i + 1) % 25 == 0
        or i + 1 == len(candidate)
    ):
        print(
            f"[{i + 1:>3}/"
            f"{len(candidate)}] "
            f"case {caseid}"
        )


    try:
        raw = pd.read_csv(
            f"{TRACK_DATA_BASE}/{tid}"
        )
    except Exception as exc:
        print(
            "  download failed:",
            exc,
        )
        continue


    if len(raw) == 0:
        continue


    time_col = raw.columns[0]
    value_col = raw.columns[1]


    t = pd.to_numeric(
        raw[time_col],
        errors="coerce",
    )

    y = pd.to_numeric(
        raw[value_col],
        errors="coerce",
    )


    finite = (
        np.isfinite(t)
        & np.isfinite(y)
    )

    t = t[finite].to_numpy()
    y = y[finite].to_numpy()


    if len(t) == 0:
        continue


    # Broad artifact removal only.
    plausible = (
        (y >= MAP_MIN)
        & (y <= MAP_MAX)
    )

    t = t[plausible]
    y = y[plausible]


    if len(t) == 0:
        continue


    minute = np.floor(
        t / 60.0
    ).astype(int)


    tmp = pd.DataFrame(
        {
            "minute": minute,
            "map": y,
        }
    )


    minute_df = (
        tmp
        .groupby(
            "minute",
            as_index=False,
        )
        .agg(
            map=(
                "map",
                "median",
            ),
            n_raw=(
                "map",
                "size",
            ),
        )
    )


    minute_df = minute_df[
        minute_df["n_raw"]
        >= MIN_RAW_VALUES_PER_MINUTE
    ].copy()


    run = longest_run(
        minute_df["minute"].to_numpy()
    )


    if run is None:
        continue


    run_start, run_end = run

    run_length = (
        run_end
        - run_start
        + 1
    )


    if run_length < WINDOW_MIN:
        continue


    # Select the central 5-hour block from
    # the longest valid contiguous run.
    extra = (
        run_length
        - WINDOW_MIN
    )

    window_start = (
        run_start
        + extra // 2
    )

    window_end = (
        window_start
        + WINDOW_MIN
        - 1
    )


    block = minute_df[
        (
            minute_df["minute"]
            >= window_start
        )
        & (
            minute_df["minute"]
            <= window_end
        )
    ].copy()


    block = block.sort_values(
        "minute"
    )


    if len(block) != WINDOW_MIN:
        continue


    expected = np.arange(
        window_start,
        window_end + 1,
    )


    if not np.array_equal(
        block["minute"].to_numpy(),
        expected,
    ):
        continue


    block["relative_minute"] = (
        np.arange(
            WINDOW_MIN
        )
    )


    for _, point in block.iterrows():

        series_rows.append(
            {
                "caseid": caseid,
                "subjectid": subjectid,
                "relative_minute": int(
                    point[
                        "relative_minute"
                    ]
                ),
                "map": float(
                    point["map"]
                ),
            }
        )


    cohort_rows.append(
        {
            "caseid": caseid,
            "subjectid": subjectid,
            "window_start_min": (
                int(window_start)
            ),
            "window_end_min": (
                int(window_end)
            ),
            "source_run_length_min": (
                int(run_length)
            ),
            "age": row.get(
                "age",
                np.nan,
            ),
            "sex": row.get(
                "sex",
                np.nan,
            ),
            "department": row.get(
                "department",
                np.nan,
            ),
            "optype": row.get(
                "optype",
                np.nan,
            ),
        }
    )


series = pd.DataFrame(
    series_rows
)

cohort = pd.DataFrame(
    cohort_rows
)


print()
print("=" * 100)
print("FINAL PREPROCESSED COHORT")
print("=" * 100)

print()
print(
    "Cases:",
    len(cohort),
)

print(
    "Subjects:",
    cohort[
        "subjectid"
    ].nunique(),
)

print(
    "Minute rows:",
    len(series),
)


if len(cohort) == 0:
    raise RuntimeError(
        "No cases survived preprocessing."
    )


expected_rows = (
    len(cohort)
    * WINDOW_MIN
)

if len(series) != expected_rows:
    raise RuntimeError(
        "Unexpected number of "
        "minute-level rows."
    )


# ---------------------------------------------------------------------
# Subject-level train / validation / held-out split
# ---------------------------------------------------------------------

subjects = np.array(
    sorted(
        cohort[
            "subjectid"
        ].unique()
    )
)


rng = np.random.default_rng(
    SEED
)

subjects = rng.permutation(
    subjects
)


n_subjects = len(
    subjects
)

n_train = int(
    np.floor(
        0.70
        * n_subjects
    )
)

n_val = int(
    np.floor(
        0.15
        * n_subjects
    )
)


train_subjects = set(
    subjects[
        :n_train
    ]
)

val_subjects = set(
    subjects[
        n_train:
        n_train + n_val
    ]
)

test_subjects = set(
    subjects[
        n_train + n_val:
    ]
)


def split_for_subject(
    subjectid,
):
    if subjectid in train_subjects:
        return "train"

    if subjectid in val_subjects:
        return "validation"

    if subjectid in test_subjects:
        return "heldout"

    raise RuntimeError(
        subjectid
    )


cohort["split"] = (
    cohort[
        "subjectid"
    ]
    .map(
        split_for_subject
    )
)


series = series.merge(
    cohort[
        [
            "caseid",
            "split",
        ]
    ],
    on="caseid",
    how="left",
    validate="many_to_one",
)


# ---------------------------------------------------------------------
# Evaluation origins
# ---------------------------------------------------------------------

origin_rows = []


for _, row in cohort.iterrows():

    caseid = int(
        row["caseid"]
    )

    subjectid = int(
        row["subjectid"]
    )

    split = row[
        "split"
    ]

    for eval_index, origin_idx in enumerate(
        ORIGIN_INDICES
    ):

        origin_rows.append(
            {
                "caseid": caseid,
                "subjectid": subjectid,
                "split": split,
                "eval_index": (
                    eval_index
                ),
                "origin_idx": (
                    origin_idx
                ),
                "horizon_min": (
                    HORIZON_MIN
                ),
                "max_context_min": (
                    max(
                        CONTEXT_GRID_MIN
                    )
                ),
            }
        )


origins = pd.DataFrame(
    origin_rows
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

series_path = (
    OUT_DIR
    / "vitaldb_art_mbp_1min.csv"
)

cohort_path = (
    OUT_DIR
    / "vitaldb_cohort.csv"
)

origins_path = (
    OUT_DIR
    / "vitaldb_evaluation_origins.csv"
)

split_path = (
    OUT_DIR
    / "vitaldb_subject_splits.csv"
)


series.to_csv(
    series_path,
    index=False,
)

cohort.to_csv(
    cohort_path,
    index=False,
)

origins.to_csv(
    origins_path,
    index=False,
)

cohort[
    [
        "subjectid",
        "split",
    ]
].drop_duplicates().to_csv(
    split_path,
    index=False,
)


print()
print("Cases by split:")
print(
    cohort[
        "split"
    ]
    .value_counts()
    .to_string()
)

print()
print("Subjects by split:")
print(
    cohort[
        [
            "subjectid",
            "split",
        ]
    ]
    .drop_duplicates()[
        "split"
    ]
    .value_counts()
    .to_string()
)

print()
print(
    "Origins by split:"
)

print(
    origins[
        "split"
    ]
    .value_counts()
    .to_string()
)


print()
print("=" * 100)
print("FROZEN VITALDB PROTOCOL")
print("=" * 100)

print()
print(
    "Target: ART_MBP"
)

print(
    "Resolution: 1 minute"
)

print(
    "Window: 300 minutes"
)

print(
    "Context grid:",
    CONTEXT_GRID_MIN,
)

print(
    "Horizon:",
    HORIZON_MIN,
    "minutes",
)

print(
    "Origins/case:",
    len(
        ORIGIN_INDICES
    ),
)

print(
    "Origin indices:",
    ORIGIN_INDICES,
)

print(
    "Split grouping: subjectid"
)

print()
print("Saved:")
print(series_path)
print(cohort_path)
print(origins_path)
print(split_path)
