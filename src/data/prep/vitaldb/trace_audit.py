from pathlib import Path

import numpy as np
import pandas as pd


AUDIT_DIR = Path(
    "results/vitaldb/audit"
)

COHORT_PATH = (
    AUDIT_DIR
    / "art_mbp_cases_ge_6h.csv"
)

TRACK_NAME = "Solar8000/ART_MBP"

TRACK_INDEX_URL = (
    "https://api.vitaldb.net/trks"
)

TRACK_DATA_BASE = (
    "https://api.vitaldb.net"
)


def longest_consecutive_run(values):
    if len(values) == 0:
        return 0

    values = np.sort(
        np.unique(values)
    )

    best = 1
    current = 1

    for i in range(
        1,
        len(values),
    ):
        if (
            values[i]
            == values[i - 1] + 1
        ):
            current += 1
            best = max(
                best,
                current,
            )
        else:
            current = 1

    return best


print("=" * 100)
print("VITALDB ART_MBP TRACE AUDIT")
print("=" * 100)


cohort = pd.read_csv(
    COHORT_PATH
)

print()
print(
    "Candidate cases:",
    len(cohort),
)


tracks = pd.read_csv(
    TRACK_INDEX_URL
)

tracks = tracks[
    tracks["tname"]
    == TRACK_NAME
].copy()


tracks = tracks[
    tracks["caseid"].isin(
        cohort["caseid"]
    )
].copy()


print(
    "ART_MBP tracks found:",
    len(tracks),
)


if len(tracks) != len(cohort):
    missing = set(
        cohort["caseid"]
    ) - set(
        tracks["caseid"]
    )

    print(
        "Missing track IDs:",
        len(missing),
    )


meta_cols = [
    "caseid",
    "subjectid",
    "case_duration_h",
    "operation_duration_h",
    "opstart",
    "opend",
]

metadata = cohort[
    meta_cols
].copy()


tracks = tracks.merge(
    metadata,
    on="caseid",
    how="left",
    validate="one_to_one",
)


rows = []


for i, row in tracks.reset_index(
    drop=True
).iterrows():

    caseid = int(
        row["caseid"]
    )

    tid = row["tid"]

    url = (
        f"{TRACK_DATA_BASE}/{tid}"
    )

    if (
        i == 0
        or (i + 1) % 25 == 0
        or i + 1 == len(tracks)
    ):
        print(
            f"[{i + 1:>3}/{len(tracks)}] "
            f"case {caseid}"
        )

    try:
        data = pd.read_csv(
            url
        )
    except Exception as exc:
        rows.append(
            {
                "caseid": caseid,
                "subjectid": (
                    row["subjectid"]
                ),
                "status": "download_failed",
                "error": str(exc),
            }
        )
        continue


    if len(data) == 0:
        rows.append(
            {
                "caseid": caseid,
                "subjectid": (
                    row["subjectid"]
                ),
                "status": "empty",
            }
        )
        continue


    # VitalDB numeric tracks contain
    # time and value columns.
    time_col = data.columns[0]
    value_col = data.columns[1]

    t = pd.to_numeric(
        data[time_col],
        errors="coerce",
    ).to_numpy()

    y = pd.to_numeric(
        data[value_col],
        errors="coerce",
    ).to_numpy()

    finite = (
        np.isfinite(t)
        & np.isfinite(y)
    )

    t = t[finite]
    y = y[finite]


    if len(t) == 0:
        rows.append(
            {
                "caseid": caseid,
                "subjectid": (
                    row["subjectid"]
                ),
                "status": "no_finite_data",
            }
        )
        continue


    order = np.argsort(t)

    t = t[order]
    y = y[order]


    dt = np.diff(t)

    positive_dt = dt[
        dt > 0
    ]


    # --------------------------------------------------------------
    # Minute-level coverage.
    #
    # A minute is counted as observed when it contains at least one
    # recorded ART_MBP value. No interpolation is performed here.
    # --------------------------------------------------------------

    minute_index = np.floor(
        t / 60.0
    ).astype(int)

    observed_minutes = np.unique(
        minute_index
    )

    n_case_minutes = int(
        np.ceil(
            row["case_duration_h"]
            * 60.0
        )
    )

    minute_coverage = (
        len(observed_minutes)
        / n_case_minutes
        if n_case_minutes > 0
        else np.nan
    )

    longest_run_min = (
        longest_consecutive_run(
            observed_minutes
        )
    )


    # --------------------------------------------------------------
    # Descriptive value checks only.
    #
    # We are NOT filtering on these thresholds yet.
    # --------------------------------------------------------------

    outside_20_200 = np.mean(
        (y < 20)
        | (y > 200)
    )


    rows.append(
        {
            "caseid": caseid,
            "subjectid": (
                row["subjectid"]
            ),
            "status": "ok",
            "case_duration_h": (
                row[
                    "case_duration_h"
                ]
            ),
            "operation_duration_h": (
                row[
                    "operation_duration_h"
                ]
            ),
            "n_raw_values": len(y),
            "first_time_min": (
                np.min(t) / 60.0
            ),
            "last_time_min": (
                np.max(t) / 60.0
            ),
            "observed_span_h": (
                (
                    np.max(t)
                    - np.min(t)
                )
                / 3600.0
            ),
            "median_interval_s": (
                np.median(
                    positive_dt
                )
                if len(
                    positive_dt
                ) > 0
                else np.nan
            ),
            "p95_interval_s": (
                np.quantile(
                    positive_dt,
                    0.95,
                )
                if len(
                    positive_dt
                ) > 0
                else np.nan
            ),
            "max_interval_s": (
                np.max(
                    positive_dt
                )
                if len(
                    positive_dt
                ) > 0
                else np.nan
            ),
            "observed_minutes": (
                len(
                    observed_minutes
                )
            ),
            "minute_coverage": (
                minute_coverage
            ),
            "longest_complete_run_min": (
                longest_run_min
            ),
            "longest_complete_run_h": (
                longest_run_min
                / 60.0
            ),
            "map_q01": (
                np.quantile(
                    y,
                    0.01,
                )
            ),
            "map_median": (
                np.median(y)
            ),
            "map_q99": (
                np.quantile(
                    y,
                    0.99,
                )
            ),
            "fraction_outside_20_200": (
                outside_20_200
            ),
        }
    )


audit = pd.DataFrame(
    rows
)


out_path = (
    AUDIT_DIR
    / "art_mbp_trace_audit_ge_6h.csv"
)

audit.to_csv(
    out_path,
    index=False,
)


ok = audit[
    audit["status"]
    == "ok"
].copy()


print()
print("=" * 100)
print("TRACE SUMMARY")
print("=" * 100)

print()
print(
    "Successful:",
    len(ok),
)

print(
    "Failed/empty:",
    len(audit) - len(ok),
)


for column in [
    "minute_coverage",
    "longest_complete_run_h",
    "median_interval_s",
    "p95_interval_s",
    "fraction_outside_20_200",
]:

    print()
    print(column)

    print(
        ok[column]
        .describe(
            percentiles=[
                0.1,
                0.25,
                0.5,
                0.75,
                0.9,
            ]
        )
        .round(4)
        .to_string()
    )


print()
print("=" * 100)
print("CASES WITH LONG CONTIGUOUS MINUTE-LEVEL RUNS")
print("=" * 100)


for hours in [
    2,
    3,
    4,
    4.25,
    5,
    6,
]:

    n = (
        ok[
            "longest_complete_run_h"
        ]
        >= hours
    ).sum()

    print(
        f">= {hours:>4g} h: "
        f"{n:>4} / {len(ok)} "
        f"({100 * n / len(ok):.1f}%)"
    )


print()
print("=" * 100)
print("MINUTE COVERAGE")
print("=" * 100)


for threshold in [
    0.80,
    0.90,
    0.95,
    0.98,
]:

    n = (
        ok[
            "minute_coverage"
        ]
        >= threshold
    ).sum()

    print(
        f">= {threshold:.0%}: "
        f"{n:>4} / {len(ok)} "
        f"({100 * n / len(ok):.1f}%)"
    )


print()
print("Saved:")
print(out_path)
