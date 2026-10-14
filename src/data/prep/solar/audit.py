from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


ARCHIVE = Path(
    "data/raw/solar/distributed_pv_australia.zip"
)

OUT_DIR = Path(
    "results/solar/audit"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REGIONS = [
    "ACT",
    "SA",
    "WA",
]

STEPS_PER_DAY = 144


def longest_true_run(mask):
    idx = np.flatnonzero(
        mask
    )

    if len(idx) == 0:
        return 0

    if len(idx) == 1:
        return 1

    gaps = np.where(
        np.diff(idx) > 1
    )[0]

    starts = np.r_[
        0,
        gaps + 1,
    ]

    ends = np.r_[
        gaps,
        len(idx) - 1,
    ]

    lengths = (
        idx[ends]
        - idx[starts]
        + 1
    )

    return int(
        lengths.max()
    )


entity_rows = []
region_rows = []


print("=" * 100)
print("AUSTRALIAN DISTRIBUTED PV AUDIT")
print("=" * 100)


with zipfile.ZipFile(
    ARCHIVE,
    "r",
) as zf:

    for region in REGIONS:

        print()
        print("=" * 100)
        print(region)
        print("=" * 100)

        metadata_name = (
            f"metadata_final_{region}.csv"
        )

        time_name = (
            f"unixtime_final_{region}.csv"
        )

        power_name = (
            f"PVpow_qc_trim_{region}.csv"
        )


        # -------------------------------------------------------------
        # Metadata
        # -------------------------------------------------------------

        with zf.open(
            metadata_name
        ) as handle:

            metadata = pd.read_csv(
                handle,
                sep=";",
            )


        metadata["IDs"] = (
            metadata["IDs"]
            .astype(str)
        )


        # -------------------------------------------------------------
        # Time grid
        # -------------------------------------------------------------

        with zf.open(
            time_name
        ) as handle:

            time = pd.read_csv(
                handle,
                sep=";",
            )


        time["datetime"] = pd.to_datetime(
            time["GMT"],
            errors="raise",
        )


        unix = (
            time["Unixtime"]
            .to_numpy(
                dtype=np.int64
            )
        )


        delta_s = np.diff(
            unix
        )


        if len(delta_s) == 0:
            raise RuntimeError(
                f"No time intervals for {region}."
            )


        median_interval_s = float(
            np.median(
                delta_s
            )
        )

        irregular_intervals = int(
            np.sum(
                delta_s
                != median_interval_s
            )
        )


        # -------------------------------------------------------------
        # QC power matrix
        # -------------------------------------------------------------

        print(
            "Loading QC matrix..."
        )

        with zf.open(
            power_name
        ) as handle:

            power = pd.read_csv(
                handle,
                sep=";",
                na_values=[
                    "NA",
                ],
                dtype=np.float32,
            )


        power.columns = [
            str(col)
            for col in power.columns
        ]


        print(
            "Power shape:",
            power.shape,
        )

        print(
            "Metadata entities:",
            len(metadata),
        )

        print(
            "Timestamp rows:",
            len(time),
        )


        # -------------------------------------------------------------
        # Structural checks
        # -------------------------------------------------------------

        if len(power) != len(time):
            raise RuntimeError(
                f"{region}: power/time "
                f"row mismatch."
            )


        metadata_ids = set(
            metadata["IDs"]
        )

        power_ids = set(
            power.columns
        )


        missing_metadata = (
            power_ids
            - metadata_ids
        )

        missing_power = (
            metadata_ids
            - power_ids
        )


        print(
            "Power IDs missing metadata:",
            len(missing_metadata),
        )

        print(
            "Metadata IDs missing power:",
            len(missing_power),
        )


        if (
            missing_metadata
            or missing_power
        ):
            raise RuntimeError(
                f"{region}: metadata/"
                f"power ID mismatch."
            )


        # -------------------------------------------------------------
        # Vectorized summaries
        # -------------------------------------------------------------

        q99 = power.quantile(
            0.99
        )

        maxima = power.max(
            axis=0
        )

        minima = power.min(
            axis=0
        )


        # -------------------------------------------------------------
        # Per-entity audit
        # -------------------------------------------------------------

        for entity_id in power.columns:

            values = (
                power[entity_id]
                .to_numpy(
                    dtype=np.float32
                )
            )


            valid = np.isfinite(
                values
            )

            n_valid = int(
                valid.sum()
            )

            if n_valid == 0:

                first_idx = np.nan
                last_idx = np.nan
                active_span = 0
                active_completeness = np.nan
                zero_fraction = np.nan
                longest_run = 0

            else:

                valid_idx = np.flatnonzero(
                    valid
                )

                first_idx = int(
                    valid_idx[0]
                )

                last_idx = int(
                    valid_idx[-1]
                )

                active_span = (
                    last_idx
                    - first_idx
                    + 1
                )

                active_completeness = (
                    n_valid
                    / active_span
                )

                zero_fraction = float(
                    np.mean(
                        values[valid]
                        == 0
                    )
                )

                longest_run = (
                    longest_true_run(
                        valid
                    )
                )


            meta = metadata[
                metadata["IDs"]
                == entity_id
            ].iloc[0]


            capacity = pd.to_numeric(
                meta.get(
                    "Capacity",
                    np.nan,
                ),
                errors="coerce",
            )


            q99_value = float(
                q99[
                    entity_id
                ]
            )

            if (
                np.isfinite(capacity)
                and capacity > 0
            ):
                q99_capacity_ratio = (
                    q99_value
                    / capacity
                )
            else:
                q99_capacity_ratio = (
                    np.nan
                )


            if n_valid > 0:

                first_time = time.iloc[
                    first_idx
                ]["datetime"]

                last_time = time.iloc[
                    last_idx
                ]["datetime"]

            else:

                first_time = pd.NaT
                last_time = pd.NaT


            entity_rows.append(
                {
                    "region": region,
                    "entity_id": (
                        entity_id
                    ),
                    "entity_key": (
                        f"{region}_"
                        f"{entity_id}"
                    ),
                    "capacity": (
                        capacity
                    ),
                    "n_total": (
                        len(values)
                    ),
                    "n_valid": (
                        n_valid
                    ),
                    "missing_fraction": (
                        1.0
                        - n_valid
                        / len(values)
                    ),
                    "first_valid_time": (
                        first_time
                    ),
                    "last_valid_time": (
                        last_time
                    ),
                    "active_span_steps": (
                        active_span
                    ),
                    "active_span_days": (
                        active_span
                        / STEPS_PER_DAY
                    ),
                    "active_completeness": (
                        active_completeness
                    ),
                    "zero_fraction_nonmissing": (
                        zero_fraction
                    ),
                    "longest_complete_run_steps": (
                        longest_run
                    ),
                    "longest_complete_run_days": (
                        longest_run
                        / STEPS_PER_DAY
                    ),
                    "power_min": float(
                        minima[
                            entity_id
                        ]
                    ),
                    "power_q99": (
                        q99_value
                    ),
                    "power_max": float(
                        maxima[
                            entity_id
                        ]
                    ),
                    "q99_capacity_ratio": (
                        q99_capacity_ratio
                    ),
                    "installation_date": (
                        meta.get(
                            "Installation_Date",
                            np.nan,
                        )
                    ),
                    "azimuth_given": (
                        meta.get(
                            "Azimuth_Giv",
                            np.nan,
                        )
                    ),
                    "tilt_given": (
                        meta.get(
                            "Tilt_Giv",
                            np.nan,
                        )
                    ),
                    "azimuth_sim": (
                        meta.get(
                            "Azimuth_Sim",
                            np.nan,
                        )
                    ),
                    "tilt_sim": (
                        meta.get(
                            "Tilt_Sim",
                            np.nan,
                        )
                    ),
                }
            )


        region_rows.append(
            {
                "region": region,
                "n_entities": (
                    power.shape[1]
                ),
                "n_timepoints": (
                    power.shape[0]
                ),
                "start_time": (
                    time[
                        "datetime"
                    ].iloc[0]
                ),
                "end_time": (
                    time[
                        "datetime"
                    ].iloc[-1]
                ),
                "median_interval_s": (
                    median_interval_s
                ),
                "irregular_intervals": (
                    irregular_intervals
                ),
            }
        )


        del power


entity = pd.DataFrame(
    entity_rows
)

regions = pd.DataFrame(
    region_rows
)


entity.to_csv(
    OUT_DIR
    / "entity_audit.csv",
    index=False,
)

regions.to_csv(
    OUT_DIR
    / "region_audit.csv",
    index=False,
)


# ---------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------

print()
print("=" * 100)
print("REGION SUMMARY")
print("=" * 100)

print(
    regions.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("ENTITY SUMMARY")
print("=" * 100)

print()
print(
    "Total entities:",
    len(entity),
)


for variable in [
    "missing_fraction",
    "active_span_days",
    "active_completeness",
    "zero_fraction_nonmissing",
    "longest_complete_run_days",
    "capacity",
    "power_q99",
    "q99_capacity_ratio",
]:

    print()
    print(variable)

    print(
        entity[
            variable
        ]
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
print("COMPLETE-RUN ELIGIBILITY")
print("=" * 100)


for days in [
    1,
    2,
    4,
    7,
    14,
    21,
    28,
    42,
    56,
]:

    n = int(
        (
            entity[
                "longest_complete_run_days"
            ]
            >= days
        ).sum()
    )

    print(
        f">= {days:>2} days: "
        f"{n:>4} / {len(entity)} "
        f"({100 * n / len(entity):.1f}%)"
    )


print()
print("=" * 100)
print("ACTIVE-SPAN COMPLETENESS")
print("=" * 100)


for threshold in [
    0.80,
    0.90,
    0.95,
    0.98,
    0.99,
]:

    n = int(
        (
            entity[
                "active_completeness"
            ]
            >= threshold
        ).sum()
    )

    print(
        f">= {threshold:.0%}: "
        f"{n:>4} / {len(entity)} "
        f"({100 * n / len(entity):.1f}%)"
    )


print()
print("=" * 100)
print("SAVED")
print("=" * 100)

print(
    OUT_DIR
    / "entity_audit.csv"
)

print(
    OUT_DIR
    / "region_audit.csv"
)
