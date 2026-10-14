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

# 1% of installed peak capacity.
# Data are already in kW/kWp.
CLEAR_SKY_THRESHOLD = 0.01


def longest_true_run(mask):
    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if len(mask) == 0:
        return 0

    padded = np.concatenate(
        [
            [False],
            mask,
            [False],
        ]
    )

    changes = np.diff(
        padded.astype(np.int8)
    )

    starts = np.flatnonzero(
        changes == 1
    )

    ends = np.flatnonzero(
        changes == -1
    )

    if len(starts) == 0:
        return 0

    return int(
        np.max(
            ends - starts
        )
    )


rows = []
region_rows = []


print("=" * 100)
print("SOLAR DAYLIGHT / RECONSTRUCTABILITY AUDIT")
print("=" * 100)

print()
print(
    "Clear-sky threshold:",
    CLEAR_SKY_THRESHOLD,
    "kW/kWp",
)


with zipfile.ZipFile(
    ARCHIVE,
    "r",
) as zf:

    for region in REGIONS:

        print()
        print("=" * 100)
        print(region)
        print("=" * 100)


        qc_name = (
            f"PVpow_qc_trim_{region}.csv"
        )

        cs_name = (
            f"PVpow_cs_trim_{region}.csv"
        )


        print(
            "Loading QC..."
        )

        with zf.open(
            qc_name
        ) as handle:

            qc = pd.read_csv(
                handle,
                sep=";",
                na_values=[
                    "NA",
                ],
                dtype=np.float32,
            )


        print(
            "Loading clear-sky..."
        )

        with zf.open(
            cs_name
        ) as handle:

            cs = pd.read_csv(
                handle,
                sep=";",
                na_values=[
                    "NA",
                ],
                dtype=np.float32,
            )


        qc.columns = [
            str(x)
            for x in qc.columns
        ]

        cs.columns = [
            str(x)
            for x in cs.columns
        ]


        if qc.shape != cs.shape:
            raise RuntimeError(
                f"{region}: QC and clear-sky "
                f"shapes differ: "
                f"{qc.shape} vs {cs.shape}"
            )


        # The clear-sky matrices were exported with
        # generic R-style column names (V1, V2, ...),
        # whereas the QC matrices retain installation IDs.
        #
        # The matrices contain exactly the same number of
        # columns and are positionally aligned by installation.
        if qc.shape[1] != cs.shape[1]:
            raise RuntimeError(
                f"{region}: QC and clear-sky "
                f"column counts differ: "
                f"{qc.shape[1]} vs {cs.shape[1]}"
            )

        cs.columns = qc.columns


        print(
            "Shape:",
            qc.shape,
        )


        region_daylight_expected = 0
        region_daylight_observed = 0
        region_known_night = 0
        region_night_qc_missing = 0


        for entity_id in qc.columns:

            y = qc[
                entity_id
            ].to_numpy(
                dtype=np.float32
            )

            clear = cs[
                entity_id
            ].to_numpy(
                dtype=np.float32
            )


            qc_observed = np.isfinite(
                y
            )

            cs_observed = np.isfinite(
                clear
            )


            daylight = (
                cs_observed
                & (
                    clear
                    > CLEAR_SKY_THRESHOLD
                )
            )


            known_night = (
                cs_observed
                & (
                    clear
                    <= CLEAR_SKY_THRESHOLD
                )
            )


            daylight_expected = int(
                daylight.sum()
            )

            daylight_observed = int(
                (
                    daylight
                    & qc_observed
                ).sum()
            )


            if daylight_expected > 0:
                daylight_coverage = (
                    daylight_observed
                    / daylight_expected
                )
            else:
                daylight_coverage = (
                    np.nan
                )


            night_count = int(
                known_night.sum()
            )

            night_qc_missing = int(
                (
                    known_night
                    & ~qc_observed
                ).sum()
            )


            if night_count > 0:
                night_missing_fraction = (
                    night_qc_missing
                    / night_count
                )
            else:
                night_missing_fraction = (
                    np.nan
                )


            # A timestamp is reconstructable when:
            #
            # 1. QC provides a value, OR
            # 2. clear-sky generation is <=1%
            #    of peak, where a zero can be
            #    assigned without imputing a
            #    daylight measurement.
            reconstructable = (
                qc_observed
                | known_night
            )


            longest_run = (
                longest_true_run(
                    reconstructable
                )
            )


            rows.append(
                {
                    "region": region,
                    "entity_id": (
                        str(entity_id)
                    ),
                    "entity_key": (
                        f"{region}_{entity_id}"
                    ),
                    "n_steps": (
                        len(y)
                    ),
                    "qc_observed_steps": int(
                        qc_observed.sum()
                    ),
                    "qc_observed_fraction": (
                        float(
                            qc_observed.mean()
                        )
                    ),
                    "clear_sky_observed_fraction": (
                        float(
                            cs_observed.mean()
                        )
                    ),
                    "daylight_expected_steps": (
                        daylight_expected
                    ),
                    "daylight_observed_steps": (
                        daylight_observed
                    ),
                    "daylight_qc_coverage": (
                        daylight_coverage
                    ),
                    "known_night_steps": (
                        night_count
                    ),
                    "night_qc_missing_fraction": (
                        night_missing_fraction
                    ),
                    "reconstructable_fraction": (
                        float(
                            reconstructable.mean()
                        )
                    ),
                    "longest_reconstructable_run_steps": (
                        longest_run
                    ),
                    "longest_reconstructable_run_days": (
                        longest_run
                        / STEPS_PER_DAY
                    ),
                }
            )


            region_daylight_expected += (
                daylight_expected
            )

            region_daylight_observed += (
                daylight_observed
            )

            region_known_night += (
                night_count
            )

            region_night_qc_missing += (
                night_qc_missing
            )


        region_rows.append(
            {
                "region": region,
                "n_entities": (
                    qc.shape[1]
                ),
                "daylight_expected_steps": (
                    region_daylight_expected
                ),
                "daylight_observed_steps": (
                    region_daylight_observed
                ),
                "daylight_qc_coverage": (
                    region_daylight_observed
                    / region_daylight_expected
                    if region_daylight_expected
                    else np.nan
                ),
                "known_night_steps": (
                    region_known_night
                ),
                "night_qc_missing_fraction": (
                    region_night_qc_missing
                    / region_known_night
                    if region_known_night
                    else np.nan
                ),
            }
        )


        del qc
        del cs


entity = pd.DataFrame(
    rows
)

regions = pd.DataFrame(
    region_rows
)


entity.to_csv(
    OUT_DIR
    / "daylight_entity_audit.csv",
    index=False,
)

regions.to_csv(
    OUT_DIR
    / "daylight_region_audit.csv",
    index=False,
)


print()
print("=" * 100)
print("REGION SUMMARY")
print("=" * 100)

print(
    regions
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("ENTITY SUMMARY")
print("=" * 100)


for column in [
    "qc_observed_fraction",
    "clear_sky_observed_fraction",
    "daylight_qc_coverage",
    "night_qc_missing_fraction",
    "reconstructable_fraction",
    "longest_reconstructable_run_days",
]:

    print()
    print(column)

    print(
        entity[
            column
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
print("RECONSTRUCTABLE-RUN ELIGIBILITY")
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
                "longest_reconstructable_run_days"
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
print("DAYLIGHT QC COVERAGE")
print("=" * 100)


for threshold in [
    0.50,
    0.70,
    0.80,
    0.90,
    0.95,
]:

    values = entity[
        "daylight_qc_coverage"
    ]

    eligible = values.notna()

    n = int(
        (
            eligible
            & (
                values
                >= threshold
            )
        ).sum()
    )

    denominator = int(
        eligible.sum()
    )

    print(
        f">= {threshold:.0%}: "
        f"{n:>4} / {denominator} "
        f"({100 * n / denominator:.1f}%)"
    )


print()
print("=" * 100)
print("SAVED")
print("=" * 100)

print(
    OUT_DIR
    / "daylight_entity_audit.csv"
)

print(
    OUT_DIR
    / "daylight_region_audit.csv"
)
