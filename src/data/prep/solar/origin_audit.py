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

HORIZON_STEPS = 6       # 1 hour
CLEAR_SKY_THRESHOLD = 0.01

CONTEXT_DAYS = [
    4,
    7,
]

# Candidate origins must be separated by at least this much.
MIN_ORIGIN_SPACING_H = 12
MIN_ORIGIN_SPACING_STEPS = (
    MIN_ORIGIN_SPACING_H * 6
)


def choose_spaced_origins(
    candidate_idx,
    min_spacing,
):
    chosen = []

    for idx in candidate_idx:
        if (
            not chosen
            or idx - chosen[-1] >= min_spacing
        ):
            chosen.append(
                int(idx)
            )

    return chosen


rows = []


print("=" * 100)
print("SOLAR FORECAST-ORIGIN AUDIT")
print("=" * 100)

print()
print(
    "Forecast horizon:",
    HORIZON_STEPS,
    "steps = 1 hour",
)

print(
    "Minimum origin spacing:",
    MIN_ORIGIN_SPACING_H,
    "hours",
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

        time_name = (
            f"unixtime_final_{region}.csv"
        )


        with zf.open(
            qc_name
        ) as handle:

            qc = pd.read_csv(
                handle,
                sep=";",
                na_values=["NA"],
                dtype=np.float32,
            )


        with zf.open(
            cs_name
        ) as handle:

            cs = pd.read_csv(
                handle,
                sep=";",
                na_values=["NA"],
                dtype=np.float32,
            )


        with zf.open(
            time_name
        ) as handle:

            time = pd.read_csv(
                handle,
                sep=";",
            )


        qc.columns = [
            str(x)
            for x in qc.columns
        ]


        if qc.shape != cs.shape:
            raise RuntimeError(
                f"{region}: QC/CS "
                f"shape mismatch."
            )


        # Clear-sky matrix is positionally
        # aligned but uses generic V* names.
        cs.columns = qc.columns


        print(
            "Entities:",
            qc.shape[1],
        )

        print(
            "Timepoints:",
            qc.shape[0],
        )


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


            known_night = (
                cs_observed
                & (
                    clear
                    <= CLEAR_SKY_THRESHOLD
                )
            )


            reconstructable = (
                qc_observed
                | known_night
            )


            for context_days in CONTEXT_DAYS:

                context_steps = (
                    context_days
                    * STEPS_PER_DAY
                )


                # Rolling count of reconstructable
                # context points ending immediately
                # before each candidate origin.
                x = reconstructable.astype(
                    np.int32
                )

                csum = np.concatenate(
                    [
                        [0],
                        np.cumsum(x),
                    ]
                )


                candidates = []


                first_origin = (
                    context_steps
                )

                last_origin = (
                    len(y)
                    - HORIZON_STEPS
                )


                for origin in range(
                    first_origin,
                    last_origin + 1,
                ):

                    history_count = (
                        csum[origin]
                        - csum[
                            origin
                            - context_steps
                        ]
                    )


                    if (
                        history_count
                        != context_steps
                    ):
                        continue


                    future_slice = slice(
                        origin,
                        origin
                        + HORIZON_STEPS,
                    )


                    # Evaluation target must be
                    # genuinely observed, not
                    # reconstructed.
                    if not np.all(
                        qc_observed[
                            future_slice
                        ]
                    ):
                        continue


                    # Avoid trivial nighttime
                    # forecasts. Require meaningful
                    # clear-sky generation somewhere
                    # across the 1-hour horizon.
                    future_clear = clear[
                        future_slice
                    ]


                    if not np.all(
                        np.isfinite(
                            future_clear
                        )
                    ):
                        continue


                    if np.max(
                        future_clear
                    ) <= CLEAR_SKY_THRESHOLD:
                        continue


                    candidates.append(
                        origin
                    )


                candidates = np.asarray(
                    candidates,
                    dtype=int,
                )


                spaced = choose_spaced_origins(
                    candidates,
                    MIN_ORIGIN_SPACING_STEPS,
                )


                rows.append(
                    {
                        "region": region,
                        "entity_id": (
                            str(entity_id)
                        ),
                        "entity_key": (
                            f"{region}_"
                            f"{entity_id}"
                        ),
                        "context_days": (
                            context_days
                        ),
                        "n_candidate_origins": (
                            len(candidates)
                        ),
                        "n_spaced_origins": (
                            len(spaced)
                        ),
                        "first_origin": (
                            spaced[0]
                            if len(spaced)
                            else np.nan
                        ),
                        "last_origin": (
                            spaced[-1]
                            if len(spaced)
                            else np.nan
                        ),
                    }
                )


        del qc
        del cs


audit = pd.DataFrame(
    rows
)


audit.to_csv(
    OUT_DIR
    / "origin_eligibility.csv",
    index=False,
)


print()
print("=" * 100)
print("ORIGIN ELIGIBILITY")
print("=" * 100)


for context_days in CONTEXT_DAYS:

    sub = audit[
        audit[
            "context_days"
        ]
        == context_days
    ]

    print()
    print(
        f"MAX CONTEXT = "
        f"{context_days} DAYS"
    )

    print("-" * 100)


    for n_origins in [
        1,
        2,
        4,
        6,
        8,
        12,
    ]:

        eligible = sub[
            sub[
                "n_spaced_origins"
            ]
            >= n_origins
        ]

        print(
            f">= {n_origins:>2} origins: "
            f"{len(eligible):>4} / "
            f"{len(sub)} "
            f"({100 * len(eligible) / len(sub):.1f}%)"
        )


    print()
    print(
        "By region for >=4 origins:"
    )

    eligible = sub[
        sub[
            "n_spaced_origins"
        ]
        >= 4
    ]

    print(
        eligible[
            "region"
        ]
        .value_counts()
        .reindex(
            REGIONS,
            fill_value=0,
        )
        .to_string()
    )


    print()
    print(
        "By region for >=8 origins:"
    )

    eligible = sub[
        sub[
            "n_spaced_origins"
        ]
        >= 8
    ]

    print(
        eligible[
            "region"
        ]
        .value_counts()
        .reindex(
            REGIONS,
            fill_value=0,
        )
        .to_string()
    )


print()
print("=" * 100)
print("SPACED-ORIGIN DISTRIBUTION")
print("=" * 100)


for context_days in CONTEXT_DAYS:

    sub = audit[
        audit[
            "context_days"
        ]
        == context_days
    ]

    print()
    print(
        f"{context_days} days"
    )

    print(
        sub[
            "n_spaced_origins"
        ]
        .describe(
            percentiles=[
                0.25,
                0.5,
                0.75,
                0.9,
            ]
        )
        .round(2)
        .to_string()
    )


print()
print("Saved:")
print(
    OUT_DIR
    / "origin_eligibility.csv"
)
