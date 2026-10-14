from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


ARCHIVE = Path(
    "data/raw/solar/distributed_pv_australia.zip"
)

OUT_DIR = Path(
    "data/processed/solar"
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

SEED = 42

STEPS_PER_HOUR = 6
STEPS_PER_DAY = 144

HORIZON_STEPS = 6

CONTEXT_STEPS = {
    "6h": 36,
    "12h": 72,
    "1d": 144,
    "2d": 288,
    "4d": 576,
}

MAX_CONTEXT_STEPS = 576

N_ORIGINS = 4

MIN_ORIGIN_SPACING_STEPS = (
    12 * STEPS_PER_HOUR
)

CLEAR_SKY_THRESHOLD = 0.01


def spaced_candidates(
    candidates,
    min_spacing,
):
    chosen = []

    for idx in candidates:
        if (
            not chosen
            or idx - chosen[-1]
            >= min_spacing
        ):
            chosen.append(
                int(idx)
            )

    return chosen


def choose_four_origins(
    candidates,
):
    candidates = np.asarray(
        candidates,
        dtype=int,
    )

    if len(candidates) < N_ORIGINS:
        return None

    # Spread the four origins across the
    # available eligible period rather than
    # simply taking the first four.
    positions = np.linspace(
        0,
        len(candidates) - 1,
        N_ORIGINS,
    )

    positions = np.round(
        positions
    ).astype(int)

    selected = candidates[
        positions
    ]

    if len(
        np.unique(selected)
    ) != N_ORIGINS:
        return None

    return selected


print("=" * 100)
print("SOLAR PREPARATION")
print("=" * 100)


series_rows = []
origin_rows = []
entity_rows = []


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

        metadata_name = (
            f"metadata_final_{region}.csv"
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


        with zf.open(
            metadata_name
        ) as handle:

            metadata = pd.read_csv(
                handle,
                sep=";",
            )


        qc.columns = [
            str(x)
            for x in qc.columns
        ]

        metadata["IDs"] = (
            metadata["IDs"]
            .astype(str)
        )


        if qc.shape != cs.shape:
            raise RuntimeError(
                f"{region}: QC/CS "
                "shape mismatch."
            )


        # Clear-sky columns are positionally
        # aligned but have generic V* names.
        cs.columns = qc.columns


        datetimes = pd.to_datetime(
            time["GMT"],
            errors="raise",
        )


        region_kept = 0


        for entity_id in qc.columns:

            y_raw = qc[
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
                y_raw
            )

            clear_observed = np.isfinite(
                clear
            )


            known_night = (
                clear_observed
                & (
                    clear
                    <= CLEAR_SKY_THRESHOLD
                )
            )


            reconstructable = (
                qc_observed
                | known_night
            )


            # Reconstructed series:
            # real QC observation when available;
            # zero only for physically negligible
            # clear-sky generation.
            y = np.full(
                len(y_raw),
                np.nan,
                dtype=np.float32,
            )

            y[
                qc_observed
            ] = y_raw[
                qc_observed
            ]

            y[
                ~qc_observed
                & known_night
            ] = 0.0


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


            for origin in range(
                MAX_CONTEXT_STEPS,
                len(y)
                - HORIZON_STEPS
                + 1,
            ):

                history_count = (
                    csum[origin]
                    - csum[
                        origin
                        - MAX_CONTEXT_STEPS
                    ]
                )


                if (
                    history_count
                    != MAX_CONTEXT_STEPS
                ):
                    continue


                future = slice(
                    origin,
                    origin
                    + HORIZON_STEPS,
                )


                # Forecast target must be truly
                # observed QC data.
                if not np.all(
                    qc_observed[
                        future
                    ]
                ):
                    continue


                future_clear = clear[
                    future
                ]


                if not np.all(
                    np.isfinite(
                        future_clear
                    )
                ):
                    continue


                # Exclude trivial nighttime
                # forecasting windows.
                if (
                    np.max(
                        future_clear
                    )
                    <= CLEAR_SKY_THRESHOLD
                ):
                    continue


                candidates.append(
                    origin
                )


            spaced = spaced_candidates(
                candidates,
                MIN_ORIGIN_SPACING_STEPS,
            )


            selected = choose_four_origins(
                spaced
            )


            if selected is None:
                continue


            entity_key = (
                f"{region}_{entity_id}"
            )


            meta = metadata[
                metadata["IDs"]
                == entity_id
            ]

            if len(meta) != 1:
                raise RuntimeError(
                    f"{entity_key}: metadata "
                    f"rows={len(meta)}"
                )

            meta = meta.iloc[0]


            entity_rows.append(
                {
                    "entity_key": (
                        entity_key
                    ),
                    "region": region,
                    "entity_id": (
                        entity_id
                    ),
                    "capacity": (
                        meta.get(
                            "Capacity",
                            np.nan,
                        )
                    ),
                    "longitude": (
                        meta.get(
                            "Longitude",
                            np.nan,
                        )
                    ),
                    "latitude": (
                        meta.get(
                            "Latitude",
                            np.nan,
                        )
                    ),
                    "n_spaced_origins": (
                        len(spaced)
                    ),
                }
            )


            # Store only the union of data
            # required for the four selected
            # evaluation cases.
            required = np.zeros(
                len(y),
                dtype=bool,
            )


            for eval_index, origin in enumerate(
                selected
            ):

                start = (
                    origin
                    - MAX_CONTEXT_STEPS
                )

                end = (
                    origin
                    + HORIZON_STEPS
                )

                required[
                    start:end
                ] = True


                origin_rows.append(
                    {
                        "entity_key": (
                            entity_key
                        ),
                        "region": region,
                        "entity_id": (
                            entity_id
                        ),
                        "eval_index": (
                            eval_index
                        ),
                        "origin_idx_global": (
                            int(origin)
                        ),
                        "origin_time": (
                            datetimes.iloc[
                                origin
                            ]
                        ),
                        "horizon_steps": (
                            HORIZON_STEPS
                        ),
                        "max_context_steps": (
                            MAX_CONTEXT_STEPS
                        ),
                    }
                )


            indices = np.flatnonzero(
                required
            )


            for idx in indices:

                series_rows.append(
                    {
                        "entity_key": (
                            entity_key
                        ),
                        "region": region,
                        "entity_id": (
                            entity_id
                        ),
                        "global_idx": (
                            int(idx)
                        ),
                        "datetime": (
                            datetimes.iloc[
                                idx
                            ]
                        ),
                        "value": (
                            float(y[idx])
                        ),
                        "was_observed": (
                            bool(
                                qc_observed[
                                    idx
                                ]
                            )
                        ),
                        "was_night_reconstructed": (
                            bool(
                                (
                                    not qc_observed[
                                        idx
                                    ]
                                )
                                and known_night[
                                    idx
                                ]
                            )
                        ),
                        "clear_sky": (
                            float(
                                clear[idx]
                            )
                            if np.isfinite(
                                clear[idx]
                            )
                            else np.nan
                        ),
                    }
                )


            region_kept += 1


        print(
            "Eligible entities:",
            region_kept,
        )


        del qc
        del cs


entities = pd.DataFrame(
    entity_rows
)

origins = pd.DataFrame(
    origin_rows
)

series = pd.DataFrame(
    series_rows
)


print()
print("=" * 100)
print("ELIGIBLE COHORT")
print("=" * 100)

print()
print(
    "Entities:",
    len(entities),
)

print()
print(
    "By region:"
)

print(
    entities[
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
    "Origins:",
    len(origins),
)


# ---------------------------------------------------------------------
# Region-stratified entity split
# ---------------------------------------------------------------------

rng = np.random.default_rng(
    SEED
)

split_rows = []


for region in REGIONS:

    keys = (
        entities[
            entities[
                "region"
            ]
            == region
        ][
            "entity_key"
        ]
        .sort_values()
        .to_numpy()
    )


    keys = rng.permutation(
        keys
    )


    n = len(keys)

    n_train = int(
        np.floor(
            0.70 * n
        )
    )

    n_val = int(
        np.floor(
            0.15 * n
        )
    )


    for i, key in enumerate(
        keys
    ):

        if i < n_train:
            split = "train"

        elif i < (
            n_train
            + n_val
        ):
            split = "validation"

        else:
            split = "heldout"


        split_rows.append(
            {
                "entity_key": key,
                "region": region,
                "split": split,
            }
        )


splits = pd.DataFrame(
    split_rows
)


entities = entities.merge(
    splits[
        [
            "entity_key",
            "split",
        ]
    ],
    on="entity_key",
    how="left",
    validate="one_to_one",
)

origins = origins.merge(
    splits[
        [
            "entity_key",
            "split",
        ]
    ],
    on="entity_key",
    how="left",
    validate="many_to_one",
)

series = series.merge(
    splits[
        [
            "entity_key",
            "split",
        ]
    ],
    on="entity_key",
    how="left",
    validate="many_to_one",
)


print()
print("=" * 100)
print("SPLITS")
print("=" * 100)

print()
print(
    pd.crosstab(
        entities[
            "region"
        ],
        entities[
            "split"
        ],
    ).to_string()
)


print()
print("Origins by split:")

print(
    origins[
        "split"
    ]
    .value_counts()
    .to_string()
)


# ---------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------

counts = (
    origins
    .groupby(
        "entity_key"
    )
    .size()
)

if not np.all(
    counts.to_numpy()
    == N_ORIGINS
):
    raise RuntimeError(
        "Not every entity has "
        "exactly four origins."
    )


if series[
    "value"
].isna().any():
    raise RuntimeError(
        "Stored analysis series "
        "contains missing values."
    )


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

series_path = (
    OUT_DIR
    / "solar_series.csv"
)

entities_path = (
    OUT_DIR
    / "solar_entities.csv"
)

origins_path = (
    OUT_DIR
    / "solar_evaluation_origins.csv"
)

splits_path = (
    OUT_DIR
    / "solar_entity_splits.csv"
)


series.to_csv(
    series_path,
    index=False,
)

entities.to_csv(
    entities_path,
    index=False,
)

origins.to_csv(
    origins_path,
    index=False,
)

splits.to_csv(
    splits_path,
    index=False,
)


print()
print("=" * 100)
print("FROZEN SOLAR PROTOCOL")
print("=" * 100)

print()
print(
    "Resolution: 10 minutes"
)

print(
    "Forecast horizon: 1 hour"
)

print(
    "Context grid:",
    CONTEXT_STEPS,
)

print(
    "Max context: 4 days"
)

print(
    "Origins/entity: 4"
)

print(
    "Minimum candidate spacing: "
    "12 hours"
)

print(
    "Splitting: entity-level, "
    "stratified by region"
)

print(
    "Primary donor matching: "
    "within-region"
)

print()
print("Saved:")
print(series_path)
print(entities_path)
print(origins_path)
print(splits_path)
