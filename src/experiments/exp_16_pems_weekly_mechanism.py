from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RAW_PATH = Path(
    "data/raw/pems_bay/PEMS-BAY.csv"
)

PROTOCOL_PATH = Path(
    "data/processed/pems_bay/"
    "pems_bay_evaluation_origins.csv"
)

NAIVE_ENTITY_PATH = Path(
    "results/pems_bay/baselines/"
    "pems_bay_naive_entity_metrics.csv"
)

CHRONOS_ENTITY_PATH = Path(
    "results/pems_bay/context/"
    "chronos2_pems_bay_context_entity_metrics.csv"
)

TIMESFM_ENTITY_PATH = Path(
    "results/pems_bay/context/"
    "timesfm3_pems_bay_context_entity_metrics.csv"
)

OUT_DIR = Path(
    "results/pems_bay/mechanism"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SENSOR_PATH = (
    OUT_DIR
    / "pems_weekly_mechanism_sensor_metrics.csv"
)

CORRELATION_PATH = (
    OUT_DIR
    / "pems_weekly_mechanism_correlations.csv"
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def lagged_pairs(series, days):

    past = series.copy()

    past.index = (
        past.index
        + pd.Timedelta(
            days=days
        )
    )

    pair = pd.concat(
        [
            series.rename(
                "current"
            ),
            past.rename(
                "lagged"
            ),
        ],
        axis=1,
        join="inner",
    ).dropna()

    return pair


def pearson_corr(x, y):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan

    if np.std(x) == 0:
        return np.nan

    if np.std(y) == 0:
        return np.nan

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


def spearman_corr(x, y):

    x = pd.Series(
        x,
        dtype=float,
    )

    y = pd.Series(
        y,
        dtype=float,
    )

    mask = (
        x.notna()
        &
        y.notna()
    )

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan

    x_rank = x.rank(
        method="average"
    )

    y_rank = y.rank(
        method="average"
    )

    return pearson_corr(
        x_rank,
        y_rank,
    )


# ---------------------------------------------------------------------
# Validation sensors and pre-evaluation cutoff
# ---------------------------------------------------------------------

protocol = pd.read_csv(
    PROTOCOL_PATH,
    parse_dates=[
        "eval_origin",
    ],
)

protocol["sensor"] = (
    protocol["sensor"]
    .astype(str)
)

validation = protocol[
    protocol["split"]
    == "validation"
].copy()

sensors = sorted(
    validation[
        "sensor"
    ].unique()
)

first_eval_origin = (
    validation[
        "eval_origin"
    ].min()
)

history_end = (
    first_eval_origin
    - pd.Timedelta(
        minutes=5
    )
)


print("=" * 78)
print("PersTSFM PEMS-BAY weekly mechanism audit")
print("=" * 78)

print()
print(
    f"Validation sensors: "
    f"{len(sensors)}"
)

print(
    f"First evaluation:   "
    f"{first_eval_origin}"
)

print(
    f"Mechanism history ends: "
    f"{history_end}"
)


# ---------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------

header = pd.read_csv(
    RAW_PATH,
    nrows=0,
)

timestamp_col = header.columns[0]

raw = pd.read_csv(
    RAW_PATH,
    usecols=[
        timestamp_col,
        *sensors,
    ],
    low_memory=False,
)

raw[timestamp_col] = pd.to_datetime(
    raw[timestamp_col]
)

raw = (
    raw
    .set_index(timestamp_col)
    .sort_index()
)

raw = raw.loc[
    :history_end
]


# ---------------------------------------------------------------------
# Temporal structure per sensor
# ---------------------------------------------------------------------

structure_rows = []


for sensor in sensors:

    series = raw[
        sensor
    ].astype(float)

    sensor_sd = float(
        series.std()
    )

    pair_1d = lagged_pairs(
        series,
        days=1,
    )

    pair_7d = lagged_pairs(
        series,
        days=7,
    )

    corr_1d = pearson_corr(
        pair_1d["current"],
        pair_1d["lagged"],
    )

    corr_7d = pearson_corr(
        pair_7d["current"],
        pair_7d["lagged"],
    )

    mae_1d_repeat = float(
        np.mean(
            np.abs(
                pair_1d[
                    "current"
                ]
                - pair_1d[
                    "lagged"
                ]
            )
        )
    )

    mae_7d_repeat = float(
        np.mean(
            np.abs(
                pair_7d[
                    "current"
                ]
                - pair_7d[
                    "lagged"
                ]
            )
        )
    )

    if sensor_sd > 0:

        normalized_1d_repeat = (
            mae_1d_repeat
            / sensor_sd
        )

        normalized_7d_repeat = (
            mae_7d_repeat
            / sensor_sd
        )

    else:

        normalized_1d_repeat = (
            np.nan
        )

        normalized_7d_repeat = (
            np.nan
        )

    structure_rows.append(
        {
            "sensor": sensor,

            "sensor_sd": (
                sensor_sd
            ),

            "lag1d_corr": (
                corr_1d
            ),

            "lag7d_corr": (
                corr_7d
            ),

            "lag7_minus_lag1_corr": (
                corr_7d
                - corr_1d
            ),

            "repeat_mae_1d": (
                mae_1d_repeat
            ),

            "repeat_mae_7d": (
                mae_7d_repeat
            ),

            "normalized_repeat_mae_1d": (
                normalized_1d_repeat
            ),

            "normalized_repeat_mae_7d": (
                normalized_7d_repeat
            ),

            "weekly_repeat_advantage": (
                normalized_1d_repeat
                - normalized_7d_repeat
            ),
        }
    )


structure = pd.DataFrame(
    structure_rows
)


# ---------------------------------------------------------------------
# Naive baseline properties
# ---------------------------------------------------------------------

naive = pd.read_csv(
    NAIVE_ENTITY_PATH
)

naive["sensor"] = (
    naive["sensor"]
    .astype(str)
)

naive_wide = (
    naive
    .pivot(
        index="sensor",
        columns="method",
        values="mean_mae",
    )
    .reset_index()
)

naive_wide = naive_wide.rename(
    columns={
        "persistence": (
            "persistence_mae"
        ),
        "daily_seasonal": (
            "daily_naive_mae"
        ),
        "weekly_seasonal": (
            "weekly_naive_mae"
        ),
    }
)

naive_wide[
    "weekly_advantage_vs_persistence"
] = (
    naive_wide[
        "persistence_mae"
    ]
    - naive_wide[
        "weekly_naive_mae"
    ]
)

naive_wide[
    "weekly_advantage_vs_daily"
] = (
    naive_wide[
        "daily_naive_mae"
    ]
    - naive_wide[
        "weekly_naive_mae"
    ]
)


# ---------------------------------------------------------------------
# Model context gains
# ---------------------------------------------------------------------

def extract_model_gains(
    path,
    prefix,
):

    df = pd.read_csv(
        path
    )

    df["sensor"] = (
        df["sensor"]
        .astype(str)
    )

    wide = (
        df
        .pivot(
            index="sensor",
            columns="context_hours",
            values="mean_mae",
        )
    )

    required = [
        24,
        72,
        120,
        168,
        240,
        336,
    ]

    missing = [
        x
        for x in required
        if x not in wide.columns
    ]

    if missing:
        raise RuntimeError(
            f"{prefix}: missing context "
            f"hours {missing}"
        )

    result = pd.DataFrame(
        {
            "sensor": (
                wide.index
            ),

            f"{prefix}_mae_1d": (
                wide[24]
                .to_numpy()
            ),

            f"{prefix}_mae_3d": (
                wide[72]
                .to_numpy()
            ),

            f"{prefix}_mae_5d": (
                wide[120]
                .to_numpy()
            ),

            f"{prefix}_mae_7d": (
                wide[168]
                .to_numpy()
            ),

            f"{prefix}_mae_10d": (
                wide[240]
                .to_numpy()
            ),

            f"{prefix}_mae_14d": (
                wide[336]
                .to_numpy()
            ),
        }
    )

    result[
        f"{prefix}_gain_1d_to_7d"
    ] = (
        result[
            f"{prefix}_mae_1d"
        ]
        - result[
            f"{prefix}_mae_7d"
        ]
    )

    result[
        f"{prefix}_relative_gain_1d_to_7d"
    ] = (
        result[
            f"{prefix}_gain_1d_to_7d"
        ]
        / result[
            f"{prefix}_mae_1d"
        ]
    )

    result[
        f"{prefix}_gain_5d_to_7d"
    ] = (
        result[
            f"{prefix}_mae_5d"
        ]
        - result[
            f"{prefix}_mae_7d"
        ]
    )

    result[
        f"{prefix}_gain_1d_to_14d"
    ] = (
        result[
            f"{prefix}_mae_1d"
        ]
        - result[
            f"{prefix}_mae_14d"
        ]
    )

    return result


chronos = extract_model_gains(
    CHRONOS_ENTITY_PATH,
    "chronos",
)

timesfm = extract_model_gains(
    TIMESFM_ENTITY_PATH,
    "timesfm",
)


# ---------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------

sensor_metrics = (
    structure
    .merge(
        naive_wide,
        on="sensor",
        how="inner",
        validate="one_to_one",
    )
    .merge(
        chronos,
        on="sensor",
        how="inner",
        validate="one_to_one",
    )
    .merge(
        timesfm,
        on="sensor",
        how="inner",
        validate="one_to_one",
    )
)


if len(sensor_metrics) != len(
    sensors
):
    raise RuntimeError(
        f"Expected {len(sensors)} "
        f"merged sensors, "
        f"found {len(sensor_metrics)}."
    )


# ---------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------

mechanism_variables = [
    "lag1d_corr",
    "lag7d_corr",
    "lag7_minus_lag1_corr",
    "normalized_repeat_mae_1d",
    "normalized_repeat_mae_7d",
    "weekly_repeat_advantage",
    "weekly_advantage_vs_persistence",
    "weekly_advantage_vs_daily",
]

gain_variables = [
    "chronos_gain_1d_to_7d",
    "chronos_relative_gain_1d_to_7d",
    "chronos_gain_5d_to_7d",
    "chronos_gain_1d_to_14d",
    "timesfm_gain_1d_to_7d",
    "timesfm_relative_gain_1d_to_7d",
    "timesfm_gain_5d_to_7d",
    "timesfm_gain_1d_to_14d",
]


correlation_rows = []


for mechanism in mechanism_variables:

    for gain in gain_variables:

        pearson = pearson_corr(
            sensor_metrics[
                mechanism
            ],
            sensor_metrics[
                gain
            ],
        )

        spearman = spearman_corr(
            sensor_metrics[
                mechanism
            ],
            sensor_metrics[
                gain
            ],
        )

        correlation_rows.append(
            {
                "mechanism_variable": (
                    mechanism
                ),
                "gain_variable": (
                    gain
                ),
                "pearson_r": (
                    pearson
                ),
                "spearman_rho": (
                    spearman
                ),
                "abs_spearman": (
                    abs(spearman)
                    if np.isfinite(
                        spearman
                    )
                    else np.nan
                ),
            }
        )


correlations = pd.DataFrame(
    correlation_rows
)

correlations = correlations.sort_values(
    "abs_spearman",
    ascending=False,
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

sensor_metrics.to_csv(
    SENSOR_PATH,
    index=False,
)

correlations.to_csv(
    CORRELATION_PATH,
    index=False,
)


# ---------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("Pre-evaluation temporal structure")
print("=" * 78)

structure_display = pd.DataFrame(
    {
        "metric": [
            "lag1d_corr",
            "lag7d_corr",
            "normalized_repeat_mae_1d",
            "normalized_repeat_mae_7d",
            "weekly_repeat_advantage",
        ],
        "mean": [
            sensor_metrics[
                "lag1d_corr"
            ].mean(),

            sensor_metrics[
                "lag7d_corr"
            ].mean(),

            sensor_metrics[
                "normalized_repeat_mae_1d"
            ].mean(),

            sensor_metrics[
                "normalized_repeat_mae_7d"
            ].mean(),

            sensor_metrics[
                "weekly_repeat_advantage"
            ].mean(),
        ],
        "median": [
            sensor_metrics[
                "lag1d_corr"
            ].median(),

            sensor_metrics[
                "lag7d_corr"
            ].median(),

            sensor_metrics[
                "normalized_repeat_mae_1d"
            ].median(),

            sensor_metrics[
                "normalized_repeat_mae_7d"
            ].median(),

            sensor_metrics[
                "weekly_repeat_advantage"
            ].median(),
        ],
    }
)

structure_display[
    [
        "mean",
        "median",
    ]
] = (
    structure_display[
        [
            "mean",
            "median",
        ]
    ].round(4)
)

print()
print(
    structure_display.to_string(
        index=False
    )
)


# ---------------------------------------------------------------------
# Report gains
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("Entity-level context gains")
print("=" * 78)

gain_summary = pd.DataFrame(
    {
        "metric": [
            "Chronos 1d -> 7d",
            "Chronos 5d -> 7d",
            "Chronos 1d -> 14d",
            "TimesFM 1d -> 7d",
            "TimesFM 5d -> 7d",
            "TimesFM 1d -> 14d",
        ],

        "mean_gain": [
            sensor_metrics[
                "chronos_gain_1d_to_7d"
            ].mean(),

            sensor_metrics[
                "chronos_gain_5d_to_7d"
            ].mean(),

            sensor_metrics[
                "chronos_gain_1d_to_14d"
            ].mean(),

            sensor_metrics[
                "timesfm_gain_1d_to_7d"
            ].mean(),

            sensor_metrics[
                "timesfm_gain_5d_to_7d"
            ].mean(),

            sensor_metrics[
                "timesfm_gain_1d_to_14d"
            ].mean(),
        ],

        "median_gain": [
            sensor_metrics[
                "chronos_gain_1d_to_7d"
            ].median(),

            sensor_metrics[
                "chronos_gain_5d_to_7d"
            ].median(),

            sensor_metrics[
                "chronos_gain_1d_to_14d"
            ].median(),

            sensor_metrics[
                "timesfm_gain_1d_to_7d"
            ].median(),

            sensor_metrics[
                "timesfm_gain_5d_to_7d"
            ].median(),

            sensor_metrics[
                "timesfm_gain_1d_to_14d"
            ].median(),
        ],
    }
)

gain_summary[
    [
        "mean_gain",
        "median_gain",
    ]
] = (
    gain_summary[
        [
            "mean_gain",
            "median_gain",
        ]
    ].round(4)
)

print()
print(
    gain_summary.to_string(
        index=False
    )
)


# ---------------------------------------------------------------------
# Most informative correlations
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("Strongest weekly-structure/context-gain associations")
print("=" * 78)

top = correlations.head(
    20
).copy()

top[
    "pearson_r"
] = (
    top[
        "pearson_r"
    ].round(4)
)

top[
    "spearman_rho"
] = (
    top[
        "spearman_rho"
    ].round(4)
)

print()
print(
    top[
        [
            "mechanism_variable",
            "gain_variable",
            "pearson_r",
            "spearman_rho",
        ]
    ].to_string(
        index=False
    )
)


# ---------------------------------------------------------------------
# Focused correlations
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("Focused 1d -> 7d tests")
print("=" * 78)

focused_variables = [
    "lag7d_corr",
    "normalized_repeat_mae_7d",
    "weekly_repeat_advantage",
    "weekly_advantage_vs_persistence",
]

for mechanism in focused_variables:

    print()
    print(mechanism)

    for gain in [
        "chronos_gain_1d_to_7d",
        "timesfm_gain_1d_to_7d",
    ]:

        row = correlations[
            (
                correlations[
                    "mechanism_variable"
                ]
                == mechanism
            )
            &
            (
                correlations[
                    "gain_variable"
                ]
                == gain
            )
        ].iloc[0]

        print(
            f"  {gain}: "
            f"Pearson={row['pearson_r']:.3f}, "
            f"Spearman={row['spearman_rho']:.3f}"
        )


print()
print("Saved:")
print(f"  {SENSOR_PATH}")
print(f"  {CORRELATION_PATH}")
