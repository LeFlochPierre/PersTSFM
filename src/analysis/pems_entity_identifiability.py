from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.registry import load_dataset


INPUT = Path(
    "results/pems_bay/information_acquisition/"
    "sensor_mechanism_curves.csv"
)

OUT_DIR = Path(
    "results/pems_bay/entity_identifiability"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def lag_correlation(
    series,
    lag,
):
    current = series.rename(
        "current"
    )

    lagged = series.copy()

    lagged.index = (
        lagged.index
        + lag
    )

    lagged = lagged.rename(
        "lagged"
    )

    joined = pd.concat(
        [
            current,
            lagged,
        ],
        axis=1,
        join="inner",
    ).dropna()

    if len(joined) < 10:
        return np.nan

    return float(
        joined[
            "current"
        ].corr(
            joined[
                "lagged"
            ]
        )
    )


def spectral_entropy(
    values,
):
    x = np.asarray(
        values,
        dtype=float,
    )

    x = (
        x
        - np.mean(x)
    )

    power = (
        np.abs(
            np.fft.rfft(x)
        )
        ** 2
    )

    if len(power) <= 1:
        return np.nan

    power = power[
        1:
    ]

    total = float(
        np.sum(power)
    )

    if total <= 0:
        return 0.0

    probability = (
        power
        / total
    )

    entropy = -np.sum(
        probability
        * np.log(
            probability
            + 1e-15
        )
    )

    maximum = np.log(
        len(probability)
    )

    if maximum <= 0:
        return 0.0

    return float(
        entropy
        / maximum
    )


def normalized_auc(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    area = np.sum(
        0.5
        * (
            y[1:]
            + y[:-1]
        )
        * np.diff(x)
    )

    width = (
        x[-1]
        - x[0]
    )

    return float(
        area / width
    )


def bh_fdr(
    p_values,
):
    p = np.asarray(
        p_values,
        dtype=float,
    )

    order = np.argsort(
        p
    )

    ranked = p[
        order
    ]

    n = len(p)

    q = (
        ranked
        * n
        / np.arange(
            1,
            n + 1,
        )
    )

    q = np.minimum.accumulate(
        q[::-1]
    )[::-1]

    q = np.minimum(
        q,
        1.0,
    )

    result = np.empty(
        n,
        dtype=float,
    )

    result[
        order
    ] = q

    return result


# ---------------------------------------------------------------------
# Load development sensors
# ---------------------------------------------------------------------

dataset = load_dataset(
    "pems_bay",
    split="validation",
)

protocol = (
    dataset.protocol.copy()
)

first_eval = (
    protocol[
        "eval_origin"
    ].min()
)

print("=" * 100)
print(
    "PEMS-BAY entity identifiability analysis"
)
print("=" * 100)

print()
print(
    "Feature cutoff:",
    first_eval,
)

print(
    "Sensors:",
    len(
        dataset.series
    ),
)


# ---------------------------------------------------------------------
# Extract fixed pre-evaluation features
# ---------------------------------------------------------------------

feature_rows = []
weekly_profiles = {}


for sensor_id, series in (
    dataset.series.items()
):

    history = series[
        series.index
        < first_eval
    ].dropna()

    values = history.to_numpy(
        dtype=float
    )

    mean = float(
        np.mean(values)
    )

    std = float(
        np.std(values)
    )

    if std < 1e-8:
        std = 1.0

    cv = float(
        std
        / max(
            abs(mean),
            1e-8,
        )
    )

    daily_corr = lag_correlation(
        history,
        pd.Timedelta(
            days=1
        ),
    )

    weekly_corr = lag_correlation(
        history,
        pd.Timedelta(
            days=7
        ),
    )

    frame = pd.DataFrame(
        {
            "value": values,
        },
        index=history.index,
    )

    frame[
        "weekly_slot"
    ] = (
        frame.index.dayofweek
        * 288
        + frame.index.hour
        * 12
        + frame.index.minute
        // 5
    )

    profile = (
        frame
        .groupby(
            "weekly_slot"
        )[
            "value"
        ]
        .mean()
        .reindex(
            range(2016)
        )
    )

    profile = (
        profile.fillna(
            mean
        )
    )

    expected = (
        frame[
            "weekly_slot"
        ]
        .map(profile)
        .to_numpy(
            dtype=float
        )
    )

    residual = (
        values
        - expected
    )

    residual_std = float(
        np.std(
            residual
        )
    )

    residual_ratio = float(
        residual_std
        / std
    )

    profile_values = (
        profile.to_numpy(
            dtype=float
        )
    )

    profile_std = float(
        np.std(
            profile_values
        )
    )

    if profile_std < 1e-8:
        profile_std = 1.0

    profile_z = (
        profile_values
        - np.mean(
            profile_values
        )
    ) / profile_std

    weekly_profiles[
        sensor_id
    ] = profile_z

    start = history.index.min()

    first_end = (
        start
        + pd.Timedelta(
            days=14
        )
    )

    last_start = (
        first_eval
        - pd.Timedelta(
            days=14
        )
    )

    early = history[
        history.index
        < first_end
    ]

    late = history[
        history.index
        >= last_start
    ]

    if (
        len(early) > 0
        and len(late) > 0
    ):
        drift = abs(
            float(
                early.mean()
            )
            - float(
                late.mean()
            )
        ) / std
    else:
        drift = np.nan

    diff_std_ratio = float(
        np.std(
            np.diff(values)
        )
        / std
    )

    entropy = spectral_entropy(
        values
    )

    feature_rows.append(
        {
            "entity_id": (
                sensor_id
            ),
            "pre_eval_points": (
                len(history)
            ),
            "mean_speed": mean,
            "std_speed": std,
            "cv": cv,
            "daily_corr": (
                daily_corr
            ),
            "weekly_corr": (
                weekly_corr
            ),
            "weekly_residual_ratio": (
                residual_ratio
            ),
            "diff_std_ratio": (
                diff_std_ratio
            ),
            "spectral_entropy": (
                entropy
            ),
            "drift_14d_std": (
                drift
            ),
        }
    )


features = pd.DataFrame(
    feature_rows
).set_index(
    "entity_id"
)


# ---------------------------------------------------------------------
# Population distinctiveness
# ---------------------------------------------------------------------

entity_ids = list(
    features.index
)

profile_matrix = np.stack(
    [
        weekly_profiles[
            entity_id
        ]
        for entity_id
        in entity_ids
    ]
)

centroid = (
    profile_matrix.mean(
        axis=0
    )
)

centroid_distance = np.sqrt(
    np.mean(
        (
            profile_matrix
            - centroid[
                None,
                :
            ]
        )
        ** 2,
        axis=1,
    )
)

nearest_distance = np.empty(
    len(entity_ids),
    dtype=float,
)


for i in range(
    len(entity_ids)
):

    distance = np.sqrt(
        np.mean(
            (
                profile_matrix
                - profile_matrix[
                    i
                ][
                    None,
                    :
                ]
            )
            ** 2,
            axis=1,
        )
    )

    distance[i] = np.inf

    nearest_distance[i] = (
        np.min(distance)
    )


features[
    "profile_centroid_distance"
] = centroid_distance

features[
    "profile_nearest_distance"
] = nearest_distance


features[
    "log_identifiability_index"
] = (
    np.log(
        features[
            "profile_nearest_distance"
        ]
        + 1e-6
    )
    - np.log(
        features[
            "weekly_residual_ratio"
        ]
        + 1e-6
    )
)


features = (
    features
    .reset_index()
)


# ---------------------------------------------------------------------
# Build sensor-level outcomes
# ---------------------------------------------------------------------

curves = pd.read_csv(
    INPUT
)

outcome_rows = []


for (
    model,
    entity_id
), group in curves.groupby(
    [
        "model",
        "entity_id",
    ]
):

    group = (
        group
        .sort_values(
            "context_days"
        )
    )

    x = group[
        "context_days"
    ].to_numpy(
        dtype=float
    )

    identity = group[
        "identity_gap"
    ].to_numpy(
        dtype=float
    )

    generic = group[
        "generic_gain"
    ].to_numpy(
        dtype=float
    )

    # Add the fixed 1-day baseline,
    # where additional-history gains
    # are exactly zero.
    x_auc = np.concatenate(
        [
            [1.0],
            x,
        ]
    )

    identity_auc = np.concatenate(
        [
            [0.0],
            identity,
        ]
    )

    generic_auc = np.concatenate(
        [
            [0.0],
            generic,
        ]
    )

    identity_lookup = dict(
        zip(
            x,
            identity,
        )
    )

    outcome_rows.append(
        {
            "model": model,
            "entity_id": (
                entity_id
            ),
            "identity_auc": (
                normalized_auc(
                    x_auc,
                    identity_auc,
                )
            ),
            "generic_auc": (
                normalized_auc(
                    x_auc,
                    generic_auc,
                )
            ),
            "identity_5d": (
                identity_lookup[
                    5.0
                ]
            ),
            "identity_7d": (
                identity_lookup[
                    7.0
                ]
            ),
            "identity_10d": (
                identity_lookup[
                    10.0
                ]
            ),
            "identity_late_mean": float(
                group[
                    group[
                        "context_days"
                    ]
                    >= 10
                ][
                    "identity_gap"
                ].mean()
            ),
        }
    )


outcomes = pd.DataFrame(
    outcome_rows
)

features["entity_id"] = (
    features["entity_id"]
    .astype(str)
)

outcomes["entity_id"] = (
    outcomes["entity_id"]
    .astype(str)
)

analysis = outcomes.merge(
    features,
    on="entity_id",
    how="left",
    validate="many_to_one",
)


# ---------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------

predictors = [
    "daily_corr",
    "weekly_corr",
    "weekly_residual_ratio",
    "diff_std_ratio",
    "spectral_entropy",
    "drift_14d_std",
    "profile_centroid_distance",
    "profile_nearest_distance",
    "log_identifiability_index",
]

responses = [
    "identity_auc",
    "identity_10d",
    "identity_late_mean",
]


correlation_rows = []


for model in [
    "chronos2",
    "timesfm3",
]:

    subset = analysis[
        analysis[
            "model"
        ]
        == model
    ]

    for response in responses:

        model_rows = []

        for predictor in predictors:

            pair = subset[
                [
                    predictor,
                    response,
                ]
            ].dropna()

            rho, p_value = (
                spearmanr(
                    pair[
                        predictor
                    ],
                    pair[
                        response
                    ],
                )
            )

            model_rows.append(
                {
                    "model": model,
                    "response": (
                        response
                    ),
                    "predictor": (
                        predictor
                    ),
                    "n_entities": (
                        len(pair)
                    ),
                    "spearman_rho": (
                        float(rho)
                    ),
                    "p_value": (
                        float(p_value)
                    ),
                }
            )

        p_values = [
            row[
                "p_value"
            ]
            for row in model_rows
        ]

        q_values = bh_fdr(
            p_values
        )

        for row, q_value in zip(
            model_rows,
            q_values,
        ):
            row[
                "fdr_q"
            ] = float(
                q_value
            )

            correlation_rows.append(
                row
            )


correlations = pd.DataFrame(
    correlation_rows
)


# ---------------------------------------------------------------------
# Save and report
# ---------------------------------------------------------------------

feature_path = (
    OUT_DIR
    / "pre_evaluation_sensor_features.csv"
)

analysis_path = (
    OUT_DIR
    / "sensor_identifiability_analysis.csv"
)

correlation_path = (
    OUT_DIR
    / "identifiability_correlations.csv"
)


features.to_csv(
    feature_path,
    index=False,
)

analysis.to_csv(
    analysis_path,
    index=False,
)

correlations.to_csv(
    correlation_path,
    index=False,
)


print()
print("=" * 100)
print(
    "Strongest associations with identity AUC"
)
print("=" * 100)


for model in [
    "chronos2",
    "timesfm3",
]:

    print()
    print(model)
    print("-" * 100)

    table = (
        correlations[
            (
                correlations[
                    "model"
                ]
                == model
            )
            & (
                correlations[
                    "response"
                ]
                == "identity_auc"
            )
        ]
        .copy()
    )

    table[
        "abs_rho"
    ] = (
        table[
            "spearman_rho"
        ].abs()
    )

    table = (
        table
        .sort_values(
            "abs_rho",
            ascending=False,
        )
    )

    print(
        table[
            [
                "predictor",
                "spearman_rho",
                "p_value",
                "fdr_q",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )


print()
print("Saved:")
print(f"  {feature_path}")
print(f"  {analysis_path}")
print(f"  {correlation_path}")
