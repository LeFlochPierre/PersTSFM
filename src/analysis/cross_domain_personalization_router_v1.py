from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from src.data.registry import (
    load_dataset,
    extract_case,
)


ROOT = Path("results")

OUT = (
    ROOT
    / "context_regime_comparison"
    / "personalization_router_v1"
    / "cross_domain"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 42
N_SPLITS = 5
EPS = 1e-8


# ============================================================
# Frozen common regimes
# ============================================================

REGIMES = {
    "pems_bay": (24, 168),
    "electricity": (96, 192),
    "hupa_ucm": (24, 96),
    "vitaldb": (0.5, 4),
    "solar": (24, 96),
    "physionet2019": (6, 48),
}

DOMAIN_LABELS = {
    "pems_bay": "PEMS-BAY",
    "electricity": "Electricity",
    "hupa_ucm": "HUPA-UCM",
    "vitaldb": "VitalDB",
    "solar": "Solar",
    "physionet2019": "PhysioNet 2019",
}

MODELS = [
    "chronos2",
    "timesfm3",
    "tirex2",
]

MODEL_LABELS = {
    "chronos2": "Chronos-2",
    "timesfm3": "TimesFM3",
    "tirex2": "TiRex-2",
}


# ============================================================
# Router-v0 forecast-state features
# ============================================================

FORECAST_FEATURES = [
    "pred_mean",
    "pred_std",
    "pred_min",
    "pred_max",
    "pred_range",
    "pred_start",
    "pred_end",
    "pred_delta",
    "pred_abs_delta",
    "pred_slope",
    "width_mean",
    "width_std",
    "width_min",
    "width_max",
    "lower_spread_mean",
    "upper_spread_mean",
    "asymmetry_mean",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


# ============================================================
# NEW Router-v1 recent observed-state features
# ============================================================

RECENT_FEATURES = [
    "recent_mean",
    "recent_std",
    "recent_min",
    "recent_max",
    "recent_range",
    "recent_start",
    "recent_end",
    "recent_delta",
    "recent_abs_delta",
    "recent_slope",
    "recent_diff_mean",
    "recent_diff_std",
    "recent_diff_abs_mean",
    "recent_diff_abs_max",
    "recent_acf1",
    "recent_last_z",
    "recent_half_shift",
    "recent_half_shift_z",
    "recent_turning_rate",
]


# ============================================================
# Forecast/recent-state interaction features
# ============================================================

INTERACTION_FEATURES = [
    "forecast_start_gap",
    "forecast_mean_gap",
    "forecast_slope_gap",
    "uncertainty_to_recent_std",
    "forecast_range_to_recent_range",
]


FEATURE_COLS = (
    FORECAST_FEATURES
    + RECENT_FEATURES
    + INTERACTION_FEATURES
)


# Freeze same RF hyperparameters used in Router v0.
ROUTER_TEMPLATE = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=5,
    max_features=0.8,
    random_state=SEED,
    n_jobs=-1,
)


BUDGETS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.75,
    1.00,
]


def fmt(x):
    return f"{x:g}"


def intervention_root(
    dataset,
    model,
    recent,
    long,
):
    return (
        ROOT
        / dataset
        / "interventions"
        / model
        / (
            f"recent_{fmt(recent)}h"
            f"_long_{fmt(long)}h"
        )
    )


# ============================================================
# Helpers
# ============================================================

def safe_acf1(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) < 3:
        return 0.0

    x = values[:-1]
    y = values[1:]

    if (
        np.std(x) < EPS
        or np.std(y) < EPS
    ):
        return 0.0

    r = np.corrcoef(
        x,
        y,
    )[0, 1]

    if not np.isfinite(r):
        return 0.0

    return float(r)


def turning_rate(values):
    d = np.diff(
        np.asarray(
            values,
            dtype=float,
        )
    )

    if len(d) < 2:
        return 0.0

    s = np.sign(d)

    # Ignore exact zero differences.
    valid = (
        (s[:-1] != 0)
        & (s[1:] != 0)
    )

    if valid.sum() == 0:
        return 0.0

    return float(
        np.mean(
            s[:-1][valid]
            != s[1:][valid]
        )
    )


def slope(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) < 2:
        return 0.0

    x = np.arange(
        len(values),
        dtype=float,
    )

    return float(
        np.polyfit(
            x,
            values,
            deg=1,
        )[0]
    )


# ============================================================
# Recent observed-state features
# ============================================================

def build_recent_features(
    values,
):
    y = np.asarray(
        values,
        dtype=float,
    )

    if (
        len(y) == 0
        or not np.isfinite(y).all()
    ):
        raise RuntimeError(
            "Invalid recent context."
        )

    d = np.diff(y)

    mean = float(
        np.mean(y)
    )

    std = float(
        np.std(y)
    )

    minimum = float(
        np.min(y)
    )

    maximum = float(
        np.max(y)
    )

    value_range = (
        maximum - minimum
    )

    recent_slope = slope(y)

    if len(d) > 0:
        diff_mean = float(
            np.mean(d)
        )

        diff_std = float(
            np.std(d)
        )

        diff_abs_mean = float(
            np.mean(
                np.abs(d)
            )
        )

        diff_abs_max = float(
            np.max(
                np.abs(d)
            )
        )
    else:
        diff_mean = 0.0
        diff_std = 0.0
        diff_abs_mean = 0.0
        diff_abs_max = 0.0

    midpoint = (
        len(y) // 2
    )

    if midpoint > 0:
        first_half = y[:midpoint]
        second_half = y[midpoint:]

        half_shift = float(
            np.mean(second_half)
            - np.mean(first_half)
        )
    else:
        half_shift = 0.0

    last_z = (
        float(
            (y[-1] - mean)
            / (std + EPS)
        )
    )

    half_shift_z = (
        float(
            half_shift
            / (std + EPS)
        )
    )

    return {
        "recent_mean": mean,
        "recent_std": std,
        "recent_min": minimum,
        "recent_max": maximum,
        "recent_range": value_range,
        "recent_start": float(
            y[0]
        ),
        "recent_end": float(
            y[-1]
        ),
        "recent_delta": float(
            y[-1] - y[0]
        ),
        "recent_abs_delta": float(
            abs(y[-1] - y[0])
        ),
        "recent_slope": (
            recent_slope
        ),
        "recent_diff_mean": (
            diff_mean
        ),
        "recent_diff_std": (
            diff_std
        ),
        "recent_diff_abs_mean": (
            diff_abs_mean
        ),
        "recent_diff_abs_max": (
            diff_abs_max
        ),
        "recent_acf1": (
            safe_acf1(y)
        ),
        "recent_last_z": (
            last_z
        ),
        "recent_half_shift": (
            half_shift
        ),
        "recent_half_shift_z": (
            half_shift_z
        ),
        "recent_turning_rate": (
            turning_rate(y)
        ),
    }


# ============================================================
# Exact recent contexts from registry
# ============================================================

def build_recent_feature_table(
    dataset_key,
    recent_hours,
):

    dataset = load_dataset(
        dataset_key
    )

    spec = dataset.spec

    rows = []

    for _, protocol_row in (
        dataset.protocol.iterrows()
    ):

        case = extract_case(
            dataset,
            protocol_row,
            recent_hours,
        )

        features = (
            build_recent_features(
                case[
                    "context"
                ].to_numpy(
                    dtype=float
                )
            )
        )

        features[
            "entity_id"
        ] = str(
            case[
                "entity_id"
            ]
        )

        features[
            "eval_index"
        ] = int(
            case[
                "eval_index"
            ]
        )

        features[
            "eval_origin_recent"
        ] = case[
            "eval_origin"
        ]

        rows.append(
            features
        )

    result = pd.DataFrame(
        rows
    )

    if result.duplicated(
        [
            "entity_id",
            "eval_index",
        ]
    ).any():
        raise RuntimeError(
            f"{dataset_key}: "
            "duplicate recent-feature cases."
        )

    return result


# ============================================================
# Short-forecast features
# ============================================================

def build_forecast_features(
    group,
):

    group = group.sort_values(
        "timestamp"
    )

    p = group[
        "prediction"
    ].to_numpy(
        dtype=float
    )

    q10 = group[
        "q10"
    ].to_numpy(
        dtype=float
    )

    q50 = group[
        "q50"
    ].to_numpy(
        dtype=float
    )

    q90 = group[
        "q90"
    ].to_numpy(
        dtype=float
    )

    width = q90 - q10
    lower = q50 - q10
    upper = q90 - q50

    origin = pd.Timestamp(
        group[
            "eval_origin"
        ].iloc[0]
    )

    hour = (
        origin.hour
        + origin.minute / 60.0
    )

    dow = origin.dayofweek

    return {
        "pred_mean": float(
            np.mean(p)
        ),
        "pred_std": float(
            np.std(p)
        ),
        "pred_min": float(
            np.min(p)
        ),
        "pred_max": float(
            np.max(p)
        ),
        "pred_range": float(
            np.max(p)
            - np.min(p)
        ),
        "pred_start": float(
            p[0]
        ),
        "pred_end": float(
            p[-1]
        ),
        "pred_delta": float(
            p[-1] - p[0]
        ),
        "pred_abs_delta": float(
            abs(p[-1] - p[0])
        ),
        "pred_slope": (
            slope(p)
        ),

        "width_mean": float(
            np.mean(width)
        ),
        "width_std": float(
            np.std(width)
        ),
        "width_min": float(
            np.min(width)
        ),
        "width_max": float(
            np.max(width)
        ),

        "lower_spread_mean": float(
            np.mean(lower)
        ),
        "upper_spread_mean": float(
            np.mean(upper)
        ),
        "asymmetry_mean": float(
            np.mean(
                upper - lower
            )
        ),

        "hour_sin": float(
            np.sin(
                2 * np.pi
                * hour / 24.0
            )
        ),
        "hour_cos": float(
            np.cos(
                2 * np.pi
                * hour / 24.0
            )
        ),
        "dow_sin": float(
            np.sin(
                2 * np.pi
                * dow / 7.0
            )
        ),
        "dow_cos": float(
            np.cos(
                2 * np.pi
                * dow / 7.0
            )
        ),
    }


# ============================================================
# Setting loader
# ============================================================

def load_setting(
    dataset,
    model,
    recent,
    long,
    recent_features,
):

    base = intervention_root(
        dataset,
        model,
        recent,
        long,
    )

    metrics = pd.read_csv(
        base
        / "case_metrics.csv"
    )

    metrics[
        "entity_id"
    ] = (
        metrics[
            "entity_id"
        ].astype(str)
    )

    metrics = (
        metrics[
            metrics[
                "intervention"
            ]
            == "matched_other_affine"
        ]
        .copy()
        .reset_index(drop=True)
    )

    # ------------------------------------------
    # TWO TARGETS
    # ------------------------------------------

    # Is long target history better than short?
    metrics[
        "long_value"
    ] = (
        metrics[
            "short_mae"
        ]
        - metrics[
            "same_mae"
        ]
    )

    metrics[
        "long_helps"
    ] = (
        metrics[
            "long_value"
        ] > 0
    ).astype(int)

    # Does it specifically matter that the
    # long history belongs to the target?
    metrics[
        "personal_value"
    ] = (
        metrics["mae"]
        - metrics[
            "same_mae"
        ]
    )

    metrics[
        "personal_helps"
    ] = (
        metrics[
            "personal_value"
        ] > 0
    ).astype(int)

    pred = pd.read_parquet(
        base
        / "predictions.parquet"
    )

    pred[
        "entity_id"
    ] = (
        pred[
            "entity_id"
        ].astype(str)
    )

    pred = (
        pred[
            pred[
                "intervention"
            ]
            == "short"
        ]
        .copy()
    )

    pred["timestamp"] = (
        pd.to_datetime(
            pred[
                "timestamp"
            ]
        )
    )

    pred["eval_origin"] = (
        pd.to_datetime(
            pred[
                "eval_origin"
            ]
        )
    )

    forecast_rows = []

    for (
        entity_id,
        eval_index,
    ), group in pred.groupby(
        [
            "entity_id",
            "eval_index",
        ],
        sort=False,
    ):

        row = (
            build_forecast_features(
                group
            )
        )

        row[
            "entity_id"
        ] = str(
            entity_id
        )

        row[
            "eval_index"
        ] = int(
            eval_index
        )

        forecast_rows.append(
            row
        )

    forecast_features = (
        pd.DataFrame(
            forecast_rows
        )
    )

    data = metrics.merge(
        forecast_features,
        on=[
            "entity_id",
            "eval_index",
        ],
        how="inner",
        validate="one_to_one",
    )

    data = data.merge(
        recent_features,
        on=[
            "entity_id",
            "eval_index",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(data) != len(
        metrics
    ):
        raise RuntimeError(
            f"{dataset}/{model}: "
            f"{len(data)} / "
            f"{len(metrics)} matched."
        )

    # ------------------------------------------
    # Interaction features
    # ------------------------------------------

    data[
        "forecast_start_gap"
    ] = (
        data[
            "pred_start"
        ]
        - data[
            "recent_end"
        ]
    )

    data[
        "forecast_mean_gap"
    ] = (
        data[
            "pred_mean"
        ]
        - data[
            "recent_mean"
        ]
    )

    data[
        "forecast_slope_gap"
    ] = (
        data[
            "pred_slope"
        ]
        - data[
            "recent_slope"
        ]
    )

    data[
        "uncertainty_to_recent_std"
    ] = (
        data[
            "width_mean"
        ]
        / (
            data[
                "recent_std"
            ]
            + EPS
        )
    )

    data[
        "forecast_range_to_recent_range"
    ] = (
        data[
            "pred_range"
        ]
        / (
            data[
                "recent_range"
            ]
            + EPS
        )
    )

    missing = [
        c
        for c in FEATURE_COLS
        if c not in data.columns
    ]

    if missing:
        raise RuntimeError(
            f"{dataset}/{model}: "
            f"missing features "
            f"{missing}"
        )

    values = data[
        FEATURE_COLS
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise RuntimeError(
            f"{dataset}/{model}: "
            "non-finite feature."
        )

    return data


# ============================================================
# Out-of-entity router
# ============================================================

def fit_oof(
    data,
    target_col,
):

    X = data[
        FEATURE_COLS
    ].to_numpy(
        dtype=float
    )

    y = data[
        target_col
    ].to_numpy(
        dtype=float
    )

    groups = data[
        "entity_id"
    ].to_numpy()

    n_entities = (
        data[
            "entity_id"
        ].nunique()
    )

    cv = GroupKFold(
        n_splits=min(
            N_SPLITS,
            n_entities,
        )
    )

    oof = np.full(
        len(data),
        np.nan,
        dtype=float,
    )

    for (
        train_idx,
        test_idx,
    ) in cv.split(
        X,
        y,
        groups=groups,
    ):

        router = clone(
            ROUTER_TEMPLATE
        )

        router.fit(
            X[train_idx],
            y[train_idx],
        )

        oof[test_idx] = (
            router.predict(
                X[test_idx]
            )
        )

    if not np.isfinite(
        oof
    ).all():
        raise RuntimeError(
            "Non-finite OOF score."
        )

    return oof


# ============================================================
# Routing evaluation
# ============================================================

def evaluate_target(
    baseline,
    action,
    score,
):

    baseline = np.asarray(
        baseline,
        dtype=float,
    )

    action = np.asarray(
        action,
        dtype=float,
    )

    actual_value = (
        baseline - action
    )

    baseline_mean = float(
        np.mean(baseline)
    )

    action_mean = float(
        np.mean(action)
    )

    learned_order = np.argsort(
        -score
    )

    oracle_order = np.argsort(
        -actual_value
    )

    rows = []

    for budget in BUDGETS:

        n = len(
            baseline
        )

        k = int(
            round(
                budget * n
            )
        )

        learned = (
            baseline.copy()
        )

        oracle = (
            baseline.copy()
        )

        if k > 0:
            learned_idx = (
                learned_order[:k]
            )

            oracle_idx = (
                oracle_order[:k]
            )

            learned[
                learned_idx
            ] = action[
                learned_idx
            ]

            oracle[
                oracle_idx
            ] = action[
                oracle_idx
            ]

        learned_mae = float(
            np.mean(
                learned
            )
        )

        oracle_mae = float(
            np.mean(
                oracle
            )
        )

        learned_gain = (
            baseline_mean
            - learned_mae
        )

        oracle_gain = (
            baseline_mean
            - oracle_mae
        )

        if oracle_gain > EPS:
            capture = (
                learned_gain
                / oracle_gain
            )
        else:
            capture = np.nan

        if k > 0:
            precision = float(
                np.mean(
                    actual_value[
                        learned_order[:k]
                    ] > 0
                )
            )
        else:
            precision = np.nan

        rows.append(
            {
                "budget_fraction": (
                    budget
                ),
                "n_selected": k,
                "learned_mae": (
                    learned_mae
                ),
                "oracle_mae": (
                    oracle_mae
                ),
                "learned_gain_pct": (
                    100.0
                    * learned_gain
                    / baseline_mean
                ),
                "oracle_gain_pct": (
                    100.0
                    * oracle_gain
                    / baseline_mean
                ),
                "oracle_capture": (
                    capture
                ),
                "precision_positive_value": (
                    precision
                ),
            }
        )

    always_action_gain_pct = (
        100.0
        * (
            baseline_mean
            - action_mean
        )
        / baseline_mean
    )

    return (
        pd.DataFrame(
            rows
        ),
        always_action_gain_pct,
    )


# ============================================================
# Build recent context features once per domain
# ============================================================

recent_feature_cache = {}

for dataset, (
    recent,
    _,
) in REGIMES.items():

    print(
        f"Building recent-state features: "
        f"{DOMAIN_LABELS[dataset]}"
    )

    recent_feature_cache[
        dataset
    ] = (
        build_recent_feature_table(
            dataset,
            recent,
        )
    )


# ============================================================
# Main analysis
# ============================================================

summary_rows = []
curve_rows = []
case_rows = []


for dataset, (
    recent,
    long,
) in REGIMES.items():

    for model in MODELS:

        print()
        print(
            "=" * 110
        )

        print(
            f"{DOMAIN_LABELS[dataset]} "
            f"| "
            f"{MODEL_LABELS[model]}"
        )

        print(
            "=" * 110
        )

        data = load_setting(
            dataset,
            model,
            recent,
            long,
            recent_feature_cache[
                dataset
            ],
        )

        print(
            f"Cases: {len(data)} | "
            f"Entities: "
            f"{data['entity_id'].nunique()}"
        )

        # ======================================
        # TARGET A:
        # NEED MORE HISTORY?
        # ======================================

        long_score = fit_oof(
            data,
            "long_value",
        )

        long_rho, long_p = (
            spearmanr(
                data[
                    "long_value"
                ],
                long_score,
            )
        )

        long_auc = roc_auc_score(
            data[
                "long_helps"
            ],
            long_score,
        )

        (
            long_curve,
            always_long_gain,
        ) = evaluate_target(
            baseline=data[
                "short_mae"
            ],
            action=data[
                "same_mae"
            ],
            score=long_score,
        )

        long25 = long_curve[
            np.isclose(
                long_curve[
                    "budget_fraction"
                ],
                0.25,
            )
        ].iloc[0]

        # ======================================
        # TARGET B:
        # NEED TARGET-SPECIFIC HISTORY?
        # ======================================

        personal_score = fit_oof(
            data,
            "personal_value",
        )

        (
            personal_rho,
            personal_p,
        ) = spearmanr(
            data[
                "personal_value"
            ],
            personal_score,
        )

        personal_auc = (
            roc_auc_score(
                data[
                    "personal_helps"
                ],
                personal_score,
            )
        )

        (
            personal_curve,
            always_personal_gain,
        ) = evaluate_target(
            baseline=data[
                "mae"
            ],
            action=data[
                "same_mae"
            ],
            score=personal_score,
        )

        personal25 = (
            personal_curve[
                np.isclose(
                    personal_curve[
                        "budget_fraction"
                    ],
                    0.25,
                )
            ].iloc[0]
        )

        print(
            "LONG CONTEXT"
        )

        print(
            f"  rho={long_rho:.3f} "
            f"(p={long_p:.3g}) | "
            f"AUROC={long_auc:.3f}"
        )

        print(
            f"  @25% gain="
            f"{long25['learned_gain_pct']:.2f}% | "
            f"oracle="
            f"{long25['oracle_gain_pct']:.2f}% | "
            f"capture="
            f"{100*long25['oracle_capture']:.1f}%"
        )

        print(
            "TARGET-SPECIFIC"
        )

        print(
            f"  rho={personal_rho:.3f} "
            f"(p={personal_p:.3g}) | "
            f"AUROC={personal_auc:.3f}"
        )

        print(
            f"  @25% gain="
            f"{personal25['learned_gain_pct']:.2f}% | "
            f"oracle="
            f"{personal25['oracle_gain_pct']:.2f}% | "
            f"capture="
            f"{100*personal25['oracle_capture']:.1f}%"
        )

        summary_rows.append(
            {
                "dataset": dataset,
                "domain": (
                    DOMAIN_LABELS[
                        dataset
                    ]
                ),
                "model": model,
                "model_label": (
                    MODEL_LABELS[
                        model
                    ]
                ),
                "n_cases": len(data),
                "n_entities": (
                    data[
                        "entity_id"
                    ].nunique()
                ),

                "long_rho": (
                    long_rho
                ),
                "long_p": (
                    long_p
                ),
                "long_auc": (
                    long_auc
                ),
                "long_router25_gain_pct": (
                    long25[
                        "learned_gain_pct"
                    ]
                ),
                "long_oracle25_gain_pct": (
                    long25[
                        "oracle_gain_pct"
                    ]
                ),
                "long_router25_capture": (
                    long25[
                        "oracle_capture"
                    ]
                ),
                "long_router25_precision": (
                    long25[
                        "precision_positive_value"
                    ]
                ),
                "always_long_gain_pct": (
                    always_long_gain
                ),

                "personal_rho": (
                    personal_rho
                ),
                "personal_p": (
                    personal_p
                ),
                "personal_auc": (
                    personal_auc
                ),
                "personal_router25_gain_pct": (
                    personal25[
                        "learned_gain_pct"
                    ]
                ),
                "personal_oracle25_gain_pct": (
                    personal25[
                        "oracle_gain_pct"
                    ]
                ),
                "personal_router25_capture": (
                    personal25[
                        "oracle_capture"
                    ]
                ),
                "personal_router25_precision": (
                    personal25[
                        "precision_positive_value"
                    ]
                ),
                "always_personal_gain_pct": (
                    always_personal_gain
                ),
            }
        )

        for target, curve in [
            (
                "long",
                long_curve,
            ),
            (
                "personal",
                personal_curve,
            ),
        ]:
            c = curve.copy()

            c[
                "target"
            ] = target

            c[
                "dataset"
            ] = dataset

            c[
                "domain"
            ] = (
                DOMAIN_LABELS[
                    dataset
                ]
            )

            c[
                "model"
            ] = model

            c[
                "model_label"
            ] = (
                MODEL_LABELS[
                    model
                ]
            )

            curve_rows.append(
                c
            )

        case = data[
            [
                "entity_id",
                "eval_index",
                "long_value",
                "personal_value",
                "long_helps",
                "personal_helps",
            ] + FEATURE_COLS
        ].copy()

        case[
            "predicted_long_value"
        ] = long_score

        case[
            "predicted_personal_value"
        ] = personal_score

        case[
            "dataset"
        ] = dataset

        case[
            "model"
        ] = model

        case_rows.append(
            case
        )


summary = pd.DataFrame(
    summary_rows
)

curves = pd.concat(
    curve_rows,
    ignore_index=True,
)

cases = pd.concat(
    case_rows,
    ignore_index=True,
)


summary.to_csv(
    OUT
    / "router_v1_summary.csv",
    index=False,
)

curves.to_csv(
    OUT
    / "router_v1_budget_curves.csv",
    index=False,
)

cases.to_csv(
    OUT
    / "router_v1_oof_scores.csv",
    index=False,
)


# ============================================================
# Terminal synthesis
# ============================================================

print()
print()
print("=" * 145)
print("ROUTER V1 CROSS-DOMAIN SUMMARY")
print("=" * 145)

display = summary[
    [
        "domain",
        "model_label",
        "long_rho",
        "long_auc",
        "long_router25_capture",
        "personal_rho",
        "personal_auc",
        "personal_router25_capture",
    ]
].copy()

for col in [
    "long_rho",
    "long_auc",
    "personal_rho",
    "personal_auc",
]:
    display[col] = (
        display[col]
        .map(
            lambda x: f"{x:.3f}"
        )
    )

for col in [
    "long_router25_capture",
    "personal_router25_capture",
]:
    display[col] = (
        display[col]
        .map(
            lambda x: f"{100*x:.1f}%"
        )
    )

print(
    display.to_string(
        index=False
    )
)


print()
print("=" * 145)
print("GLOBAL: LONG-CONTEXT ROUTING")
print("=" * 145)

print(
    "Positive rho: "
    f"{int((summary['long_rho'] > 0).sum())}"
    f"/18"
)

print(
    "Significant rho: "
    f"{int((summary['long_p'] < 0.05).sum())}"
    f"/18"
)

print(
    "AUROC > .5: "
    f"{int((summary['long_auc'] > 0.5).sum())}"
    f"/18"
)

print(
    f"Mean rho: "
    f"{summary['long_rho'].mean():.3f}"
)

print(
    f"Median rho: "
    f"{summary['long_rho'].median():.3f}"
)

print(
    f"Mean AUROC: "
    f"{summary['long_auc'].mean():.3f}"
)

print(
    f"Median AUROC: "
    f"{summary['long_auc'].median():.3f}"
)

print(
    "Mean oracle capture @25%: "
    f"{100*summary['long_router25_capture'].mean():.1f}%"
)

print(
    "Median oracle capture @25%: "
    f"{100*summary['long_router25_capture'].median():.1f}%"
)


print()
print("=" * 145)
print("GLOBAL: TARGET-SPECIFIC ROUTING")
print("=" * 145)

print(
    "Positive rho: "
    f"{int((summary['personal_rho'] > 0).sum())}"
    f"/18"
)

print(
    "Significant rho: "
    f"{int((summary['personal_p'] < 0.05).sum())}"
    f"/18"
)

print(
    "AUROC > .5: "
    f"{int((summary['personal_auc'] > 0.5).sum())}"
    f"/18"
)

print(
    f"Mean rho: "
    f"{summary['personal_rho'].mean():.3f}"
)

print(
    f"Median rho: "
    f"{summary['personal_rho'].median():.3f}"
)

print(
    f"Mean AUROC: "
    f"{summary['personal_auc'].mean():.3f}"
)

print(
    f"Median AUROC: "
    f"{summary['personal_auc'].median():.3f}"
)

print(
    "Mean oracle capture @25%: "
    f"{100*summary['personal_router25_capture'].mean():.1f}%"
)

print(
    "Median oracle capture @25%: "
    f"{100*summary['personal_router25_capture'].median():.1f}%"
)


# ============================================================
# Compare against Router v0 if available
# ============================================================

v0_long_path = (
    ROOT
    / "context_regime_comparison"
    / "personalization_router_v0"
    / "cross_domain"
    / "cross_domain_summary.csv"
)

v0_personal_path = (
    ROOT
    / "context_regime_comparison"
    / "personalization_specific_router_v0"
    / "cross_domain"
    / "personalization_specific_summary.csv"
)


if (
    v0_long_path.exists()
    and v0_personal_path.exists()
):

    v0_long = pd.read_csv(
        v0_long_path
    )

    v0_personal = pd.read_csv(
        v0_personal_path
    )

    comparison = summary[
        [
            "dataset",
            "model",
            "domain",
            "model_label",
            "long_rho",
            "long_auc",
            "long_router25_capture",
            "personal_rho",
            "personal_auc",
            "personal_router25_capture",
        ]
    ].copy()

    comparison = (
        comparison.merge(
            v0_long[
                [
                    "dataset",
                    "model",
                    "spearman_rho",
                    "auroc_long_helps",
                    "router25_capture",
                ]
            ],
            on=[
                "dataset",
                "model",
            ],
            how="left",
        )
    )

    comparison = (
        comparison.rename(
            columns={
                "spearman_rho":
                    "v0_long_rho",
                "auroc_long_helps":
                    "v0_long_auc",
                "router25_capture":
                    "v0_long_capture",
            }
        )
    )

    comparison = (
        comparison.merge(
            v0_personal[
                [
                    "dataset",
                    "model",
                    "spearman_rho",
                    "auroc_personal_helps",
                    "router25_capture",
                ]
            ],
            on=[
                "dataset",
                "model",
            ],
            how="left",
        )
    )

    comparison = (
        comparison.rename(
            columns={
                "spearman_rho":
                    "v0_personal_rho",
                "auroc_personal_helps":
                    "v0_personal_auc",
                "router25_capture":
                    "v0_personal_capture",
            }
        )
    )

    comparison[
        "delta_long_rho"
    ] = (
        comparison[
            "long_rho"
        ]
        - comparison[
            "v0_long_rho"
        ]
    )

    comparison[
        "delta_personal_rho"
    ] = (
        comparison[
            "personal_rho"
        ]
        - comparison[
            "v0_personal_rho"
        ]
    )

    comparison[
        "delta_long_capture"
    ] = (
        comparison[
            "long_router25_capture"
        ]
        - comparison[
            "v0_long_capture"
        ]
    )

    comparison[
        "delta_personal_capture"
    ] = (
        comparison[
            "personal_router25_capture"
        ]
        - comparison[
            "v0_personal_capture"
        ]
    )

    comparison.to_csv(
        OUT
        / "router_v1_vs_v0.csv",
        index=False,
    )

    print()
    print("=" * 145)
    print("V1 - V0 CHANGE")
    print("=" * 145)

    print(
        "Mean delta LONG rho:       "
        f"{comparison['delta_long_rho'].mean():+.3f}"
    )

    print(
        "Mean delta PERSONAL rho:   "
        f"{comparison['delta_personal_rho'].mean():+.3f}"
    )

    print(
        "Mean delta LONG capture:   "
        f"{100*comparison['delta_long_capture'].mean():+.1f} pp"
    )

    print(
        "Mean delta PERSONAL capture: "
        f"{100*comparison['delta_personal_capture'].mean():+.1f} pp"
    )

    print(
        "Personal rho improved:     "
        f"{int((comparison['delta_personal_rho'] > 0).sum())}/18"
    )

    print(
        "Personal capture improved: "
        f"{int((comparison['delta_personal_capture'] > 0).sum())}/18"
    )


print()
print(
    "Number of Router-v1 features:",
    len(FEATURE_COLS),
)

print(
    "Saved to:",
    OUT,
)
