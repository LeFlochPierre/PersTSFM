from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


CURVES_PATH = Path(
    "results/pems_bay/information_acquisition/"
    "sensor_mechanism_curves.csv"
)

FEATURES_PATH = Path(
    "results/pems_bay/entity_identifiability/"
    "pre_evaluation_sensor_features.csv"
)

OUT_DIR = Path(
    "results/pems_bay/entity_identifiability"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIT_OUTPUT = (
    OUT_DIR
    / "sensor_identity_strength_speed.csv"
)

STATS_OUTPUT = (
    OUT_DIR
    / "strength_speed_correlations.csv"
)


MODELS = [
    "chronos2",
    "timesfm3",
]

TAU_GRID = np.logspace(
    np.log10(0.1),
    np.log10(100.0),
    3000,
)

N_BOOTSTRAP = 20000
N_PERMUTATION = 100000
SEED = 42


def fit_identity_curve(
    context_days,
    identity_gap,
):
    x = (
        np.asarray(
            context_days,
            dtype=float,
        )
        - 1.0
    )

    y = np.asarray(
        identity_gap,
        dtype=float,
    )

    F = (
        1.0
        - np.exp(
            -x[None, :]
            / TAU_GRID[:, None]
        )
    )

    denominator = np.sum(
        F ** 2,
        axis=1,
    )

    numerator = (
        F @ y
    )

    amplitude = (
        numerator
        / denominator
    )

    amplitude = np.maximum(
        amplitude,
        0.0,
    )

    fitted = (
        amplitude[:, None]
        * F
    )

    sse = np.sum(
        (
            fitted
            - y[None, :]
        )
        ** 2,
        axis=1,
    )

    best = int(
        np.argmin(sse)
    )

    tau = float(
        TAU_GRID[best]
    )

    asymptote = float(
        amplitude[best]
    )

    prediction = (
        asymptote
        * F[best]
    )

    total_ss = float(
        np.sum(
            (
                y
                - np.mean(y)
            )
            ** 2
        )
    )

    if total_ss > 0:
        r2 = (
            1.0
            - float(
                sse[best]
            )
            / total_ss
        )
    else:
        r2 = np.nan

    boundary = bool(
        best == 0
        or best
        == len(
            TAU_GRID
        )
        - 1
    )

    return {
        "identity_tau": tau,
        "identity_asymptote": (
            asymptote
        ),
        "identity_fit_r2": r2,
        "tau_boundary": (
            boundary
        ),
    }


def bootstrap_spearman(
    x,
    y,
    rng,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    n = len(x)

    observed = float(
        spearmanr(
            x,
            y,
        ).statistic
    )

    boot = []

    for _ in range(
        N_BOOTSTRAP
    ):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        rho = spearmanr(
            x[idx],
            y[idx],
        ).statistic

        if np.isfinite(
            rho
        ):
            boot.append(
                float(rho)
            )

    boot = np.asarray(
        boot,
        dtype=float,
    )

    ci_low, ci_high = (
        np.quantile(
            boot,
            [
                0.025,
                0.975,
            ],
        )
    )

    return (
        observed,
        float(ci_low),
        float(ci_high),
    )


def permutation_spearman(
    x,
    y,
    rng,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    observed = float(
        spearmanr(
            x,
            y,
        ).statistic
    )

    extreme = 0

    for _ in range(
        N_PERMUTATION
    ):
        permuted = rng.permutation(
            y
        )

        rho = spearmanr(
            x,
            permuted,
        ).statistic

        if abs(rho) >= abs(
            observed
        ):
            extreme += 1

    return (
        extreme + 1
    ) / (
        N_PERMUTATION + 1
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


curves = pd.read_csv(
    CURVES_PATH,
    dtype={
        "entity_id": str,
    },
)

features = pd.read_csv(
    FEATURES_PATH,
    dtype={
        "entity_id": str,
    },
)


fit_rows = []


for (
    model,
    entity_id
), group in curves.groupby(
    [
        "model",
        "entity_id",
    ]
):
    group = group.sort_values(
        "context_days"
    )

    fit = fit_identity_curve(
        group[
            "context_days"
        ].to_numpy(
            dtype=float
        ),
        group[
            "identity_gap"
        ].to_numpy(
            dtype=float
        ),
    )

    late = group[
        group[
            "context_days"
        ]
        >= 10
    ]

    fit_rows.append(
        {
            "model": model,
            "entity_id": (
                entity_id
            ),
            "identity_auc": float(
                np.trapezoid(
                    group[
                        "identity_gap"
                    ],
                    group[
                        "context_days"
                    ],
                )
                / (
                    group[
                        "context_days"
                    ].max()
                    - group[
                        "context_days"
                    ].min()
                )
            ),
            "identity_late_mean": float(
                late[
                    "identity_gap"
                ].mean()
            ),
            **fit,
        }
    )


fits = pd.DataFrame(
    fit_rows
)

analysis = fits.merge(
    features,
    on="entity_id",
    how="left",
    validate="many_to_one",
)


analysis.to_csv(
    FIT_OUTPUT,
    index=False,
)


print("=" * 100)
print(
    "PEMS-BAY entity identity strength "
    "and acquisition speed"
)
print("=" * 100)


for model in MODELS:

    subset = analysis[
        analysis[
            "model"
        ]
        == model
    ].copy()

    print()
    print("-" * 100)
    print(
        f"Model: {model}"
    )
    print("-" * 100)

    print(
        "Sensors:",
        len(subset),
    )

    print(
        "Tau boundary fits:",
        int(
            subset[
                "tau_boundary"
            ].sum()
        ),
        "/",
        len(subset),
    )

    print(
        "Median identity-fit R2:",
        round(
            float(
                subset[
                    "identity_fit_r2"
                ].median()
            ),
            3,
        ),
    )

    print(
        "Median tau:",
        round(
            float(
                subset[
                    "identity_tau"
                ].median()
            ),
            3,
        ),
        "days",
    )

    print(
        "Median asymptote:",
        round(
            float(
                subset[
                    "identity_asymptote"
                ].median()
            ),
            4,
        ),
    )


rng = np.random.default_rng(
    SEED
)

response_specs = [
    (
        "identity_auc",
        "Identity AUC",
        "positive",
    ),
    (
        "identity_late_mean",
        "Late identity strength",
        "positive",
    ),
    (
        "identity_asymptote",
        "Fitted identity asymptote",
        "positive",
    ),
    (
        "identity_tau",
        "Identity acquisition tau",
        "negative",
    ),
]


stat_rows = []


print()
print("=" * 100)
print(
    "Identifiability-index associations"
)
print("=" * 100)


for model in MODELS:

    subset = analysis[
        analysis[
            "model"
        ]
        == model
    ].copy()

    model_rows = []

    print()
    print(model)
    print("-" * 100)

    for (
        response,
        label,
        expected_direction,
    ) in response_specs:

        pair = subset[
            [
                "log_identifiability_index",
                response,
            ]
        ].dropna()

        x = pair[
            "log_identifiability_index"
        ].to_numpy(
            dtype=float
        )

        y = pair[
            response
        ].to_numpy(
            dtype=float
        )

        (
            rho,
            ci_low,
            ci_high,
        ) = bootstrap_spearman(
            x,
            y,
            rng,
        )

        p_value = permutation_spearman(
            x,
            y,
            rng,
        )

        model_rows.append(
            {
                "model": model,
                "response": (
                    response
                ),
                "label": label,
                "expected_direction": (
                    expected_direction
                ),
                "n_entities": (
                    len(pair)
                ),
                "spearman_rho": (
                    rho
                ),
                "bootstrap_ci_low": (
                    ci_low
                ),
                "bootstrap_ci_high": (
                    ci_high
                ),
                "permutation_p": (
                    p_value
                ),
            }
        )

    q_values = bh_fdr(
        [
            row[
                "permutation_p"
            ]
            for row in model_rows
        ]
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

        stat_rows.append(
            row
        )

        print()
        print(
            row[
                "label"
            ]
        )

        print(
            "  rho:       "
            f"{row['spearman_rho']:.3f}"
        )

        print(
            "  95% CI:    "
            f"[{row['bootstrap_ci_low']:.3f}, "
            f"{row['bootstrap_ci_high']:.3f}]"
        )

        print(
            "  perm p:    "
            f"{row['permutation_p']:.6g}"
        )

        print(
            "  FDR q:     "
            f"{row['fdr_q']:.6g}"
        )


stats = pd.DataFrame(
    stat_rows
)

stats.to_csv(
    STATS_OUTPUT,
    index=False,
)


# ---------------------------------------------------------------------
# Cross-model reproducibility
# ---------------------------------------------------------------------

print()
print("=" * 100)
print(
    "Cross-model sensor reproducibility"
)
print("=" * 100)


for response in [
    "identity_auc",
    "identity_late_mean",
    "identity_asymptote",
    "identity_tau",
]:

    wide = analysis.pivot(
        index="entity_id",
        columns="model",
        values=response,
    ).dropna()

    rho, p_value = spearmanr(
        wide[
            "chronos2"
        ],
        wide[
            "timesfm3"
        ],
    )

    print()
    print(response)

    print(
        "  Spearman rho: "
        f"{rho:.3f}"
    )

    print(
        "  p-value:      "
        f"{p_value:.6g}"
    )


print()
print("Saved:")
print(f"  {FIT_OUTPUT}")
print(f"  {STATS_OUTPUT}")
