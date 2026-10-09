from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT = Path(
    "results/pems_bay/interventions"
)

OUT_DIR = Path(
    "results/pems_bay/information_acquisition"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS = [
    "chronos2",
    "timesfm3",
]

CONTEXT_HOURS = [
    48,
    72,
    120,
    168,
    240,
    336,
    504,
    672,
]

N_BOOTSTRAP = 10000
SEED = 42

# Characteristic exponential timescale search.
TAU_GRID = np.logspace(
    np.log10(0.1),
    np.log10(100.0),
    2000,
)

METRICS = [
    "total_gain",
    "generic_gain",
    "identity_gap",
]


# ---------------------------------------------------------------------
# Saturating model
#
# G(x) = A * (1 - exp(-x / tau))
#
# x = additional context beyond the fixed 1-day recent history.
# ---------------------------------------------------------------------

def design_matrix(
    x,
):
    return (
        1.0
        - np.exp(
            -x[None, :]
            / TAU_GRID[:, None]
        )
    )


def fit_curve(
    y,
    F,
):
    y = np.asarray(
        y,
        dtype=float,
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

    prediction = (
        amplitude[:, None]
        * F
    )

    sse = np.sum(
        (
            prediction
            - y[None, :]
        )
        ** 2,
        axis=1,
    )

    best = int(
        np.argmin(sse)
    )

    tau = float(
        TAU_GRID[
            best
        ]
    )

    asymptote = float(
        amplitude[
            best
        ]
    )

    fitted = (
        asymptote
        * F[
            best
        ]
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
                sse[
                    best
                ]
            )
            / total_ss
        )
    else:
        r2 = np.nan

    return {
        "tau": tau,
        "asymptote": asymptote,
        "r2": r2,
        "fitted": fitted,
    }


def fit_bootstrap_curves(
    Y,
    F,
    chunk_size=500,
):
    Y = np.asarray(
        Y,
        dtype=float,
    )

    denominator = np.sum(
        F ** 2,
        axis=1,
    )

    n = len(Y)

    best_tau = np.empty(
        n,
        dtype=float,
    )

    best_asymptote = np.empty(
        n,
        dtype=float,
    )

    for start in range(
        0,
        n,
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            n,
        )

        chunk = (
            Y[
                start:stop
            ]
        )

        numerator = (
            chunk @ F.T
        )

        amplitude = (
            numerator
            / denominator[
                None,
                :
            ]
        )

        amplitude = np.maximum(
            amplitude,
            0.0,
        )

        y2 = np.sum(
            chunk ** 2,
            axis=1,
        )[
            :,
            None,
        ]

        sse = (
            y2
            - 2.0
            * amplitude
            * numerator
            + (
                amplitude ** 2
            )
            * denominator[
                None,
                :
            ]
        )

        best = np.argmin(
            sse,
            axis=1,
        )

        row_index = np.arange(
            stop - start
        )

        best_tau[
            start:stop
        ] = (
            TAU_GRID[
                best
            ]
        )

        best_asymptote[
            start:stop
        ] = (
            amplitude[
                row_index,
                best,
            ]
        )

    return (
        best_tau,
        best_asymptote,
    )


# ---------------------------------------------------------------------
# Load entity-level intervention curves
# ---------------------------------------------------------------------

sensor_rows = []


for model in MODELS:

    for long_hours in CONTEXT_HOURS:

        path = (
            ROOT
            / model
            / (
                "recent_24h"
                f"_long_{long_hours}h"
            )
            / "entity_metrics.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        df = pd.read_csv(
            path
        )

        wide = df.pivot(
            index="entity_id",
            columns="intervention",
            values="mean_mae",
        )

        required = [
            "short",
            "same_entity",
            "matched_other_affine",
            "same_entity_day_shuffle",
        ]

        missing = [
            name
            for name in required
            if name not in wide.columns
        ]

        if missing:
            raise RuntimeError(
                f"{path}: missing "
                f"{missing}"
            )

        for entity_id, row in (
            wide.iterrows()
        ):
            short = float(
                row[
                    "short"
                ]
            )

            same = float(
                row[
                    "same_entity"
                ]
            )

            other = float(
                row[
                    "matched_other_affine"
                ]
            )

            shuffled = float(
                row[
                    "same_entity_day_shuffle"
                ]
            )

            sensor_rows.append(
                {
                    "model": model,
                    "entity_id": (
                        entity_id
                    ),
                    "context_hours": (
                        long_hours
                    ),
                    "context_days": (
                        long_hours
                        / 24.0
                    ),
                    "additional_days": (
                        long_hours
                        / 24.0
                        - 1.0
                    ),
                    "short_mae": short,
                    "same_entity_mae": (
                        same
                    ),
                    "matched_other_mae": (
                        other
                    ),
                    "shuffled_same_mae": (
                        shuffled
                    ),
                    "total_gain": (
                        short
                        - same
                    ),
                    "generic_gain": (
                        short
                        - other
                    ),
                    "identity_gap": (
                        other
                        - same
                    ),
                    "order_gap": (
                        shuffled
                        - same
                    ),
                }
            )


sensor_df = pd.DataFrame(
    sensor_rows
)

sensor_path = (
    OUT_DIR
    / "sensor_mechanism_curves.csv"
)

sensor_df.to_csv(
    sensor_path,
    index=False,
)


# ---------------------------------------------------------------------
# Fit acquisition scales with entity bootstrap
# ---------------------------------------------------------------------

rng = np.random.default_rng(
    SEED
)

summary_rows = []
fit_rows = []
bootstrap_rows = []


print("=" * 100)
print(
    "PEMS-BAY information acquisition scales"
)
print("=" * 100)


for model in MODELS:

    model_df = sensor_df[
        sensor_df[
            "model"
        ]
        == model
    ].copy()

    entity_ids = sorted(
        model_df[
            "entity_id"
        ].unique()
    )

    context_days = np.array(
        sorted(
            model_df[
                "context_days"
            ].unique()
        ),
        dtype=float,
    )

    additional_days = (
        context_days
        - 1.0
    )

    F = design_matrix(
        additional_days
    )

    n_entities = len(
        entity_ids
    )

    # One paired bootstrap sample is reused
    # across all mechanisms.
    bootstrap_indices = (
        rng.integers(
            0,
            n_entities,
            size=(
                N_BOOTSTRAP,
                n_entities,
            ),
        )
    )

    model_bootstrap = {}

    print()
    print("-" * 100)
    print(
        f"Model: {model}"
    )
    print("-" * 100)

    for metric in METRICS:

        matrix_df = (
            model_df
            .pivot(
                index="entity_id",
                columns="context_days",
                values=metric,
            )
            .reindex(
                index=entity_ids,
                columns=context_days,
            )
        )

        if matrix_df.isna().any().any():
            raise RuntimeError(
                f"{model} / {metric}: "
                "missing sensor/context "
                "values."
            )

        matrix = (
            matrix_df.to_numpy(
                dtype=float
            )
        )

        cohort_curve = (
            matrix.mean(
                axis=0
            )
        )

        observed = fit_curve(
            cohort_curve,
            F,
        )

        bootstrap_curves = (
            matrix[
                bootstrap_indices,
                :
            ]
            .mean(
                axis=1
            )
        )

        (
            boot_tau,
            boot_asymptote,
        ) = fit_bootstrap_curves(
            bootstrap_curves,
            F,
        )

        model_bootstrap[
            metric
        ] = {
            "tau": boot_tau,
            "asymptote": (
                boot_asymptote
            ),
        }

        tau_low, tau_high = (
            np.quantile(
                boot_tau,
                [
                    0.025,
                    0.975,
                ],
            )
        )

        amp_low, amp_high = (
            np.quantile(
                boot_asymptote,
                [
                    0.025,
                    0.975,
                ],
            )
        )

        tau = observed[
            "tau"
        ]

        asymptote = observed[
            "asymptote"
        ]

        # Context measured from the original
        # forecast origin, including the fixed
        # 1-day recent block.
        l50 = (
            1.0
            + tau
            * np.log(2.0)
        )

        l90 = (
            1.0
            + tau
            * np.log(10.0)
        )

        summary_rows.append(
            {
                "model": model,
                "mechanism": metric,
                "n_entities": (
                    n_entities
                ),
                "tau_additional_days": (
                    tau
                ),
                "tau_ci_low": float(
                    tau_low
                ),
                "tau_ci_high": float(
                    tau_high
                ),
                "asymptotic_gain": (
                    asymptote
                ),
                "asymptotic_gain_ci_low": (
                    float(
                        amp_low
                    )
                ),
                "asymptotic_gain_ci_high": (
                    float(
                        amp_high
                    )
                ),
                "l50_context_days": (
                    l50
                ),
                "l90_context_days": (
                    l90
                ),
                "r2": observed[
                    "r2"
                ],
            }
        )

        for (
            context_day,
            observed_gain,
            fitted_gain,
        ) in zip(
            context_days,
            cohort_curve,
            observed[
                "fitted"
            ],
        ):
            fit_rows.append(
                {
                    "model": model,
                    "mechanism": metric,
                    "context_days": (
                        context_day
                    ),
                    "observed_gain": (
                        observed_gain
                    ),
                    "fitted_gain": (
                        fitted_gain
                    ),
                }
            )

        print()
        print(
            f"{metric}"
        )

        print(
            "  tau additional days: "
            f"{tau:.3f} "
            f"[{tau_low:.3f}, "
            f"{tau_high:.3f}]"
        )

        print(
            "  asymptotic gain:      "
            f"{asymptote:.4f} "
            f"[{amp_low:.4f}, "
            f"{amp_high:.4f}]"
        )

        print(
            "  L50 context:          "
            f"{l50:.2f} d"
        )

        print(
            "  L90 context:          "
            f"{l90:.2f} d"
        )

        print(
            "  R2:                   "
            f"{observed['r2']:.4f}"
        )

    # -------------------------------------------------------------
    # Primary theoretical comparison:
    # does entity information accumulate more slowly?
    # -------------------------------------------------------------

    generic_tau = (
        model_bootstrap[
            "generic_gain"
        ][
            "tau"
        ]
    )

    identity_tau = (
        model_bootstrap[
            "identity_gap"
        ][
            "tau"
        ]
    )

    total_tau = (
        model_bootstrap[
            "total_gain"
        ][
            "tau"
        ]
    )

    tau_difference = (
        identity_tau
        - generic_tau
    )

    diff_low, diff_high = (
        np.quantile(
            tau_difference,
            [
                0.025,
                0.975,
            ],
        )
    )

    probability = float(
        np.mean(
            identity_tau
            > generic_tau
        )
    )

    print()
    print(
        "Primary staged-acquisition test"
    )

    print(
        "  P(tau_identity > "
        "tau_generic): "
        f"{probability:.4f}"
    )

    print(
        "  bootstrap difference "
        "95% interval: "
        f"[{diff_low:.3f}, "
        f"{diff_high:.3f}] days"
    )

    for b in range(
        N_BOOTSTRAP
    ):
        bootstrap_rows.append(
            {
                "model": model,
                "bootstrap_index": b,
                "tau_total": (
                    total_tau[b]
                ),
                "tau_generic": (
                    generic_tau[b]
                ),
                "tau_identity": (
                    identity_tau[b]
                ),
                "identity_minus_generic_tau": (
                    tau_difference[b]
                ),
            }
        )


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

summary = pd.DataFrame(
    summary_rows
)

fits = pd.DataFrame(
    fit_rows
)

bootstrap = pd.DataFrame(
    bootstrap_rows
)


summary_path = (
    OUT_DIR
    / "acquisition_scale_summary.csv"
)

fits_path = (
    OUT_DIR
    / "acquisition_curve_fits.csv"
)

bootstrap_path = (
    OUT_DIR
    / "bootstrap_acquisition_scales.csv"
)


summary.to_csv(
    summary_path,
    index=False,
)

fits.to_csv(
    fits_path,
    index=False,
)

bootstrap.to_csv(
    bootstrap_path,
    index=False,
)


print()
print("=" * 100)
print("Summary")
print("=" * 100)
print()

display = summary.copy()

numeric_cols = [
    "tau_additional_days",
    "tau_ci_low",
    "tau_ci_high",
    "asymptotic_gain",
    "l50_context_days",
    "l90_context_days",
    "r2",
]

for column in numeric_cols:
    display[column] = (
        display[column]
        .round(4)
    )

print(
    display[
        [
            "model",
            "mechanism",
            "tau_additional_days",
            "tau_ci_low",
            "tau_ci_high",
            "asymptotic_gain",
            "l50_context_days",
            "l90_context_days",
            "r2",
        ]
    ].to_string(
        index=False
    )
)

print()
print("Saved:")
print(f"  {sensor_path}")
print(f"  {summary_path}")
print(f"  {fits_path}")
print(f"  {bootstrap_path}")
