from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from src.data.registry import (
    load_dataset,
    extract_case,
    context_steps,
)

from src.evaluation.context_intervention import (
    _safe_standardize,
)


RESULTS = Path("results")

OUT = (
    RESULTS
    / "context_regime_comparison"
    / "matched_history_divergence"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


CONFIGS = {
    "pems_bay": {
        "label": "PEMS-BAY",
        "recent_hours": 24,
        "long_hours": {
            "chronos2": 240,
            "timesfm3": 504,
        },
    },
    "electricity": {
        "label": "Electricity",
        "recent_hours": 96,
        "long_hours": {
            "chronos2": 192,
            "timesfm3": 240,
        },
    },
    "hupa_ucm": {
        "label": "HUPA CGM",
        "recent_hours": 24,
        "long_hours": {
            "chronos2": 96,
            "timesfm3": 168,
        },
    },
    "vitaldb": {
        "label": "VitalDB MAP",
        "recent_hours": 0.5,
        "long_hours": {
            "chronos2": 4,
            "timesfm3": 4,
        },
    },
    "solar": {
        "label": "Solar PV",
        "recent_hours": 24,
        "long_hours": {
            "chronos2": 96,
            "timesfm3": 96,
        },
    },
    "physionet2019": {
        "label": "PhysioNet HR",
        "recent_hours": 6,
        "long_hours": {
            "chronos2": 48,
            "timesfm3": 48,
        },
    },
}


MODELS = [
    "chronos2",
    "timesfm3",
]


MODEL_LABELS = {
    "chronos2": "Chronos-2",
    "timesfm3": "TimesFM3",
}


RELATIVE_DONOR_DATASETS = {
    "hupa_ucm",
    "vitaldb",
    "solar",
    "physionet2019",
}


EPS = 1e-12


def result_folder(
    dataset,
    model,
    recent_hours,
    long_hours,
):
    return (
        RESULTS
        / dataset
        / "interventions"
        / model
        / (
            f"recent_{recent_hours:g}h"
            f"_long_{long_hours:g}h"
        )
    )


def rmse(
    a,
    b,
):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                (
                    a - b
                )
                ** 2
            )
        )
    )


def get_protocol_row(
    dataset,
    entity_id,
    eval_index,
):
    spec = dataset.spec

    rows = dataset.protocol[
        (
            dataset.protocol[
                spec.id_col
            ].astype(str)
            == str(entity_id)
        )
        &
        (
            dataset.protocol[
                "eval_index"
            ].astype(int)
            == int(eval_index)
        )
    ]

    if len(rows) != 1:
        raise RuntimeError(
            f"{spec.key}: expected one "
            f"protocol row for "
            f"{entity_id}/eval={eval_index}, "
            f"found {len(rows)}."
        )

    return rows.iloc[0]


def read_manifest(
    dataset_key,
    model,
    recent_hours,
    long_hours,
):
    path = (
        result_folder(
            dataset_key,
            model,
            recent_hours,
            long_hours,
        )
        / "donor_manifest.csv"
    )

    df = pd.read_csv(
        path,
        dtype={
            "entity_id": str,
            "donor_id": str,
        },
    )

    return df


def verify_manifests(
    a,
    b,
    dataset_key,
):
    cols = [
        "case_id",
        "entity_id",
        "eval_index",
        "donor_id",
    ]

    aa = (
        a[
            cols
        ]
        .sort_values(
            "case_id"
        )
        .reset_index(
            drop=True
        )
    )

    bb = (
        b[
            cols
        ]
        .sort_values(
            "case_id"
        )
        .reset_index(
            drop=True
        )
    )

    if not aa.equals(
        bb
    ):
        raise RuntimeError(
            f"{dataset_key}: donor identities "
            "differ between models."
        )

    da = (
        a.sort_values(
            "case_id"
        )[
            "donor_distance"
        ]
        .to_numpy(
            dtype=float
        )
    )

    db = (
        b.sort_values(
            "case_id"
        )[
            "donor_distance"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.allclose(
        da,
        db,
        atol=1e-10,
        rtol=1e-8,
    ):
        raise RuntimeError(
            f"{dataset_key}: donor distances "
            "differ between models."
        )


def reconstruct_geometry(
    dataset_key,
    recent_hours,
    long_hours,
    manifest,
):
    dataset = load_dataset(
        dataset_key
    )

    spec = dataset.spec

    n_recent = context_steps(
        recent_hours,
        spec.resolution_minutes,
    )

    n_long = context_steps(
        long_hours,
        spec.resolution_minutes,
    )

    n_old = (
        n_long
        - n_recent
    )

    rows = []

    for i, m in (
        manifest.iterrows()
    ):
        target_id = str(
            m[
                "entity_id"
            ]
        )

        donor_id = str(
            m[
                "donor_id"
            ]
        )

        eval_index = int(
            m[
                "eval_index"
            ]
        )

        target_row = (
            get_protocol_row(
                dataset,
                target_id,
                eval_index,
            )
        )

        target_long_case = (
            extract_case(
                dataset,
                target_row,
                long_hours,
            )
        )

        target_full = (
            target_long_case[
                "context"
            ]
        )

        target_old = (
            target_full.iloc[
                :n_old
            ]
        )

        if (
            dataset_key
            == "physionet2019"
        ):
            target_recent = (
                extract_case(
                    dataset,
                    target_row,
                    recent_hours,
                )[
                    "context"
                ]
            )

        else:
            target_recent = (
                target_full.iloc[
                    n_old:
                ]
            )

        # ----------------------------------------------------
        # Reconstruct the exact donor history used by the
        # intervention.
        # ----------------------------------------------------

        if (
            dataset_key
            in RELATIVE_DONOR_DATASETS
        ):
            donor_eval_index = int(
                m[
                    "donor_eval_index"
                ]
            )

            donor_row = (
                get_protocol_row(
                    dataset,
                    donor_id,
                    donor_eval_index,
                )
            )

            donor_long_case = (
                extract_case(
                    dataset,
                    donor_row,
                    long_hours,
                )
            )

            donor_full = (
                donor_long_case[
                    "context"
                ]
            )

            donor_old = (
                donor_full.iloc[
                    :n_old
                ]
            )

            if (
                dataset_key
                == "physionet2019"
            ):
                donor_recent = (
                    extract_case(
                        dataset,
                        donor_row,
                        recent_hours,
                    )[
                        "context"
                    ]
                )

            else:
                donor_recent = (
                    donor_full.iloc[
                        n_old:
                    ]
                )

        else:
            # PEMS and Electricity donors use the
            # target's absolute timestamps.
            donor_series = (
                dataset.series[
                    donor_id
                ]
            )

            donor_old = (
                donor_series.reindex(
                    target_old.index
                )
            )

            donor_recent = (
                donor_series.reindex(
                    target_recent.index
                )
            )

        arrays = [
            target_old,
            target_recent,
            donor_old,
            donor_recent,
        ]

        if any(
            x.isna().any()
            for x in arrays
        ):
            raise RuntimeError(
                f"{dataset_key}: NaN while "
                f"reconstructing {target_id}/"
                f"{eval_index}."
            )

        target_old_values = (
            target_old.to_numpy(
                dtype=float
            )
        )

        target_recent_values = (
            target_recent.to_numpy(
                dtype=float
            )
        )

        donor_old_values = (
            donor_old.to_numpy(
                dtype=float
            )
        )

        donor_recent_values = (
            donor_recent.to_numpy(
                dtype=float
            )
        )

        # ----------------------------------------------------
        # Recent matching distance.
        # Recompute it and verify it exactly reproduces
        # the value saved during intervention construction.
        # ----------------------------------------------------

        (
            target_recent_z,
            target_recent_mean,
            target_recent_std,
        ) = _safe_standardize(
            target_recent_values
        )

        (
            donor_recent_z,
            donor_recent_mean,
            donor_recent_std,
        ) = _safe_standardize(
            donor_recent_values
        )

        reconstructed_recent_distance = (
            rmse(
                target_recent_z,
                donor_recent_z,
            )
        )

        saved_recent_distance = float(
            m[
                "donor_distance"
            ]
        )

        if not np.isclose(
            reconstructed_recent_distance,
            saved_recent_distance,
            atol=1e-8,
            rtol=1e-6,
        ):
            raise RuntimeError(
                f"{dataset_key}: donor-distance "
                f"reconstruction failed for "
                f"{target_id}/{eval_index}: "
                f"{reconstructed_recent_distance} "
                f"vs {saved_recent_distance}."
            )

        # ----------------------------------------------------
        # OLD-HISTORY DIVERGENCE
        #
        # Express each old history relative to its own
        # recent mean/std. This is exactly the coordinate
        # system implied by the affine donor intervention.
        # ----------------------------------------------------

        target_old_recent_scaled = (
            (
                target_old_values
                - target_recent_mean
            )
            / target_recent_std
        )

        donor_old_recent_scaled = (
            (
                donor_old_values
                - donor_recent_mean
            )
            / donor_recent_std
        )

        old_context_divergence = (
            rmse(
                target_old_recent_scaled,
                donor_old_recent_scaled,
            )
        )

        # Same quantity reconstructed in raw target units:
        donor_old_affine = (
            (
                donor_old_values
                - donor_recent_mean
            )
            / donor_recent_std
            * target_recent_std
            + target_recent_mean
        )

        affine_input_distance_norm = (
            rmse(
                target_old_values,
                donor_old_affine,
            )
            / target_recent_std
        )

        if not np.isclose(
            old_context_divergence,
            affine_input_distance_norm,
            atol=1e-8,
            rtol=1e-6,
        ):
            raise RuntimeError(
                "Affine divergence identity failed."
            )

        # Pure old-history shape difference after
        # independently standardizing the old blocks.
        (
            target_old_z,
            _,
            _,
        ) = _safe_standardize(
            target_old_values
        )

        (
            donor_old_z,
            _,
            _,
        ) = _safe_standardize(
            donor_old_values
        )

        old_shape_distance = rmse(
            target_old_z,
            donor_old_z,
        )

        divergence_increment = (
            old_context_divergence
            - reconstructed_recent_distance
        )

        divergence_ratio = (
            old_context_divergence
            / (
                reconstructed_recent_distance
                + EPS
            )
        )

        boundary_excess_norm = (
            float(
                m[
                    "affine_boundary_jump_norm"
                ]
            )
            - float(
                m[
                    "same_boundary_jump_norm"
                ]
            )
        )

        rows.append(
            {
                "case_id": (
                    m[
                        "case_id"
                    ]
                ),
                "entity_id": (
                    target_id
                ),
                "eval_index": (
                    eval_index
                ),
                "donor_id": (
                    donor_id
                ),
                "recent_distance": (
                    reconstructed_recent_distance
                ),
                "old_context_divergence": (
                    old_context_divergence
                ),
                "old_shape_distance": (
                    old_shape_distance
                ),
                "divergence_increment": (
                    divergence_increment
                ),
                "divergence_ratio": (
                    divergence_ratio
                ),
                "same_boundary_jump_norm": (
                    float(
                        m[
                            "same_boundary_jump_norm"
                        ]
                    )
                ),
                "affine_boundary_jump_norm": (
                    float(
                        m[
                            "affine_boundary_jump_norm"
                        ]
                    )
                ),
                "boundary_excess_norm": (
                    boundary_excess_norm
                ),
            }
        )

        if (
            i == 0
            or (i + 1) % 250 == 0
            or (i + 1) == len(
                manifest
            )
        ):
            print(
                f"  geometry "
                f"{i + 1:4d}/"
                f"{len(manifest)}"
            )

    return pd.DataFrame(
        rows
    )


def load_outcomes(
    dataset_key,
    model,
    recent_hours,
    long_hours,
):
    path = (
        result_folder(
            dataset_key,
            model,
            recent_hours,
            long_hours,
        )
        / "case_metrics.csv"
    )

    df = pd.read_csv(
        path,
        dtype={
            "entity_id": str,
            "donor_id": str,
        },
    )

    df = (
        df[
            df[
                "intervention"
            ]
            == "matched_other_affine"
        ]
        .copy()
    )

    # Positive = correct entity history is better.
    df[
        "entity_gain_case"
    ] = (
        df[
            "mae"
        ]
        - df[
            "same_mae"
        ]
    )

    # Positive = generic wrong-entity history
    # is still better than short context.
    df[
        "generic_gain_case"
    ] = (
        df[
            "short_mae"
        ]
        - df[
            "mae"
        ]
    )

    df[
        "total_gain_case"
    ] = (
        df[
            "short_mae"
        ]
        - df[
            "same_mae"
        ]
    )

    return df[
        [
            "entity_id",
            "eval_index",
            "donor_id",
            "entity_gain_case",
            "generic_gain_case",
            "total_gain_case",
        ]
    ]


def spearman(
    x,
    y,
):
    valid = (
        np.isfinite(
            x
        )
        & np.isfinite(
            y
        )
    )

    rho, p = spearmanr(
        np.asarray(
            x
        )[valid],
        np.asarray(
            y
        )[valid],
    )

    return (
        float(rho),
        float(p),
        int(
            valid.sum()
        ),
    )


PREDICTORS = [
    "old_context_divergence",
    "divergence_increment",
    "divergence_ratio",
    "old_shape_distance",
    "recent_distance",
    "affine_boundary_jump_norm",
    "boundary_excess_norm",
]


all_case_rows = []
all_entity_rows = []
correlation_rows = []
entity_tables = {}


print("=" * 104)
print(
    "MATCHED-HISTORY DIVERGENCE ANALYSIS"
)
print("=" * 104)


for dataset_key, config in (
    CONFIGS.items()
):
    recent_hours = float(
        config[
            "recent_hours"
        ]
    )

    print()
    print("=" * 104)
    print(
        config[
            "label"
        ]
    )
    print("=" * 104)

    for model in MODELS:

        long_hours = float(
            config[
                "long_hours"
            ][
                model
            ]
        )

        print()
        print(
            f"{MODEL_LABELS[model]} | "
            f"context={recent_hours:g} h "
            f"-> {long_hours:g} h"
        )

        manifest = read_manifest(
            dataset_key,
            model,
            recent_hours,
            long_hours,
        )

        geometry = (
            reconstruct_geometry(
                dataset_key,
                recent_hours,
                long_hours,
                manifest,
            )
        )

        geometry.to_csv(
            OUT
            / (
                f"{dataset_key}_"
                f"{model}_"
                "matched_history_geometry.csv"
            ),
            index=False,
        )

        outcomes = load_outcomes(
            dataset_key,
            model,
            recent_hours,
            long_hours,
        )

        merged = geometry.merge(
            outcomes,
            on=[
                "entity_id",
                "eval_index",
                "donor_id",
            ],
            how="inner",
            validate="one_to_one",
        )

        if (
            len(merged)
            != len(
                geometry
            )
        ):
            raise RuntimeError(
                f"{dataset_key}/{model}: "
                "geometry/outcome merge "
                "lost cases."
            )

        merged[
            "dataset"
        ] = dataset_key

        merged[
            "dataset_label"
        ] = config[
            "label"
        ]

        merged[
            "model"
        ] = model

        merged[
            "model_label"
        ] = MODEL_LABELS[
            model
        ]

        all_case_rows.append(
            merged
        )

        # ---------------------------------------------
        # Primary unit: entity, not individual case.
        # This respects the repeated-origin design.
        # ---------------------------------------------

        aggregate_cols = (
            PREDICTORS
            + [
                "entity_gain_case",
                "generic_gain_case",
                "total_gain_case",
            ]
        )

        entity = (
            merged.groupby(
                "entity_id",
                as_index=False,
            )[
                aggregate_cols
            ]
            .mean()
        )

        entity[
            "dataset"
        ] = dataset_key

        entity[
            "dataset_label"
        ] = config[
            "label"
        ]

        entity[
            "model"
        ] = model

        entity[
            "model_label"
        ] = MODEL_LABELS[
            model
        ]

        entity[
            "entity_gain"
        ] = entity[
            "entity_gain_case"
        ]

        entity[
            "generic_gain"
        ] = entity[
            "generic_gain_case"
        ]

        entity[
            "total_gain"
        ] = entity[
            "total_gain_case"
        ]

        all_entity_rows.append(
            entity
        )

        entity_tables[
            (
                dataset_key,
                model,
            )
        ] = entity

        print()
        print(
            f"{MODEL_LABELS[model]} "
            f"| N entities={len(entity)}"
        )

        for predictor in (
            PREDICTORS
        ):
            rho, p, n = spearman(
                entity[
                    predictor
                ].to_numpy(),
                entity[
                    "entity_gain"
                ].to_numpy(),
            )

            correlation_rows.append(
                {
                    "dataset": (
                        dataset_key
                    ),
                    "dataset_label": (
                        config[
                            "label"
                        ]
                    ),
                    "model": (
                        model
                    ),
                    "model_label": (
                        MODEL_LABELS[
                            model
                        ]
                    ),
                    "level": (
                        "entity"
                    ),
                    "predictor": (
                        predictor
                    ),
                    "outcome": (
                        "entity_gain"
                    ),
                    "n": (
                        n
                    ),
                    "rho": (
                        rho
                    ),
                    "p": (
                        p
                    ),
                }
            )

        # Case-level correlation is descriptive only.
        for predictor in (
            PREDICTORS
        ):
            rho, p, n = spearman(
                merged[
                    predictor
                ].to_numpy(),
                merged[
                    "entity_gain_case"
                ].to_numpy(),
            )

            correlation_rows.append(
                {
                    "dataset": (
                        dataset_key
                    ),
                    "dataset_label": (
                        config[
                            "label"
                        ]
                    ),
                    "model": (
                        model
                    ),
                    "model_label": (
                        MODEL_LABELS[
                            model
                        ]
                    ),
                    "level": (
                        "case_descriptive"
                    ),
                    "predictor": (
                        predictor
                    ),
                    "outcome": (
                        "entity_gain"
                    ),
                    "n": (
                        n
                    ),
                    "rho": (
                        rho
                    ),
                    "p": (
                        p
                    ),
                }
            )


cases = pd.concat(
    all_case_rows,
    ignore_index=True,
)

entities = pd.concat(
    all_entity_rows,
    ignore_index=True,
)

correlations = pd.DataFrame(
    correlation_rows
)


cases.to_csv(
    OUT
    / "all_case_level_divergence.csv",
    index=False,
)

entities.to_csv(
    OUT
    / "all_entity_level_divergence.csv",
    index=False,
)

correlations.to_csv(
    OUT
    / "divergence_gain_correlations.csv",
    index=False,
)


# ============================================================
# Primary results table
# ============================================================

primary_predictors = [
    "old_context_divergence",
    "divergence_increment",
    "old_shape_distance",
    "recent_distance",
    "boundary_excess_norm",
]


primary = (
    correlations[
        (
            correlations[
                "level"
            ]
            == "entity"
        )
        &
        (
            correlations[
                "predictor"
            ].isin(
                primary_predictors
            )
        )
    ]
    .copy()
)


print()
print("=" * 104)
print(
    "PRIMARY ENTITY-LEVEL CORRELATIONS"
)
print(
    "Positive rho: larger divergence "
    "associated with larger wrong-entity penalty"
)
print("=" * 104)


for predictor in (
    primary_predictors
):
    print()
    print(
        predictor
    )

    part = (
        primary[
            primary[
                "predictor"
            ]
            == predictor
        ]
        [
            [
                "dataset_label",
                "model_label",
                "n",
                "rho",
                "p",
            ]
        ]
    )

    print(
        part.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:8.3f}"
            ),
        )
    )


# ============================================================
# Within-domain standardized pooled analysis
# ============================================================

pooled_rows = []


for model in MODELS:

    part = entities[
        entities[
            "model"
        ] == model
    ].copy()

    for column in [
        "old_context_divergence",
        "divergence_increment",
        "old_shape_distance",
        "boundary_excess_norm",
        "entity_gain",
    ]:

        part[
            f"{column}_z"
        ] = (
            part.groupby(
                "dataset"
            )[
                column
            ]
            .transform(
                lambda x: (
                    (
                        x
                        - x.mean()
                    )
                    / (
                        x.std(
                            ddof=1
                        )
                        + EPS
                    )
                )
            )
        )

    for predictor in [
        "old_context_divergence",
        "divergence_increment",
        "old_shape_distance",
        "boundary_excess_norm",
    ]:

        rho, p, n = spearman(
            part[
                f"{predictor}_z"
            ].to_numpy(),
            part[
                "entity_gain_z"
            ].to_numpy(),
        )

        pooled_rows.append(
            {
                "model": (
                    model
                ),
                "model_label": (
                    MODEL_LABELS[
                        model
                    ]
                ),
                "predictor": (
                    predictor
                ),
                "n_entities": (
                    n
                ),
                "rho": (
                    rho
                ),
                "p": (
                    p
                ),
            }
        )


pooled = pd.DataFrame(
    pooled_rows
)

pooled.to_csv(
    OUT
    / "within_domain_standardized_pooled.csv",
    index=False,
)


print()
print("=" * 104)
print(
    "POOLED WITHIN-DOMAIN STANDARDIZED ANALYSIS"
)
print("=" * 104)

print(
    pooled.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:8.3f}"
        ),
    )
)


# ============================================================
# Cross-model agreement within entities
# ============================================================

agreement_rows = []


for dataset_key, config in (
    CONFIGS.items()
):

    c = entity_tables[
        (
            dataset_key,
            "chronos2",
        )
    ][
        [
            "entity_id",
            "entity_gain",
            "generic_gain",
            "total_gain",
        ]
    ].copy()

    t = entity_tables[
        (
            dataset_key,
            "timesfm3",
        )
    ][
        [
            "entity_id",
            "entity_gain",
            "generic_gain",
            "total_gain",
        ]
    ].copy()

    merged = c.merge(
        t,
        on="entity_id",
        suffixes=(
            "_chronos",
            "_timesfm",
        ),
        validate="one_to_one",
    )

    for effect in [
        "entity_gain",
        "generic_gain",
        "total_gain",
    ]:

        rho, p, n = spearman(
            merged[
                f"{effect}_chronos"
            ].to_numpy(),
            merged[
                f"{effect}_timesfm"
            ].to_numpy(),
        )

        agreement_rows.append(
            {
                "dataset": (
                    dataset_key
                ),
                "dataset_label": (
                    config[
                        "label"
                    ]
                ),
                "effect": (
                    effect
                ),
                "n_entities": (
                    n
                ),
                "rho": (
                    rho
                ),
                "p": (
                    p
                ),
            }
        )


agreement = pd.DataFrame(
    agreement_rows
)

agreement.to_csv(
    OUT
    / "cross_model_entity_level_agreement.csv",
    index=False,
)


print()
print("=" * 104)
print(
    "CROSS-MODEL AGREEMENT WITHIN ENTITIES"
)
print("=" * 104)

print(
    agreement.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:8.3f}"
        ),
    )
)


# ============================================================
# Figures
# ============================================================

def save(
    fig,
    filename,
):
    fig.tight_layout()

    fig.savefig(
        OUT
        / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT
        / f"{filename}.pdf",
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# One entity-level plot for every dataset/model.
for (
    dataset_key,
    model,
), entity in (
    entity_tables.items()
):

    label = CONFIGS[
        dataset_key
    ][
        "label"
    ]

    rho, p, _ = spearman(
        entity[
            "divergence_increment"
        ].to_numpy(),
        entity[
            "entity_gain"
        ].to_numpy(),
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.scatter(
        entity[
            "divergence_increment"
        ],
        entity[
            "entity_gain"
        ],
        s=55,
        alpha=0.75,
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Historical divergence increment\n"
        "(old-context divergence − recent matching distance)"
    )

    ax.set_ylabel(
        "Entity / episode gain in MAE"
    )

    ax.set_title(
        (
            f"{label} — "
            f"{MODEL_LABELS[model]}\n"
            f"Spearman ρ={rho:.3f}, "
            f"p={p:.3g}"
        )
    )

    save(
        fig,
        (
            f"divergence_vs_gain_"
            f"{dataset_key}_{model}"
        ),
    )


# Boundary-artifact diagnostic for every dataset/model.
for (
    dataset_key,
    model,
), entity in (
    entity_tables.items()
):

    label = CONFIGS[
        dataset_key
    ][
        "label"
    ]

    rho, p, _ = spearman(
        entity[
            "boundary_excess_norm"
        ].to_numpy(),
        entity[
            "entity_gain"
        ].to_numpy(),
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.scatter(
        entity[
            "boundary_excess_norm"
        ],
        entity[
            "entity_gain"
        ],
        s=55,
        alpha=0.75,
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Extra affine splice boundary jump\n"
        "relative to same-entity history"
    )

    ax.set_ylabel(
        "Entity / episode gain in MAE"
    )

    ax.set_title(
        (
            f"Boundary artifact check: "
            f"{label} — "
            f"{MODEL_LABELS[model]}\n"
            f"Spearman ρ={rho:.3f}, "
            f"p={p:.3g}"
        )
    )

    save(
        fig,
        (
            f"boundary_vs_gain_"
            f"{dataset_key}_{model}"
        ),
    )


# Pooled within-domain standardized plots.
for model in MODELS:

    part = entities[
        entities[
            "model"
        ] == model
    ].copy()

    part[
        "divergence_z"
    ] = (
        part.groupby(
            "dataset"
        )[
            "divergence_increment"
        ]
        .transform(
            lambda x: (
                (
                    x - x.mean()
                )
                / (
                    x.std(
                        ddof=1
                    )
                    + EPS
                )
            )
        )
    )

    part[
        "gain_z"
    ] = (
        part.groupby(
            "dataset"
        )[
            "entity_gain"
        ]
        .transform(
            lambda x: (
                (
                    x - x.mean()
                )
                / (
                    x.std(
                        ddof=1
                    )
                    + EPS
                )
            )
        )
    )

    rho, p, _ = spearman(
        part[
            "divergence_z"
        ].to_numpy(),
        part[
            "gain_z"
        ].to_numpy(),
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.scatter(
        part[
            "divergence_z"
        ],
        part[
            "gain_z"
        ],
        s=40,
        alpha=0.6,
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Historical divergence increment\n"
        "(within-domain z-score)"
    )

    ax.set_ylabel(
        "Entity / episode gain\n"
        "(within-domain z-score)"
    )

    ax.set_title(
        (
            f"Pooled across domains — "
            f"{MODEL_LABELS[model]}\n"
            f"Spearman ρ={rho:.3f}, "
            f"p={p:.3g}"
        )
    )

    save(
        fig,
        (
            "pooled_standardized_"
            f"{model}"
        ),
    )


print()
print("=" * 104)
print("SAVED")
print("=" * 104)
print(OUT)
