from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RAW = Path(
    "data/raw/physionet2019/"
    "physionet.org/files/challenge-2019/1.0.0/"
    "training"
)

OUT = Path(
    "data/processed/physionet2019"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


LONG_CONTEXT = 48
HORIZON = 6
MIN_HISTORY_COVERAGE = 0.75
MIN_ORIGIN_SPACING = 6
N_ORIGINS = 4
SEED = 42

CONTEXT_GRID = [
    6,
    12,
    24,
    36,
    48,
]


def valid_origins(hr):
    observed = np.isfinite(hr)

    cumulative = np.concatenate(
        [
            [0],
            np.cumsum(
                observed.astype(int)
            ),
        ]
    )

    valid = []

    n = len(hr)

    for origin in range(
        LONG_CONTEXT,
        n - HORIZON + 1,
    ):
        future_complete = observed[
            origin:
            origin + HORIZON
        ].all()

        if not future_complete:
            continue

        coverage_ok = True

        for context in CONTEXT_GRID:
            history_n = (
                cumulative[origin]
                - cumulative[
                    origin - context
                ]
            )

            history_coverage = (
                history_n
                / context
            )

            if (
                history_coverage
                < MIN_HISTORY_COVERAGE
            ):
                coverage_ok = False
                break

        if coverage_ok:
            valid.append(origin)

    return valid


def spaced_origins(origins):
    selected = []

    last = None

    for origin in origins:
        if (
            last is None
            or origin - last
            >= MIN_ORIGIN_SPACING
        ):
            selected.append(origin)
            last = origin

    return selected


def choose_four(origins):
    indices = np.rint(
        np.linspace(
            0,
            len(origins) - 1,
            N_ORIGINS,
        )
    ).astype(int)

    chosen = [
        origins[i]
        for i in indices
    ]

    if len(set(chosen)) != N_ORIGINS:
        raise RuntimeError(
            "Origin selection produced duplicates."
        )

    return chosen


files = sorted(
    RAW.rglob("*.psv")
)

print("=" * 90)
print("PHYSIONET 2019 | FROZEN HR COHORT")
print("=" * 90)
print("Raw files:", len(files))
print()


entity_rows = []
series_rows = []
origin_rows = []


for j, path in enumerate(
    files,
    start=1,
):
    df = pd.read_csv(
        path,
        sep="|",
        usecols=["HR"],
    )

    hr = pd.to_numeric(
        df["HR"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    valid = valid_origins(hr)

    spaced = spaced_origins(
        valid
    )

    if len(spaced) < N_ORIGINS:
        if (
            j == 1
            or j % 5000 == 0
            or j == len(files)
        ):
            print(
                f"[{j:5d}/{len(files)}]"
            )
        continue

    selected = choose_four(
        spaced
    )

    entity = path.stem

    source_set = (
        "A"
        if "training_setA" in path.parts
        else "B"
    )

    coverage = float(
        np.isfinite(hr).mean()
    )

    entity_rows.append(
        {
            "entity": entity,
            "source_set": source_set,
            "n_hours": len(hr),
            "hr_coverage": coverage,
            "n_valid_origins_48h": len(valid),
            "n_spaced_origins_48h": len(spaced),
            "first_selected_origin_h": selected[0],
            "last_selected_origin_h": selected[-1],
            "selected_origin_span_h": (
                selected[-1]
                - selected[0]
            ),
        }
    )

    for hour, value in enumerate(hr):
        series_rows.append(
            {
                "entity": entity,
                "hour": hour,
                "value": value,
                "source_set": source_set,
            }
        )

    for eval_index, origin in enumerate(
        selected
    ):
        history = hr[
            origin - LONG_CONTEXT:
            origin
        ]

        history_coverage = float(
            np.isfinite(
                history
            ).mean()
        )

        origin_rows.append(
            {
                "entity": entity,
                "source_set": source_set,
                "eval_index": eval_index,
                "origin_hour": origin,
                "history_coverage_48h": (
                    history_coverage
                ),
            }
        )

    if (
        j == 1
        or j % 5000 == 0
        or j == len(files)
    ):
        print(
            f"[{j:5d}/{len(files)}]"
        )


entities = pd.DataFrame(
    entity_rows
).sort_values(
    "entity"
).reset_index(
    drop=True
)

series = pd.DataFrame(
    series_rows
).sort_values(
    [
        "entity",
        "hour",
    ]
).reset_index(
    drop=True
)

origins = pd.DataFrame(
    origin_rows
).sort_values(
    [
        "entity",
        "eval_index",
    ]
).reset_index(
    drop=True
)


# ------------------------------------------------------------------
# Frozen entity-level split
# ------------------------------------------------------------------

train, remaining = train_test_split(
    entities,
    test_size=0.30,
    random_state=SEED,
    stratify=entities[
        "source_set"
    ],
)

validation, heldout = train_test_split(
    remaining,
    test_size=0.50,
    random_state=SEED,
    stratify=remaining[
        "source_set"
    ],
)


split_rows = []

for split_name, frame in [
    ("train", train),
    ("validation", validation),
    ("heldout", heldout),
]:
    for row in frame.itertuples():
        split_rows.append(
            {
                "entity": row.entity,
                "source_set": row.source_set,
                "split": split_name,
            }
        )


splits = pd.DataFrame(
    split_rows
).sort_values(
    "entity"
).reset_index(
    drop=True
)


split_map = dict(
    zip(
        splits["entity"],
        splits["split"],
    )
)

origins["split"] = origins[
    "entity"
].map(
    split_map
)


# ------------------------------------------------------------------
# Integrity checks
# ------------------------------------------------------------------

assert len(entities) == 930

assert (
    entities[
        "source_set"
    ].value_counts().to_dict()
    == {
        "A": 489,
        "B": 441,
    }
)

assert origins.shape[0] == (
    len(entities)
    * N_ORIGINS
)

assert origins[
    "history_coverage_48h"
].min() >= MIN_HISTORY_COVERAGE


for entity, group in origins.groupby(
    "entity"
):
    x = np.sort(
        group[
            "origin_hour"
        ].to_numpy()
    )

    assert len(x) == N_ORIGINS

    assert np.all(
        np.diff(x)
        >= MIN_ORIGIN_SPACING
    )


# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

entities.to_csv(
    OUT
    / "physionet2019_entities.csv",
    index=False,
)

series.to_csv(
    OUT
    / "physionet2019_series.csv",
    index=False,
)

origins.to_csv(
    OUT
    / "physionet2019_evaluation_origins.csv",
    index=False,
)

splits.to_csv(
    OUT
    / "physionet2019_entity_splits.csv",
    index=False,
)


protocol = {
    "dataset": "physionet2019",
    "target": "HR",
    "entity_definition": "ICU record/stay",
    "resolution_hours": 1,
    "forecast_horizon_hours": HORIZON,
    "context_grid_hours": CONTEXT_GRID,
    "maximum_context_hours": LONG_CONTEXT,
    "minimum_history_coverage": (
        MIN_HISTORY_COVERAGE
    ),
    "coverage_requirement": (
        "minimum coverage required "
        "at every context length"
    ),
    "coverage_contexts_hours": (
        CONTEXT_GRID
    ),
    "origins_per_entity": N_ORIGINS,
    "minimum_origin_spacing_hours": (
        MIN_ORIGIN_SPACING
    ),
    "split_seed": SEED,
    "split_fractions": {
        "train": 0.70,
        "validation": 0.15,
        "heldout": 0.15,
    },
    "split_stratification": "source_set",
    "matched_other_restriction": (
        "same source_set and same eval_index"
    ),
    "heldout_status": "FROZEN_UNTOUCHED",
}

with open(
    OUT
    / "physionet2019_protocol.json",
    "w",
) as f:
    json.dump(
        protocol,
        f,
        indent=2,
    )


# ------------------------------------------------------------------
# Terminal QC
# ------------------------------------------------------------------

print()
print("=" * 90)
print("FROZEN COHORT")
print("=" * 90)

print(
    "Entities:",
    len(entities),
)

print()
print("Source sets:")
print(
    entities[
        "source_set"
    ].value_counts().sort_index()
)

print()
print("Split counts:")
print(
    splits[
        "split"
    ].value_counts()
)

print()
print("Split x source:")
print(
    pd.crosstab(
        splits["split"],
        splits["source_set"],
    )
)

print()
print("Evaluation origins:")
print(
    origins[
        "split"
    ].value_counts()
)

print()
print(
    "Minimum selected 48h "
    "history coverage:",
    round(
        origins[
            "history_coverage_48h"
        ].min(),
        4,
    ),
)

print(
    "Median selected 48h "
    "history coverage:",
    round(
        origins[
            "history_coverage_48h"
        ].median(),
        4,
    ),
)

print()
print(
    "Series rows:",
    len(series),
)

missing_fraction = float(
    series[
        "value"
    ].isna().mean()
)

print(
    "Raw HR missing fraction:",
    round(
        missing_fraction,
        4,
    ),
)

print()
print("Saved:")
for path in sorted(
    OUT.glob(
        "physionet2019_*"
    )
):
    print(
        " ",
        path,
    )

print()
print(
    "STATUS: "
    "PHYSIONET2019_HR_PROTOCOL_FROZEN_GRID_COVERAGE"
)
