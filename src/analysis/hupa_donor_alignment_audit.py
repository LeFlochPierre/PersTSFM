import numpy as np
import pandas as pd

from src.data.registry import (
    context_steps,
    extract_case,
    load_dataset,
)


RECENT_HOURS = 24
LONG_HOURS_LIST = [
    96,   # Chronos: 4 d
    168,  # TimesFM: 7 d
]


def standardize(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    mean = float(
        np.mean(values)
    )

    std = float(
        np.std(values)
    )

    if std < 1e-8:
        std = 1.0

    return (
        values - mean
    ) / std


dataset = load_dataset(
    "hupa_ucm"
)

spec = dataset.spec
protocol = dataset.protocol.copy()

print("=" * 90)
print("HUPA-UCM donor alignment and relative-window audit")
print("=" * 90)

print()
print("Entities:", len(dataset.series))
print("Protocol cases:", len(protocol))
print("Resolution minutes:", spec.resolution_minutes)
print("ID column:", spec.id_col)


for long_hours in LONG_HOURS_LIST:

    print()
    print("=" * 90)
    print(
        f"LONG CONTEXT = {long_hours} h "
        f"({long_hours / 24:.0f} d)"
    )
    print("=" * 90)

    n_recent = context_steps(
        RECENT_HOURS,
        spec.resolution_minutes,
    )

    # ---------------------------------------------------------
    # Precompute all valid relative-time donor windows.
    # These are taken ONLY from frozen protocol cases.
    # ---------------------------------------------------------

    candidate_windows = []

    for donor_row_index, donor_row in (
        protocol.iterrows()
    ):
        try:
            donor_case = extract_case(
                dataset,
                donor_row,
                long_hours,
            )
        except Exception:
            continue

        donor_context = donor_case[
            "context"
        ]

        donor_recent = (
            donor_context.iloc[
                -n_recent:
            ]
        )

        if len(donor_recent) != n_recent:
            continue

        if donor_recent.isna().any():
            continue

        candidate_windows.append(
            {
                "row_index": (
                    donor_row_index
                ),
                "entity_id": (
                    str(
                        donor_case[
                            "entity_id"
                        ]
                    )
                ),
                "eval_index": int(
                    donor_case[
                        "eval_index"
                    ]
                ),
                "eval_origin": (
                    donor_case[
                        "eval_origin"
                    ]
                ),
                "recent_values": (
                    donor_recent.to_numpy(
                        dtype=float
                    )
                ),
            }
        )

    print()
    print(
        "Valid frozen donor windows:",
        len(candidate_windows),
    )

    print(
        "Distinct donor entities:",
        len(
            {
                x["entity_id"]
                for x in candidate_windows
            }
        ),
    )

    # ---------------------------------------------------------
    # Audit each target case.
    # ---------------------------------------------------------

    rows = []

    for target_row_index, target_row in (
        protocol.iterrows()
    ):
        target_case = extract_case(
            dataset,
            target_row,
            long_hours,
        )

        target_id = str(
            target_case[
                "entity_id"
            ]
        )

        target_recent = (
            target_case[
                "context"
            ]
            .iloc[
                -n_recent:
            ]
        )

        target_values = (
            target_recent.to_numpy(
                dtype=float
            )
        )

        target_z = standardize(
            target_values
        )

        # -----------------------------------------------------
        # Existing exact-calendar donor availability.
        # -----------------------------------------------------

        exact_calendar = 0

        for donor_id, donor_series in (
            dataset.series.items()
        ):
            if str(donor_id) == target_id:
                continue

            aligned = donor_series.reindex(
                target_recent.index
            )

            if not aligned.isna().any():
                exact_calendar += 1

        # -----------------------------------------------------
        # Relative-time candidate windows from other people.
        # -----------------------------------------------------

        candidates = []

        for donor in candidate_windows:

            if (
                donor[
                    "entity_id"
                ]
                == target_id
            ):
                continue

            donor_z = standardize(
                donor[
                    "recent_values"
                ]
            )

            distance = float(
                np.sqrt(
                    np.mean(
                        (
                            target_z
                            - donor_z
                        )
                        ** 2
                    )
                )
            )

            candidates.append(
                (
                    distance,
                    donor,
                )
            )

        candidates.sort(
            key=lambda x: (
                x[0],
                x[1][
                    "entity_id"
                ],
                x[1][
                    "eval_index"
                ],
            )
        )

        if candidates:
            best_distance = float(
                candidates[0][0]
            )

            best_donor = (
                candidates[0][1]
            )

            selected_id = (
                best_donor[
                    "entity_id"
                ]
            )

            selected_eval_index = (
                best_donor[
                    "eval_index"
                ]
            )
        else:
            best_distance = np.nan
            selected_id = None
            selected_eval_index = None

        rows.append(
            {
                "target_entity": (
                    target_id
                ),
                "target_eval_index": int(
                    target_case[
                        "eval_index"
                    ]
                ),
                "exact_calendar_donors": (
                    exact_calendar
                ),
                "relative_candidate_windows": (
                    len(
                        candidates
                    )
                ),
                "relative_candidate_entities": (
                    len(
                        {
                            x[1][
                                "entity_id"
                            ]
                            for x
                            in candidates
                        }
                    )
                ),
                "best_relative_distance": (
                    best_distance
                ),
                "selected_donor": (
                    selected_id
                ),
                "selected_donor_eval_index": (
                    selected_eval_index
                ),
            }
        )

    audit = pd.DataFrame(
        rows
    )

    print()
    print("Exact-calendar donor availability")
    print("-" * 90)

    print(
        audit[
            "exact_calendar_donors"
        ]
        .describe()
        .round(3)
        .to_string()
    )

    print()
    print(
        "Cases with >=1 exact-calendar donor:",
        int(
            (
                audit[
                    "exact_calendar_donors"
                ]
                >= 1
            ).sum()
        ),
        "/",
        len(audit),
    )

    print()
    print("Relative-time donor availability")
    print("-" * 90)

    print(
        audit[
            [
                "relative_candidate_windows",
                "relative_candidate_entities",
                "best_relative_distance",
            ]
        ]
        .describe()
        .round(3)
        .to_string()
    )

    print()
    print(
        "Cases with >=1 relative donor:",
        int(
            (
                audit[
                    "relative_candidate_windows"
                ]
                >= 1
            ).sum()
        ),
        "/",
        len(audit),
    )

    print()
    print("Selected donor reuse")
    print("-" * 90)

    print(
        audit[
            "selected_donor"
        ]
        .value_counts()
        .head(10)
        .to_string()
    )

    output = (
        "results/hupa_ucm/"
        f"hupa_relative_donor_audit_{long_hours}h.csv"
    )

    audit.to_csv(
        output,
        index=False,
    )

    print()
    print(
        "Saved:",
        output,
    )
