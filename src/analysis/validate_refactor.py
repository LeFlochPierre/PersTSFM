from pathlib import Path

import numpy as np
import pandas as pd


TOLERANCE = 1e-5


CHECKS = [
    {
        "name": "HUPA Chronos-2",
        "old": Path(
            "results/hupa_ucm/context/"
            "chronos2_hupa_context_summary.csv"
        ),
        "new": Path(
            "results/modular/hupa_ucm/context/"
            "chronos2/summary.csv"
        ),
        "old_context": "context_hours",
        "new_context": "context_hours",
        "old_metric": "mean_participant_mae",
        "new_metric": "mean_entity_mae",
    },
    {
        "name": "HUPA TimesFM 3",
        "old": Path(
            "results/hupa_ucm/context/"
            "timesfm3_hupa_context_summary.csv"
        ),
        "new": Path(
            "results/modular/hupa_ucm/context/"
            "timesfm3/summary.csv"
        ),
        "old_context": "context_hours",
        "new_context": "context_hours",
        "old_metric": "mean_participant_mae",
        "new_metric": "mean_entity_mae",
    },
    {
        "name": "PEMS-BAY Chronos-2",
        "old": Path(
            "results/pems_bay/context/"
            "chronos2_pems_bay_context_summary.csv"
        ),
        "new": Path(
            "results/modular/pems_bay/context/"
            "chronos2/summary.csv"
        ),
        "old_context": "context_hours",
        "new_context": "context_hours",
        "old_metric": "mean_mae",
        "new_metric": "mean_entity_mae",
    },
    {
        "name": "PEMS-BAY TimesFM 3",
        "old": Path(
            "results/pems_bay/context/"
            "timesfm3_pems_bay_context_summary.csv"
        ),
        "new": Path(
            "results/modular/pems_bay/context/"
            "timesfm3/summary.csv"
        ),
        "old_context": "context_hours",
        "new_context": "context_hours",
        "old_metric": "mean_mae",
        "new_metric": "mean_entity_mae",
    },
    {
        "name": "Electricity Chronos-2",
        "old": Path(
            "results/context_threshold/"
            "chronos2_context_summary.csv"
        ),
        "new": Path(
            "results/modular/electricity/context/"
            "chronos2/summary.csv"
        ),
        "old_context": "context_days",
        "new_context": "context_hours",
        "old_metric": "mean_mae",
        "new_metric": "mean_entity_mae",
    },
    {
        "name": "Electricity TimesFM 3",
        "old": Path(
            "results/context_threshold/"
            "timesfm3_context_summary.csv"
        ),
        "new": Path(
            "results/modular/electricity/context/"
            "timesfm3/summary.csv"
        ),
        "old_context": "context_days",
        "new_context": "context_hours",
        "old_metric": "mean_mae",
        "new_metric": "mean_entity_mae",
    },
]


def load_check(spec):
    old = pd.read_csv(
        spec["old"]
    )

    new = pd.read_csv(
        spec["new"]
    )

    old_context = old[
        spec["old_context"]
    ].astype(float)

    if spec["old_context"] == "context_days":
        old_context = (
            old_context * 24.0
        )

    old_clean = pd.DataFrame(
        {
            "context_hours": (
                old_context
            ),
            "old_mae": old[
                spec["old_metric"]
            ].astype(float),
        }
    )

    new_clean = pd.DataFrame(
        {
            "context_hours": new[
                spec["new_context"]
            ].astype(float),
            "new_mae": new[
                spec["new_metric"]
            ].astype(float),
        }
    )

    merged = old_clean.merge(
        new_clean,
        on="context_hours",
        how="inner",
        validate="one_to_one",
    )

    merged["difference"] = (
        merged["new_mae"]
        - merged["old_mae"]
    )

    return merged


print("=" * 88)
print("PersTSFM refactor reproduction audit")
print("=" * 88)

all_passed = True

for spec in CHECKS:
    print()
    print(spec["name"])

    if not spec["old"].exists():
        print(
            f"  OLD FILE MISSING: "
            f"{spec['old']}"
        )
        all_passed = False
        continue

    if not spec["new"].exists():
        print(
            f"  NEW FILE MISSING: "
            f"{spec['new']}"
        )
        all_passed = False
        continue

    result = load_check(
        spec
    )

    maximum = float(
        result[
            "difference"
        ].abs().max()
    )

    passed = bool(
        maximum <= TOLERANCE
    )

    print(
        f"  matched contexts: "
        f"{len(result)}"
    )

    print(
        f"  max absolute difference: "
        f"{maximum:.10g}"
    )

    print(
        f"  status: "
        f"{'PASS' if passed else 'FAIL'}"
    )

    if not passed:
        all_passed = False

        print()
        print(
            result
            .sort_values(
                "context_hours"
            )
            .to_string(
                index=False
            )
        )


print()
print("=" * 88)

if all_passed:
    print(
        "ALL REFACTOR CHECKS PASSED"
    )
else:
    print(
        "ONE OR MORE REFACTOR CHECKS FAILED"
    )

print("=" * 88)

if not all_passed:
    raise SystemExit(1)
