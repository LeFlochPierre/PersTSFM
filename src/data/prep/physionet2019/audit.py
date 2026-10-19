from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path(
    "data/raw/physionet2019"
)

OUT_DIR = Path(
    "results/physionet2019/audit"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TARGETS = [
    "HR",
    "MAP",
    "O2Sat",
    "Resp",
]

files = sorted(
    RAW_DIR.rglob("*.psv")
)

print("=" * 100)
print("PHYSIONET 2019 ICU AUDIT")
print("=" * 100)

print()
print(
    "Patient files:",
    len(files),
)


rows = []


for i, path in enumerate(files):

    if (
        i == 0
        or (i + 1) % 5000 == 0
        or i + 1 == len(files)
    ):
        print(
            f"[{i + 1:>5}/{len(files)}]"
        )


    df = pd.read_csv(
        path,
        sep="|",
    )


    patient_id = path.stem

    source_set = (
        "A"
        if "training_setA"
        in str(path)
        else "B"
    )


    n_hours = len(df)


    result = {
        "patient_id": (
            patient_id
        ),
        "source_set": (
            source_set
        ),
        "n_hours": (
            n_hours
        ),
        "icu_los_last": (
            df["ICULOS"].iloc[-1]
            if "ICULOS" in df.columns
            else np.nan
        ),
        "ever_sepsis": (
            int(
                df["SepsisLabel"]
                .fillna(0)
                .max()
            )
            if "SepsisLabel"
            in df.columns
            else np.nan
        ),
    }


    for target in TARGETS:

        y = pd.to_numeric(
            df[target],
            errors="coerce",
        ).to_numpy()


        observed = np.isfinite(
            y
        )


        result[
            f"{target}_n_observed"
        ] = int(
            observed.sum()
        )


        result[
            f"{target}_coverage"
        ] = float(
            observed.mean()
        )


        # Longest gap between observations
        # in hourly samples.
        observed_idx = np.flatnonzero(
            observed
        )


        if len(observed_idx) >= 2:

            gaps = np.diff(
                observed_idx
            ) - 1

            result[
                f"{target}_max_gap_h"
            ] = int(
                gaps.max()
            )

        elif len(observed_idx) == 1:

            result[
                f"{target}_max_gap_h"
            ] = (
                n_hours - 1
            )

        else:

            result[
                f"{target}_max_gap_h"
            ] = np.nan


        # Number of origins where the previous
        # 24 h contain >=75% observed target
        # measurements and the following 6 h
        # target is completely observed.
        #
        # This is diagnostic only; protocol is
        # not frozen yet.
        n_candidate_24h = 0
        n_candidate_48h = 0


        for context_h, key in [
            (
                24,
                "24h",
            ),
            (
                48,
                "48h",
            ),
        ]:

            count = 0


            for origin in range(
                context_h,
                n_hours - 6 + 1,
            ):

                history = observed[
                    origin
                    - context_h:
                    origin
                ]

                future = observed[
                    origin:
                    origin + 6
                ]


                if (
                    history.mean()
                    >= 0.75
                    and np.all(
                        future
                    )
                ):
                    count += 1


            result[
                f"{target}_candidate_"
                f"{key}_6h"
            ] = count


    rows.append(
        result
    )


audit = pd.DataFrame(
    rows
)


out_path = (
    OUT_DIR
    / "patient_audit.csv"
)

audit.to_csv(
    out_path,
    index=False,
)


print()
print("=" * 100)
print("STAY LENGTH")
print("=" * 100)

print(
    audit[
        "n_hours"
    ]
    .describe(
        percentiles=[
            0.1,
            0.25,
            0.5,
            0.75,
            0.9,
        ]
    )
    .round(2)
    .to_string()
)


for target in TARGETS:

    print()
    print("=" * 100)
    print(target)
    print("=" * 100)

    print()
    print("Coverage:")

    print(
        audit[
            f"{target}_coverage"
        ]
        .describe(
            percentiles=[
                0.1,
                0.25,
                0.5,
                0.75,
                0.9,
            ]
        )
        .round(3)
        .to_string()
    )


    for context_h in [
        24,
        48,
    ]:

        col = (
            f"{target}_candidate_"
            f"{context_h}h_6h"
        )

        print()
        print(
            f"Patients with candidate "
            f"{context_h}h context + "
            f"6h horizon:"
        )

        for n_origins in [
            1,
            2,
            4,
            8,
        ]:

            n = int(
                (
                    audit[col]
                    >= n_origins
                ).sum()
            )

            print(
                f"  >= {n_origins} origins: "
                f"{n}"
            )


print()
print("=" * 100)
print("SOURCE SETS")
print("=" * 100)

print(
    audit[
        "source_set"
    ]
    .value_counts()
    .to_string()
)

print()
print("Saved:")
print(out_path)
