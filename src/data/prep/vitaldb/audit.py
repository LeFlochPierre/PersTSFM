from pathlib import Path

import numpy as np
import pandas as pd
import vitaldb


OUT_DIR = Path(
    "results/vitaldb/audit"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TRACKS = {
    "heart_rate": "Solar8000/HR",
    "art_mbp": "Solar8000/ART_MBP",
    "nibp_mbp": "Solar8000/NIBP_MBP",
}


print("=" * 90)
print("VITALDB INITIAL AUDIT")
print("=" * 90)


clinical = pd.read_csv(
    "https://api.vitaldb.net/cases"
)

print()
print("Clinical cases:", len(clinical))
print("Clinical columns:")
print(
    list(clinical.columns)
)


rows = []


for label, track in TRACKS.items():

    try:
        caseids = vitaldb.find_cases(
            [track]
        )
    except Exception as exc:
        print()
        print(
            f"{label}: FAILED"
        )
        print(exc)
        continue

    caseids = list(
        caseids
    )

    print()
    print(
        f"{label}: {track}"
    )

    print(
        "Cases:",
        len(caseids)
    )

    for caseid in caseids:
        rows.append(
            {
                "caseid": int(
                    caseid
                ),
                "signal": label,
                "track": track,
            }
        )


tracks = pd.DataFrame(
    rows
)

tracks.to_csv(
    OUT_DIR
    / "available_tracks.csv",
    index=False,
)


if len(tracks) > 0:

    counts = (
        tracks
        .groupby(
            "caseid"
        )[
            "signal"
        ]
        .nunique()
    )

    print()
    print("=" * 90)
    print("TRACK OVERLAP")
    print("=" * 90)

    print(
        counts
        .value_counts()
        .sort_index()
        .to_string()
    )

    all_three = counts[
        counts >= 3
    ].index

    any_map = (
        tracks[
            tracks[
                "signal"
            ].isin(
                [
                    "art_mbp",
                    "nibp_mbp",
                ]
            )
        ][
            "caseid"
        ]
        .unique()
    )

    print()
    print(
        "Cases with all 3 candidate tracks:",
        len(all_three),
    )

    print(
        "Cases with any MAP track:",
        len(any_map),
    )


clinical.to_csv(
    OUT_DIR
    / "clinical_metadata.csv",
    index=False,
)

print()
print("Saved to:", OUT_DIR)
