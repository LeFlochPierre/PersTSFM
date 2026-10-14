from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Dataset specification
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    key: str
    id_col: str

    protocol_path: Path
    raw_path: Path

    default_split: str

    resolution_minutes: int
    forecast_steps: int

    context_hours: tuple
    reference_context_hours: int

    chronos_batch_size: int
    timesfm_batch_size: int

    loader_kind: str


@dataclass
class LoadedDataset:
    spec: DatasetSpec
    protocol: pd.DataFrame
    series: dict


# ---------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------

DATASETS = {
    "electricity": DatasetSpec(
        key="electricity",
        id_col="entity",

        protocol_path=Path(
            "data/processed/electricity/"
            "electricity_evaluation_origins.csv"
        ),

        raw_path=Path(
            "data/raw/electricity/"
            "LD2011_2014.txt"
        ),

        default_split="validation",

        resolution_minutes=15,
        forecast_steps=96,

        # Full development context curve:
        # 3-31 days.
        context_hours=tuple(
            day * 24
            for day in range(3, 32)
        ),

        reference_context_hours=(
            4 * 24
        ),

        chronos_batch_size=64,
        timesfm_batch_size=64,

        loader_kind="electricity",
    ),

    "pems_bay": DatasetSpec(
        key="pems_bay",
        id_col="sensor",

        protocol_path=Path(
            "data/processed/pems_bay/"
            "pems_bay_evaluation_origins.csv"
        ),

        raw_path=Path(
            "data/raw/pems_bay/"
            "PEMS-BAY.csv"
        ),

        default_split="validation",

        resolution_minutes=5,
        forecast_steps=12,

        context_hours=(
            3,
            6,
            12,
            24,
            48,
            72,
            120,
            168,
            240,
            336,
            504,
            672,
        ),

        reference_context_hours=24,

        chronos_batch_size=32,
        timesfm_batch_size=16,

        loader_kind="pems_bay",
    ),

    "hupa_ucm": DatasetSpec(
        key="hupa_ucm",
        id_col="person_id",

        protocol_path=Path(
            "data/processed/hupa_ucm/"
            "hupa_ucm_evaluation_origins.csv"
        ),

        raw_path=Path(
            "data/raw/hupa_ucm/"
            "HUPA-UCM-extracted-glucose-files"
        ),

        default_split="development",

        resolution_minutes=5,
        forecast_steps=12,

        context_hours=(
            3,
            6,
            12,
            18,
            24,
            36,
            48,
            72,
            96,
            120,
            168,
        ),

        reference_context_hours=24,

        chronos_batch_size=32,
        timesfm_batch_size=16,

        loader_kind="hupa_ucm",
    ),
    "vitaldb": DatasetSpec(
        key="vitaldb",
        id_col="caseid",

        protocol_path=Path(
            "data/processed/vitaldb/"
            "vitaldb_context_protocol.csv"
        ),

        raw_path=Path(
            "data/processed/vitaldb/"
            "vitaldb_art_mbp_1min.csv"
        ),

        default_split="validation",

        resolution_minutes=1,
        forecast_steps=15,

        context_hours=(
            0.5,
            1,
            2,
            3,
            4,
        ),

        reference_context_hours=0.5,

        chronos_batch_size=64,
        timesfm_batch_size=64,

        loader_kind="vitaldb",
    ),

    "solar": DatasetSpec(
        key="solar",
        id_col="entity_key",

        protocol_path=Path(
            "data/processed/solar/"
            "solar_context_protocol.csv"
        ),

        raw_path=Path(
            "data/processed/solar/"
            "solar_series.csv"
        ),

        default_split="validation",

        resolution_minutes=10,
        forecast_steps=6,

        context_hours=(
            6,
            12,
            24,
            48,
            96,
        ),

        reference_context_hours=6,

        chronos_batch_size=32,
        timesfm_batch_size=16,

        loader_kind="solar",
    ),
}


# ---------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------

def get_dataset_spec(
    dataset_key,
):
    if dataset_key not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_key}'. "
            f"Available datasets: "
            f"{sorted(DATASETS)}"
        )

    return DATASETS[
        dataset_key
    ]


def context_label(
    context_hours,
):
    context_hours = float(
        context_hours
    )

    if context_hours < 24:
        if context_hours.is_integer():
            return (
                f"{int(context_hours)}h"
            )

        return (
            f"{context_hours:g}h"
        )

    if context_hours % 24 == 0:
        days = (
            context_hours / 24
        )

        if days.is_integer():
            return f"{int(days)}d"

        return f"{days:g}d"

    if context_hours.is_integer():
        return (
            f"{int(context_hours)}h"
        )

    return f"{context_hours:g}h"


def context_steps(
    context_hours,
    resolution_minutes,
):
    steps = (
        float(context_hours)
        * 60.0
        / float(
            resolution_minutes
        )
    )

    rounded = int(
        round(steps)
    )

    if not np.isclose(
        steps,
        rounded,
    ):
        raise ValueError(
            f"Context {context_hours} h "
            f"is not divisible by "
            f"{resolution_minutes} min."
        )

    return rounded


# ---------------------------------------------------------------------
# Protocol loading
# ---------------------------------------------------------------------

def load_protocol(
    spec,
    split=None,
):
    if split is None:
        split = spec.default_split

    protocol = pd.read_csv(
        spec.protocol_path,
        parse_dates=[
            "eval_origin",
            "target_start",
            "target_end",
        ],
    )

    if spec.id_col not in (
        protocol.columns
    ):
        raise RuntimeError(
            f"{spec.key}: expected ID "
            f"column '{spec.id_col}'. "
            f"Columns were "
            f"{list(protocol.columns)}"
        )

    protocol[
        spec.id_col
    ] = (
        protocol[
            spec.id_col
        ]
        .astype(str)
    )

    protocol = (
        protocol[
            protocol[
                "split"
            ] == split
        ]
        .copy()
        .sort_values(
            [
                spec.id_col,
                "eval_index",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(protocol) == 0:
        raise RuntimeError(
            f"{spec.key}: no cases "
            f"for split '{split}'."
        )

    return protocol


# ---------------------------------------------------------------------
# Raw dataset loaders
# ---------------------------------------------------------------------

def _load_electricity(
    spec,
    entity_ids,
):
    header = pd.read_csv(
        spec.raw_path,
        sep=";",
        decimal=",",
        nrows=0,
    )

    timestamp_col = (
        header.columns[0]
    )

    available = set(
        header.columns
    )

    missing = [
        entity
        for entity in entity_ids
        if entity not in available
    ]

    if missing:
        raise RuntimeError(
            "Electricity entities "
            f"missing from raw file: "
            f"{missing[:10]}"
        )

    raw = pd.read_csv(
        spec.raw_path,
        sep=";",
        decimal=",",
        usecols=[
            timestamp_col,
            *entity_ids,
        ],
        low_memory=False,
    )

    raw[
        timestamp_col
    ] = pd.to_datetime(
        raw[
            timestamp_col
        ],
        format="%Y-%m-%d %H:%M:%S",
        errors="raise",
    )

    raw = (
        raw
        .set_index(
            timestamp_col
        )
        .sort_index()
    )

    result = {}

    for entity in entity_ids:
        result[
            entity
        ] = (
            pd.to_numeric(
                raw[entity],
                errors="raise",
            )
            .astype(float)
        )

    return result


def _load_pems_bay(
    spec,
    sensor_ids,
):
    header = pd.read_csv(
        spec.raw_path,
        nrows=0,
    )

    timestamp_col = (
        header.columns[0]
    )

    available = set(
        header.columns
    )

    missing = [
        sensor
        for sensor in sensor_ids
        if sensor not in available
    ]

    if missing:
        raise RuntimeError(
            "PEMS sensors missing "
            f"from raw file: "
            f"{missing[:10]}"
        )

    raw = pd.read_csv(
        spec.raw_path,
        usecols=[
            timestamp_col,
            *sensor_ids,
        ],
        low_memory=False,
    )

    raw[
        timestamp_col
    ] = pd.to_datetime(
        raw[
            timestamp_col
        ],
        errors="raise",
    )

    raw = (
        raw
        .set_index(
            timestamp_col
        )
        .sort_index()
    )

    result = {}

    for sensor in sensor_ids:
        result[
            sensor
        ] = (
            pd.to_numeric(
                raw[sensor],
                errors="raise",
            )
            .astype(float)
        )

    return result


def _load_hupa_ucm(
    spec,
    person_ids,
):
    result = {}

    for person_id in person_ids:

        path = (
            spec.raw_path
            / f"{person_id}.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing HUPA file: "
                f"{path}"
            )

        df = pd.read_csv(
            path,
            usecols=[
                "timestamp",
                "glucose_value_mg_dl",
            ],
        )

        df[
            "timestamp"
        ] = pd.to_datetime(
            df[
                "timestamp"
            ],
            errors="raise",
        )

        df[
            "glucose_value_mg_dl"
        ] = pd.to_numeric(
            df[
                "glucose_value_mg_dl"
            ],
            errors="raise",
        )

        df = (
            df
            .sort_values(
                "timestamp"
            )
            .drop_duplicates(
                subset="timestamp",
                keep="first",
            )
            .set_index(
                "timestamp"
            )
        )

        result[
            person_id
        ] = (
            df[
                "glucose_value_mg_dl"
            ]
            .astype(float)
        )

    return result


def _load_vitaldb(
    spec,
    case_ids,
):
    requested = {
        str(x)
        for x in case_ids
    }

    raw = pd.read_csv(
        spec.raw_path,
        usecols=[
            "caseid",
            "relative_minute",
            "map",
        ],
    )

    raw["caseid"] = (
        raw["caseid"]
        .astype(str)
    )

    raw = raw[
        raw["caseid"].isin(
            requested
        )
    ].copy()

    base = pd.Timestamp(
        "2000-01-01 00:00:00"
    )

    result = {}

    for case_id, part in raw.groupby(
        "caseid",
        sort=False,
    ):
        part = part.sort_values(
            "relative_minute"
        )

        index = (
            base
            + pd.to_timedelta(
                part[
                    "relative_minute"
                ],
                unit="min",
            )
        )

        values = pd.to_numeric(
            part["map"],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

        result[
            str(case_id)
        ] = pd.Series(
            values,
            index=pd.DatetimeIndex(
                index
            ),
        )

    missing = (
        requested
        - set(result)
    )

    if missing:
        raise RuntimeError(
            "Missing VitalDB cases: "
            f"{sorted(missing)[:10]}"
        )

    return result


def _load_solar(
    spec,
    entity_ids,
):
    requested = {
        str(x)
        for x in entity_ids
    }

    raw = pd.read_csv(
        spec.raw_path,
        usecols=[
            "entity_key",
            "datetime",
            "value",
        ],
    )

    raw["entity_key"] = (
        raw["entity_key"]
        .astype(str)
    )

    raw = raw[
        raw["entity_key"].isin(
            requested
        )
    ].copy()

    raw["datetime"] = pd.to_datetime(
        raw["datetime"],
        errors="raise",
    )

    result = {}

    for entity_id, part in raw.groupby(
        "entity_key",
        sort=False,
    ):
        part = (
            part
            .sort_values(
                "datetime"
            )
            .drop_duplicates(
                subset="datetime",
                keep="first",
            )
        )

        values = pd.to_numeric(
            part["value"],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

        result[
            str(entity_id)
        ] = pd.Series(
            values,
            index=pd.DatetimeIndex(
                part["datetime"]
            ),
        )

    missing = (
        requested
        - set(result)
    )

    if missing:
        raise RuntimeError(
            "Missing Solar entities: "
            f"{sorted(missing)[:10]}"
        )

    return result


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_dataset(
    dataset_key,
    split=None,
):
    spec = get_dataset_spec(
        dataset_key
    )

    protocol = load_protocol(
        spec,
        split=split,
    )

    ids = sorted(
        protocol[
            spec.id_col
        ].unique()
    )

    if (
        spec.loader_kind
        == "electricity"
    ):
        series = (
            _load_electricity(
                spec,
                ids,
            )
        )

    elif (
        spec.loader_kind
        == "pems_bay"
    ):
        series = (
            _load_pems_bay(
                spec,
                ids,
            )
        )

    elif (
        spec.loader_kind
        == "hupa_ucm"
    ):
        series = (
            _load_hupa_ucm(
                spec,
                ids,
            )
        )

    elif (
        spec.loader_kind
        == "vitaldb"
    ):
        series = (
            _load_vitaldb(
                spec,
                ids,
            )
        )

    elif (
        spec.loader_kind
        == "solar"
    ):
        series = (
            _load_solar(
                spec,
                ids,
            )
        )

    else:
        raise RuntimeError(
            f"Unknown loader kind: "
            f"{spec.loader_kind}"
        )

    return LoadedDataset(
        spec=spec,
        protocol=protocol,
        series=series,
    )


# ---------------------------------------------------------------------
# Case extraction
# ---------------------------------------------------------------------

def extract_case(
    dataset,
    row,
    context_hours_value,
):
    spec = dataset.spec

    entity_id = str(
        row[
            spec.id_col
        ]
    )

    series = dataset.series[
        entity_id
    ]

    n_context_steps = (
        context_steps(
            context_hours_value,
            spec.resolution_minutes,
        )
    )

    context_span = (
        n_context_steps - 1
    ) * pd.Timedelta(
        minutes=(
            spec.resolution_minutes
        )
    )

    context_start = (
        row[
            "eval_origin"
        ]
        - context_span
    )

    context = series.loc[
        context_start:
        row[
            "eval_origin"
        ]
    ]

    truth = series.loc[
        row[
            "target_start"
        ]:
        row[
            "target_end"
        ]
    ]

    if (
        len(context)
        != n_context_steps
    ):
        raise RuntimeError(
            f"{spec.key} / "
            f"{entity_id} / "
            f"eval {row['eval_index']}: "
            f"expected "
            f"{n_context_steps} "
            f"context points, "
            f"found {len(context)}."
        )

    if (
        len(truth)
        != spec.forecast_steps
    ):
        raise RuntimeError(
            f"{spec.key} / "
            f"{entity_id} / "
            f"eval {row['eval_index']}: "
            f"expected "
            f"{spec.forecast_steps} "
            f"target points, "
            f"found {len(truth)}."
        )

    case_id = (
        f"{entity_id}__"
        f"{int(row['eval_index']):02d}"
    )

    return {
        "case_id": case_id,
        "entity_id": entity_id,
        "eval_index": int(
            row[
                "eval_index"
            ]
        ),
        "eval_origin": row[
            "eval_origin"
        ],
        "context": context,
        "truth": truth,
        "context_steps": (
            n_context_steps
        ),
    }
