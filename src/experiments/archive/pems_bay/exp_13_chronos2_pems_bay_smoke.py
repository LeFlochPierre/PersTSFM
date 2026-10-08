from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chronos import Chronos2Pipeline


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RAW_PATH = Path(
    "data/raw/pems_bay/PEMS-BAY.csv"
)

PROTOCOL_PATH = Path(
    "data/processed/pems_bay/"
    "pems_bay_evaluation_origins.csv"
)

MODEL_ID = "amazon/chronos-2"

CONTEXT_HOURS = 744
CONTEXT_STEPS = 8928

FORECAST_STEPS = 12
RESOLUTION_MINUTES = 5


# ---------------------------------------------------------------------
# Pick first validation case
# ---------------------------------------------------------------------

protocol = pd.read_csv(
    PROTOCOL_PATH,
    parse_dates=[
        "eval_origin",
        "target_start",
        "target_end",
    ],
)

protocol["sensor"] = (
    protocol["sensor"]
    .astype(str)
)

case = (
    protocol[
        protocol["split"] == "validation"
    ]
    .sort_values(
        [
            "sensor",
            "eval_index",
        ]
    )
    .iloc[0]
)

sensor = case["sensor"]
origin = case["eval_origin"]


# ---------------------------------------------------------------------
# Load one sensor
# ---------------------------------------------------------------------

header = pd.read_csv(
    RAW_PATH,
    nrows=0,
)

timestamp_col = header.columns[0]

raw = pd.read_csv(
    RAW_PATH,
    usecols=[
        timestamp_col,
        sensor,
    ],
    low_memory=False,
)

raw[timestamp_col] = pd.to_datetime(
    raw[timestamp_col]
)

raw = (
    raw
    .set_index(timestamp_col)
    .sort_index()
)


# ---------------------------------------------------------------------
# Context and target
# ---------------------------------------------------------------------

context_start = (
    origin
    - pd.Timedelta(
        minutes=(
            CONTEXT_STEPS - 1
        )
        * RESOLUTION_MINUTES
    )
)

context = raw.loc[
    context_start:origin,
    sensor,
]

truth = raw.loc[
    case["target_start"]:
    case["target_end"],
    sensor,
]


print("=" * 78)
print("PersTSFM Chronos-2 PEMS-BAY max-context smoke test")
print("=" * 78)

print()
print(f"Sensor:          {sensor}")
print(f"Origin:          {origin}")
print(f"Context hours:   {CONTEXT_HOURS}")
print(f"Context steps:   {len(context)}")
print(f"Forecast steps:  {len(truth)}")


if len(context) != CONTEXT_STEPS:
    raise RuntimeError(
        f"Expected {CONTEXT_STEPS} "
        f"context points, got "
        f"{len(context)}."
    )

if len(truth) != FORECAST_STEPS:
    raise RuntimeError(
        f"Expected {FORECAST_STEPS} "
        f"target points, got "
        f"{len(truth)}."
    )


# ---------------------------------------------------------------------
# Build Chronos dataframe
# ---------------------------------------------------------------------

context_df = pd.DataFrame(
    {
        "id": sensor,
        "timestamp": context.index,
        "target": context.to_numpy(
            dtype=float
        ),
    }
)


# ---------------------------------------------------------------------
# Load Chronos
# ---------------------------------------------------------------------

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print(f"Loading {MODEL_ID}")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        "GPU: "
        + torch.cuda.get_device_name(0)
    )

pipeline = Chronos2Pipeline.from_pretrained(
    MODEL_ID,
    device_map=device,
)


# ---------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------

pred_df = pipeline.predict_df(
    context_df,
    prediction_length=FORECAST_STEPS,
    quantile_levels=[
        0.1,
        0.5,
        0.9,
    ],
    id_column="id",
    timestamp_column="timestamp",
    target="target",
    context_length=CONTEXT_STEPS,
    batch_size=1,
    freq="5min",
)


pred_df = pred_df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


if len(pred_df) != FORECAST_STEPS:
    raise RuntimeError(
        f"Expected {FORECAST_STEPS} "
        f"prediction rows, got "
        f"{len(pred_df)}."
    )


prediction = pred_df[
    "predictions"
].to_numpy(
    dtype=float
)

truth_values = truth.to_numpy(
    dtype=float
)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

mae = float(
    np.mean(
        np.abs(
            prediction
            - truth_values
        )
    )
)

rmse = float(
    np.sqrt(
        np.mean(
            (
                prediction
                - truth_values
            )
            ** 2
        )
    )
)


print()
print("=" * 78)
print("Smoke-test result")
print("=" * 78)

print()
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

print()
print("Forecast:")

display = pd.DataFrame(
    {
        "timestamp": truth.index,
        "truth": truth_values,
        "prediction": prediction,
    }
)

print()
print(
    display
    .round(3)
    .to_string(
        index=False
    )
)
