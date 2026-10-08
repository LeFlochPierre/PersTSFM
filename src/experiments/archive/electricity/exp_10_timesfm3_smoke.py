from pathlib import Path

import numpy as np
import pandas as pd
import torch

from timesfm3 import (
    ModelConfig,
    TimesFM3Evaluator,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RAW_PATH = Path(
    "data/raw/electricity/LD2011_2014.txt"
)

PROTOCOL_PATH = Path(
    "data/processed/electricity/"
    "electricity_evaluation_origins.csv"
)

MODEL_ID = "google/timesfm-3.0-pytorch"

CONTEXT_DAYS = 4
CONTEXT_STEPS = CONTEXT_DAYS * 96
FORECAST_STEPS = 96


# ---------------------------------------------------------------------
# First validation case
# ---------------------------------------------------------------------

protocol = pd.read_csv(
    PROTOCOL_PATH,
    parse_dates=[
        "eval_origin",
        "target_start",
        "target_end",
    ],
)

case = (
    protocol[
        protocol["split"] == "validation"
    ]
    .sort_values(
        [
            "entity",
            "eval_index",
        ]
    )
    .iloc[0]
)

entity = case["entity"]
origin = case["eval_origin"]

context_start = (
    origin
    - pd.Timedelta(
        minutes=15
        * (
            CONTEXT_STEPS - 1
        )
    )
)


print("=" * 76)
print("PersTSFM TimesFM 3 smoke test")
print("=" * 76)

print()
print(f"Entity:        {entity}")
print(f"Origin:        {origin}")
print(f"Context days:  {CONTEXT_DAYS}")
print(f"Context steps: {CONTEXT_STEPS}")
print(f"Target steps:  {FORECAST_STEPS}")


# ---------------------------------------------------------------------
# Load one entity
# ---------------------------------------------------------------------

header = pd.read_csv(
    RAW_PATH,
    sep=";",
    decimal=",",
    nrows=0,
)

timestamp_col = header.columns[0]

raw = pd.read_csv(
    RAW_PATH,
    sep=";",
    decimal=",",
    usecols=[
        timestamp_col,
        entity,
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


context = raw.loc[
    context_start:origin,
    entity,
].to_numpy(
    dtype=np.float32
)

truth = raw.loc[
    case["target_start"]:
    case["target_end"],
    entity,
].to_numpy(
    dtype=np.float32
)


assert len(context) == CONTEXT_STEPS
assert len(truth) == FORECAST_STEPS


# ---------------------------------------------------------------------
# Load TimesFM 3
# ---------------------------------------------------------------------

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

config = ModelConfig(
    checkpoint_path=MODEL_ID,
    per_core_batch_size=16,
    device=device,
)

print()
print(f"Loading {MODEL_ID}")
print(f"Device: {device}")

forecaster = TimesFM3Evaluator(
    config
)


# ---------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------

outputs = list(
    forecaster.predict_batch(
        contexts=[
            context
        ],
        horizon=FORECAST_STEPS,
        return_quantiles=True,
        use_symmetric_averaging=False,
    )
)


if len(outputs) != 1:
    raise RuntimeError(
        f"Expected 1 forecast output, "
        f"received {len(outputs)}."
    )


output = outputs[0]

prediction = np.asarray(
    output.forecast,
    dtype=float,
)

quantiles = np.asarray(
    output.quantiles,
    dtype=float,
)


print()
print(
    "Point forecast shape:",
    prediction.shape,
)

print(
    "Quantile forecast shape:",
    quantiles.shape,
)


assert prediction.shape == (
    FORECAST_STEPS,
)

assert quantiles.shape[0] == (
    FORECAST_STEPS
)

assert quantiles.shape[1] == 9


# ---------------------------------------------------------------------
# TimesFM 3 quantiles:
#
# columns correspond to q10, q20, ..., q90.
# ---------------------------------------------------------------------

q10 = quantiles[:, 0]
q50 = quantiles[:, 4]
q90 = quantiles[:, 8]


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

error = (
    prediction
    - truth
)

mae = float(
    np.mean(
        np.abs(error)
    )
)

rmse = float(
    np.sqrt(
        np.mean(
            error ** 2
        )
    )
)

coverage_80 = float(
    np.mean(
        (
            truth >= q10
        )
        &
        (
            truth <= q90
        )
    )
)

negative_fraction = float(
    np.mean(
        prediction < 0
    )
)


print()
print("=" * 76)
print("Smoke-test result")
print("=" * 76)

print()
print(f"MAE:                  {mae:.4f}")
print(f"RMSE:                 {rmse:.4f}")
print(f"80% interval coverage:{coverage_80:8.3f}")
print(f"Negative forecasts:   {negative_fraction:8.3f}")

print()
print("First 10 predictions:")

display = pd.DataFrame(
    {
        "truth": truth[:10],
        "prediction": prediction[:10],
        "q10": q10[:10],
        "q50": q50[:10],
        "q90": q90[:10],
    }
)

print(
    display
    .round(3)
    .to_string(
        index=False
    )
)
