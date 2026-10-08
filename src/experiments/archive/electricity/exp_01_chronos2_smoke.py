from pathlib import Path

import pandas as pd
import torch

from chronos import Chronos2Pipeline


RAW_PATH = Path(
    "data/raw/electricity/LD2011_2014.txt"
)

PROTOCOL_PATH = Path(
    "data/processed/electricity/"
    "electricity_evaluation_origins.csv"
)

MODEL_ID = "amazon/chronos-2"

CONTEXT_STEPS = 384
FORECAST_STEPS = 96


# ---------------------------------------------------------------------
# Choose first validation case
# ---------------------------------------------------------------------

protocol = pd.read_csv(
    PROTOCOL_PATH,
    parse_dates=[
        "eval_origin",
        "context_start",
        "context_end",
        "target_start",
        "target_end",
    ],
)

case = (
    protocol[
        protocol["split"] == "validation"
    ]
    .iloc[0]
)

entity = case["entity"]

print("=" * 72)
print("Chronos-2 smoke test")
print("=" * 72)

print()
print(f"Entity:       {entity}")
print(f"Origin:       {case['eval_origin']}")
print(f"Context:      {case['context_start']} -> {case['context_end']}")
print(f"Target:       {case['target_start']} -> {case['target_end']}")


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

raw = raw.set_index(
    timestamp_col
).sort_index()


# ---------------------------------------------------------------------
# Extract exact frozen context and ground truth
# ---------------------------------------------------------------------

context = raw.loc[
    case["context_start"]:
    case["context_end"],
    entity,
]

truth = raw.loc[
    case["target_start"]:
    case["target_end"],
    entity,
]

assert len(context) == CONTEXT_STEPS
assert len(truth) == FORECAST_STEPS


context_df = pd.DataFrame(
    {
        "id": entity,
        "timestamp": context.index,
        "target": context.to_numpy(
            dtype=float
        ),
    }
)


# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print(f"Loading {MODEL_ID}")
print(f"Device: {device}")

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
    freq="15min",
)


# ---------------------------------------------------------------------
# Add ground truth
# ---------------------------------------------------------------------

truth_df = pd.DataFrame(
    {
        "timestamp": truth.index,
        "truth": truth.to_numpy(
            dtype=float
        ),
    }
)

pred_df = pred_df.merge(
    truth_df,
    on="timestamp",
    how="left",
    validate="one_to_one",
)

assert len(pred_df) == FORECAST_STEPS
assert pred_df["truth"].notna().all()


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

error = (
    pred_df["predictions"]
    - pred_df["truth"]
)

mae = (
    error.abs().mean()
)

rmse = (
    (error ** 2).mean()
    ** 0.5
)

coverage_80 = (
    (
        pred_df["truth"]
        >= pred_df["0.1"]
    )
    &
    (
        pred_df["truth"]
        <= pred_df["0.9"]
    )
).mean()

negative_fraction = (
    pred_df["predictions"]
    < 0
).mean()


print()
print("=" * 72)
print("Smoke-test result")
print("=" * 72)

print()
print(f"Rows:                 {len(pred_df)}")
print(f"MAE:                  {mae:.4f}")
print(f"RMSE:                 {rmse:.4f}")
print(f"80% interval coverage:{coverage_80:8.3f}")
print(f"Negative forecasts:   {negative_fraction:8.3f}")

print()
print(
    pred_df[
        [
            "timestamp",
            "truth",
            "predictions",
            "0.1",
            "0.5",
            "0.9",
        ]
    ]
    .head(10)
    .round(3)
    .to_string(index=False)
)
