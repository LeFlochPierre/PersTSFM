from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# VitalDB
# ---------------------------------------------------------------------

vital_in = Path(
    "data/processed/vitaldb/"
    "vitaldb_evaluation_origins.csv"
)

vital_out = Path(
    "data/processed/vitaldb/"
    "vitaldb_context_protocol.csv"
)

vital = pd.read_csv(
    vital_in
)

base = pd.Timestamp(
    "2000-01-01 00:00:00"
)

# origin_idx is the first forecasted minute.
vital["eval_origin"] = (
    base
    + pd.to_timedelta(
        vital["origin_idx"] - 1,
        unit="min",
    )
)

vital["target_start"] = (
    base
    + pd.to_timedelta(
        vital["origin_idx"],
        unit="min",
    )
)

vital["target_end"] = (
    vital["target_start"]
    + pd.Timedelta(
        minutes=14
    )
)

vital.to_csv(
    vital_out,
    index=False,
)


# ---------------------------------------------------------------------
# Solar
# ---------------------------------------------------------------------

solar_in = Path(
    "data/processed/solar/"
    "solar_evaluation_origins.csv"
)

solar_out = Path(
    "data/processed/solar/"
    "solar_context_protocol.csv"
)

solar = pd.read_csv(
    solar_in
)

solar["origin_time"] = pd.to_datetime(
    solar["origin_time"],
    errors="raise",
)

# origin_time is the first forecasted
# 10-minute sample.
solar["eval_origin"] = (
    solar["origin_time"]
    - pd.Timedelta(
        minutes=10
    )
)

solar["target_start"] = (
    solar["origin_time"]
)

solar["target_end"] = (
    solar["target_start"]
    + pd.Timedelta(
        minutes=50
    )
)

solar.to_csv(
    solar_out,
    index=False,
)


print("Created:")
print(vital_out)
print(solar_out)

print()
print(
    "VitalDB rows:",
    len(vital),
)

print(
    "Solar rows:",
    len(solar),
)
