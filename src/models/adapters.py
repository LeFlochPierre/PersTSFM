from dataclasses import dataclass 
import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------
# Forecast I/O
# ---------------------------------------------------------------------

@dataclass
class ForecastInput:
    case_id: str
    timestamps: pd.DatetimeIndex
    values: np.ndarray


@dataclass
class ForecastOutput:
    case_id: str
    prediction: np.ndarray
    q10: np.ndarray
    q50: np.ndarray
    q90: np.ndarray


# ---------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------

class ForecastModel:

    name = "base"

    def predict_batch(
        self,
        inputs,
        horizon,
        resolution_minutes,
        context_steps,
        batch_size,
    ):
        raise NotImplementedError


# ---------------------------------------------------------------------
# Chronos-2
# ---------------------------------------------------------------------

class Chronos2Model(
    ForecastModel
):

    name = "chronos2"

    def __init__(
        self,
        model_id=(
            "amazon/chronos-2"
        ),
    ):
        from chronos import (
            Chronos2Pipeline
        )

        self.model_id = model_id

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.device = device

        self.pipeline = (
            Chronos2Pipeline
            .from_pretrained(
                model_id,
                device_map=device,
            )
        )

    def predict_batch(
        self,
        inputs,
        horizon,
        resolution_minutes,
        context_steps,
        batch_size,
    ):
        frames = []

        for item in inputs:

            frames.append(
                pd.DataFrame(
                    {
                        "id": (
                            item.case_id
                        ),
                        "timestamp": (
                            item.timestamps
                        ),
                        "target": (
                            item.values
                        ),
                    }
                )
            )

        context_df = pd.concat(
            frames,
            ignore_index=True,
        )

        pred_df = (
            self.pipeline.predict_df(
                context_df,
                prediction_length=(
                    horizon
                ),
                quantile_levels=[
                    0.1,
                    0.5,
                    0.9,
                ],
                id_column="id",
                timestamp_column=(
                    "timestamp"
                ),
                target="target",
                context_length=(
                    context_steps
                ),
                batch_size=(
                    batch_size
                ),
                freq=(
                    f"{resolution_minutes}"
                    "min"
                ),
            )
        )

        outputs = []

        for item in inputs:

            part = (
                pred_df[
                    pred_df[
                        "id"
                    ] == item.case_id
                ]
                .sort_values(
                    "timestamp"
                )
                .reset_index(
                    drop=True
                )
            )

            if len(part) != horizon:
                raise RuntimeError(
                    f"{item.case_id}: "
                    f"Chronos returned "
                    f"{len(part)} rows, "
                    f"expected {horizon}."
                )

            outputs.append(
                ForecastOutput(
                    case_id=(
                        item.case_id
                    ),
                    prediction=(
                        part[
                            "predictions"
                        ]
                        .to_numpy(
                            dtype=float
                        )
                    ),
                    q10=(
                        part[
                            "0.1"
                        ]
                        .to_numpy(
                            dtype=float
                        )
                    ),
                    q50=(
                        part[
                            "0.5"
                        ]
                        .to_numpy(
                            dtype=float
                        )
                    ),
                    q90=(
                        part[
                            "0.9"
                        ]
                        .to_numpy(
                            dtype=float
                        )
                    ),
                )
            )

        return outputs


# ---------------------------------------------------------------------
# TimesFM 3
# ---------------------------------------------------------------------

class TimesFM3Model(
    ForecastModel
):

    name = "timesfm3"

    def __init__(
        self,
        model_id=(
            "google/"
            "timesfm-3.0-pytorch"
        ),
        batch_size=16,
    ):
        from timesfm3 import (
            ModelConfig,
            TimesFM3Evaluator,
        )

        self.model_id = model_id

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.device = device

        config = ModelConfig(
            checkpoint_path=(
                model_id
            ),
            per_core_batch_size=(
                batch_size
            ),
            device=device,
        )

        self.forecaster = (
            TimesFM3Evaluator(
                config
            )
        )

    def predict_batch(
        self,
        inputs,
        horizon,
        resolution_minutes,
        context_steps,
        batch_size,
    ):
        contexts = [
            np.asarray(
                item.values,
                dtype=np.float32,
            )
            for item in inputs
        ]

        raw_outputs = list(
            self.forecaster.predict_batch(
                contexts=contexts,
                horizon=horizon,
                return_quantiles=True,
                use_symmetric_averaging=False,
            )
        )

        if (
            len(raw_outputs)
            != len(inputs)
        ):
            raise RuntimeError(
                "TimesFM returned "
                f"{len(raw_outputs)} "
                f"outputs for "
                f"{len(inputs)} inputs."
            )

        outputs = []

        for item, raw in zip(
            inputs,
            raw_outputs,
        ):

            prediction = np.asarray(
                raw.forecast,
                dtype=float,
            )

            quantiles = np.asarray(
                raw.quantiles,
                dtype=float,
            )

            if prediction.shape != (
                horizon,
            ):
                raise RuntimeError(
                    f"{item.case_id}: "
                    "unexpected TimesFM "
                    "point shape "
                    f"{prediction.shape}."
                )

            if quantiles.shape != (
                horizon,
                9,
            ):
                raise RuntimeError(
                    f"{item.case_id}: "
                    "unexpected TimesFM "
                    "quantile shape "
                    f"{quantiles.shape}."
                )

            outputs.append(
                ForecastOutput(
                    case_id=(
                        item.case_id
                    ),
                    prediction=(
                        prediction
                    ),
                    q10=(
                        quantiles[
                            :,
                            0,
                        ]
                    ),
                    q50=(
                        quantiles[
                            :,
                            4,
                        ]
                    ),
                    q90=(
                        quantiles[
                            :,
                            8,
                        ]
                    ),
                )
            )

        return outputs


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------

def build_model(
    model_key,
    batch_size,
):
    if model_key == "chronos2":
        return Chronos2Model()

    if model_key == "timesfm3":
        return TimesFM3Model(
            batch_size=batch_size
        )

    raise ValueError(
        f"Unknown model "
        f"'{model_key}'. "
        "Available models: "
        "chronos2, timesfm3"
    )
